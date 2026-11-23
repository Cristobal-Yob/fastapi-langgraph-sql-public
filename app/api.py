"""FastAPI entry point for authenticated synthetic sales questions."""

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import google.auth
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, StringConstraints

from .agent import ROOT, ChatResult, answer_question, build_graph
from .query import SalesDatabase
from .telemetry import LangSmithSink, Trace


SYNTHETIC_TRACE_QUESTION = (
    "¿Cuál fue la ciudad con el mayor total de ventas en pesos chilenos para pedidos "
    "completados entre el 1 de enero y el 31 de marzo de 2026? Indicá la ciudad y el importe exacto."
)
SYNTHETIC_DATABASE_SHA256 = "c9febbdaf539c81c79174e4477b5ea9ccb13b61cf2904207e132938acf8117df"


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    vertex_project: str | None = None
    vertex_location: str = "global"
    model_id: str = "gemini-3.5-flash-lite"
    trace_synthetic_content: bool = False

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.environ.get("SQL_AGENT_API_KEY", ""),
            vertex_project=os.environ.get("SQL_AGENT_VERTEX_PROJECT") or None,
            vertex_location=os.environ.get("SQL_AGENT_VERTEX_LOCATION", "global"),
            model_id=os.environ.get("SQL_AGENT_MODEL", "gemini-3.5-flash-lite"),
            trace_synthetic_content=os.environ.get("SQL_AGENT_TRACE_SYNTHETIC_CONTENT") == "true",
        )


class Ask(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)]


def vertex_model(settings: Settings):
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return ChatGoogleGenerativeAI(
        model=settings.model_id, vertexai=True, project=settings.vertex_project,
        location=settings.vertex_location, credentials=credentials, google_api_key=None,
        max_output_tokens=700, timeout=15, max_retries=0,
    )


def configured_sink() -> LangSmithSink | None:
    if os.environ.get("SQL_AGENT_TRACE_ENABLED") != "true":
        return None
    values = [os.environ.get(name) for name in (
        "SQL_AGENT_LANGSMITH_PROJECT", "SQL_AGENT_LANGSMITH_WORKSPACE_ID", "SQL_AGENT_LANGSMITH_API_KEY",
    )]
    if not all(values):
        raise RuntimeError("The LangSmith destination is incomplete.")
    return LangSmithSink(*values)


def _is_shipped_fixture(path: Path) -> bool:
    try:
        return sha256(path.read_bytes()).hexdigest() == SYNTHETIC_DATABASE_SHA256
    except OSError:
        return False


def create_app(*, settings: Settings | None = None, model=None, sink: LangSmithSink | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        config = settings or Settings.from_env()
        database_path = ROOT / "data" / "sales.sqlite3"
        database = SalesDatabase(database_path)
        database.query("SELECT count(*) FROM orders")
        chat_model = model
        if chat_model is None and config.api_key and config.vertex_project:
            chat_model = vertex_model(config)
        application.state.api_key = config.api_key
        application.state.graph = build_graph(chat_model, database, config.model_id) if chat_model else None
        application.state.sink = sink if sink is not None else configured_sink()
        application.state.revision = os.environ.get("K_REVISION")
        application.state.trace_synthetic_content = config.trace_synthetic_content
        application.state.database_path = database_path
        try:
            yield
        finally:
            if application.state.sink is not None:
                application.state.sink.close()

    app = FastAPI(title="SQL Agent", lifespan=lifespan)
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(request: Request, supplied: Annotated[str | None, Security(header)]):
        allowed = request.app.state.api_key
        if not allowed:
            raise HTTPException(503, "The SQL agent is not configured.")
        if not supplied or not secrets.compare_digest(supplied.encode(), allowed.encode()):
            raise HTTPException(401, "A valid API key is required.")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"detail": "Provide a non-empty question of at most 1500 characters."}, status_code=422)

    @app.get("/health")
    def health(request: Request):
        state = request.app.state
        return {"status": "ok", "configured": bool(state.api_key and state.graph is not None),
                "tracing": state.sink is not None}

    @app.post("/chat", dependencies=[Depends(require_key)])
    def chat(body: Ask, request: Request, response: Response) -> ChatResult:
        state = request.app.state
        if state.graph is None:
            raise HTTPException(503, "The SQL agent is not configured.")
        question = body.question
        capture_content = (
            state.trace_synthetic_content
            and state.sink is not None
            and question == SYNTHETIC_TRACE_QUESTION
            and _is_shipped_fixture(state.database_path)
        )
        trace = Trace(state.revision, capture_content=capture_content)
        result = None
        try:
            with trace.span("chat", "chain", {"question_length": len(question)}) as span:
                if capture_content:
                    span.inputs["question"] = question
                result = answer_question(state.graph, question, trace, span)
                span.outputs.update(status=result.status, query_count=len(result.queries))
                if capture_content:
                    span.outputs.update(result.model_dump())
        except Exception:
            # Provider or query internals may include request data.
            result = None
        headers = {}
        if state.sink is not None and state.sink.emit(trace, redact=(state.api_key,) if capture_content else ()):
            headers["X-Trace-Id"] = trace.root_id
        if result is None:
            raise HTTPException(502, "The SQL agent could not complete this request.", headers=headers)
        response.headers.update(headers)
        return result

    return app


app = create_app()
