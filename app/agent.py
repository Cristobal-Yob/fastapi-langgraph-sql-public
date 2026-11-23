"""The two-node, stateless LangGraph tool loop."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import httpx
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context
from pydantic import BaseModel

from .query import QueryRejected, QueryResult, SalesDatabase
from .telemetry import Span, Trace


ROOT = Path(__file__).resolve().parents[1]
MAX_MODEL_CALLS = 3


class AgentState(MessagesState):
    model_calls: int
    queries: list[QueryResult]
    next_node: Literal["sql", "end"]


@dataclass(frozen=True)
class RunContext:
    trace: Trace
    graph_span: Span


class ChatResult(BaseModel):
    status: Literal["answered", "no_data"]
    answer: str | None
    queries: list[QueryResult]


def model_failure(error: Exception) -> dict[str, str | int]:
    cause: BaseException | None = error
    for _ in range(4):
        if isinstance(cause, APIError) and type(cause.code) is int and 400 <= cause.code <= 599:
            return {"failure_kind": "provider_http", "provider_status": cause.code}
        if isinstance(cause, GoogleAuthError):
            return {"failure_kind": "provider_auth"}
        if isinstance(cause, httpx.TimeoutException):
            return {"failure_kind": "provider_timeout"}
        if isinstance(cause, httpx.TransportError):
            return {"failure_kind": "provider_transport"}
        if cause is None:
            break
        cause = cause.__cause__
    return {"failure_kind": "model_error"}


def _trace_message(message: BaseMessage) -> dict:
    """Project visible messages, never the SDK object or its thought signatures."""
    role = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}[message.type]
    blocks = []
    for block in message.content_blocks:
        if block["type"] == "text" and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
        elif block["type"] == "reasoning" and isinstance(block.get("reasoning"), str):
            blocks.append({"type": "reasoning", "text": block["reasoning"]})
    projected = {"role": role, "content": blocks}
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            sql = call["args"].get("sql")
            if call["name"] == "query_sales" and isinstance(sql, str):
                blocks.append({"type": "tool_call", "name": "query_sales", "args": {"sql": sql}, "id": call["id"]})
    elif isinstance(message, ToolMessage):
        projected["tool_call_id"] = message.tool_call_id
    return projected


def build_graph(model, database: SalesDatabase, model_id: str):
    prompt = (ROOT / "app" / "sales_prompt.txt").read_text(encoding="utf-8").strip()
    schema = (ROOT / "data" / "schema.sql").read_text(encoding="utf-8")
    system_content = f"{prompt}\n\nSQLite schema:\n{schema}"
    system = SystemMessage(content=system_content)
    model_metadata = {
        "node": "model", "ls_provider": "google_genai", "ls_model_name": model_id,
        "prompt_sha256": sha256(system_content.encode("utf-8")).hexdigest(),
    }

    @tool
    def query_sales(sql: str) -> QueryResult:
        """Execute one read-only SQLite SELECT and return bounded rows."""
        return database.query(sql)

    bound_model = model.bind_tools([query_sales])

    def model_destination(message: AIMessage, attempt: int) -> Literal["sql", "end"]:
        calls = message.tool_calls
        if message.invalid_tool_calls or len(calls) > 1:
            raise RuntimeError("Expected at most one valid tool call.")
        if not calls:
            return "end"
        if attempt >= MAX_MODEL_CALLS or not calls[0]["id"] or calls[0]["name"] != query_sales.name:
            raise RuntimeError("The next tool call is not permitted.")
        return "sql"

    def model_node(state: AgentState, runtime: Runtime[RunContext]):
        context = runtime.context
        attempt = state["model_calls"] + 1
        capture_content = context.trace.capture_content
        messages = [system, *state["messages"]]
        with context.trace.span(
            "model", "llm", {"attempt": attempt, "messages_count": len(state["messages"])},
            parent=context.graph_span,
            metadata=model_metadata,
        ) as span:
            if capture_content:
                span.inputs["messages"] = [_trace_message(item) for item in messages]
            try:
                message = bound_model.invoke(messages, **({"include_thoughts": True} if capture_content else {}))
            except Exception as error:
                span.outputs.update(model_failure(error))
                raise
            if not isinstance(message, AIMessage):
                raise TypeError("Expected a model response.")
            span.outputs.update(status="returned", tool_call_count=len(message.tool_calls))
            span.record_usage(message.usage_metadata, detailed=capture_content)
            if capture_content:
                span.outputs["messages"] = [_trace_message(message)]
            next_node = model_destination(message, attempt)
            span.outputs["next_node"] = next_node
        return {"model_calls": attempt, "messages": [message], "next_node": next_node}

    def sql_node(state: AgentState, runtime: Runtime[RunContext]):
        context = runtime.context
        call = state["messages"][-1].tool_calls[0]
        attempt = state["model_calls"]
        evidence = None
        with context.trace.span("sql", "chain", {"attempt": attempt}, parent=context.graph_span,
                                metadata={"node": "sql"}) as node:
            with context.trace.span("query_sales", "tool", {"attempt": attempt}, parent=node,
                                    metadata={"store": "sqlite"}) as span:
                args = call["args"]
                try:
                    if set(args) != {"sql"} or not isinstance(args["sql"], str):
                        raise QueryRejected("The tool requires one SQL string.")
                    if context.trace.capture_content:
                        span.inputs.update(sql=args["sql"], tool_call_id=call["id"])
                    evidence = query_sales.invoke(args)
                except QueryRejected as error:
                    span.outputs.update(status="rejected", row_count=0)
                    result = ToolMessage(content=str(error), tool_call_id=call["id"], name=query_sales.name,
                                         status="error")
                else:
                    rows = evidence["rows"]
                    span.outputs.update(status="returned" if rows else "empty", row_count=len(rows),
                                        truncated=evidence["truncated"])
                    if context.trace.capture_content:
                        span.outputs.update(columns=evidence["columns"], rows=rows)
                    result = ToolMessage(
                        content=json.dumps({key: evidence[key] for key in ("columns", "rows", "truncated")},
                                           ensure_ascii=False, allow_nan=False),
                        tool_call_id=call["id"], name=query_sales.name,
                    )
            node.outputs.update(status=span.outputs["status"], next_node="model")
        if evidence is None:
            return {"messages": [result]}
        return {"messages": [result], "queries": [*state["queries"], evidence]}

    builder = StateGraph(AgentState, context_schema=RunContext)
    builder.add_node("model", model_node)
    builder.add_node("sql", sql_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", lambda state: state["next_node"], {"sql": "sql", "end": END})
    builder.add_edge("sql", "model")
    return builder.compile()


def answer_question(graph, question: str, trace: Trace, parent: Span) -> ChatResult:
    with trace.span("langgraph", "chain", {"entry_node": "model", "max_model_calls": MAX_MODEL_CALLS},
                    parent=parent) as span:
        # Do not let ambient SDK settings export the raw graph state.
        with tracing_context(enabled=False):
            state = graph.invoke(
                {"messages": [HumanMessage(content=question)], "model_calls": 0, "queries": []},
                context=RunContext(trace, span),
            )
        queries = state["queries"]
        answer = state["messages"][-1].text.strip() if any(query["rows"] for query in queries) else None
        if answer == "":
            raise RuntimeError("The model returned no text after a successful query.")
        status = "answered" if answer else "no_data"
        span.outputs.update(status=status, model_calls=state["model_calls"], query_count=len(queries))
        return ChatResult(status=status, answer=answer, queries=queries)
