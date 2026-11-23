import base64
import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from google import genai
from google.auth.credentials import Credentials
from langchain_core.messages import AIMessage

from app.api import create_app
from app.telemetry import LangSmithSink
from test_chat import ANSWER, API_KEY, DATABASE, SQL, WORKSPACE, ScriptedModel, tool_call


TRACE_KEY = "trace-content-fixture-key"
HEADERS = {"X-API-Key": API_KEY}
SYNTHETIC_QUESTION = (
    "¿Cuál fue la ciudad con el mayor total de ventas en pesos chilenos para pedidos "
    "completados entre el 1 de enero y el 31 de marzo de 2026? Indicá la ciudad y el importe exacto."
)
REVISION = "fastapi-langgraph-sql-test"


class ChatTraceTests(unittest.TestCase):
    def client(self, *, content=True, model=None, tracing=True, vertex=False):
        self.batches = []

        def capture(request):
            self.assertEqual(str(request.url), "https://api.smith.langchain.com/runs/batch")
            self.assertEqual(request.headers["x-tenant-id"], WORKSPACE)
            self.batches.append(json.loads(request.content)["post"])
            return httpx.Response(202)

        env = {
            "SQL_AGENT_API_KEY": API_KEY,
            "SQL_AGENT_TRACE_SYNTHETIC_CONTENT": str(content).lower(),
            "SQL_AGENT_VERTEX_PROJECT": "fixture-project" if vertex else "",
            "SQL_AGENT_TRACE_ENABLED": "false",
            "K_REVISION": REVISION,
        }
        sink = LangSmithSink("fixture-only", WORKSPACE, TRACE_KEY, transport=httpx.MockTransport(capture)) if tracing else None
        with patch.dict("os.environ", env):
            return self.enterContext(TestClient(create_app(model=model, sink=sink)))

    def assert_tags(self, runs, mode):
        common = {"service:fastapi-langgraph-sql", "environment:production", "route:chat",
                  "dataset:synthetic-sales", f"trace:{mode}"}
        specific = {
            "chat": {"component:fastapi"},
            "langgraph": {"component:langgraph"},
            "model": {"component:vertex-ai", "provider:vertex-ai", "node:model"},
            "sql": {"component:langgraph", "node:sql"},
            "query_sales": {"component:sqlite", "node:sql", "tool:query_sales"},
        }
        for run in runs:
            expected = common | specific[run["name"]]
            self.assertCountEqual(run["tags"], expected)
            metadata = run["extra"]["metadata"]
            self.assertEqual(
                {key: metadata[key] for key in ("route", "dataset", "trace_mode", "revision")},
                {"route": "/chat", "dataset": "synthetic-sales", "trace_mode": mode, "revision": REVISION},
            )

    def test_normal_chat_is_the_only_post_route_and_still_requires_key(self):
        model = ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER))
        client = self.client(content=False, model=model)
        self.assertEqual(set(client.get("/openapi.json").json()["paths"]), {"/health", "/chat"})
        self.assertEqual(client.post("/chat/demo", headers=HEADERS).status_code, 404)
        self.assertEqual(client.post("/chat", json={"question": SYNTHETIC_QUESTION}).status_code, 401)
        self.assertFalse(model.prompts or self.batches)
        reply = client.post("/chat", headers=HEADERS, json={"question": SYNTHETIC_QUESTION})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(self.batches[0][0]["inputs"], {"question_length": len(SYNTHETIC_QUESTION)})
        self.assert_tags(self.batches[0], "structural")

    def test_absent_sink_or_changed_fixture_does_not_block_normal_chat(self):
        body = {"question": SYNTHETIC_QUESTION}
        client = self.client(model=ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER)), tracing=False)
        reply = client.post("/chat", headers=HEADERS, json=body)
        self.assertEqual(reply.status_code, 200)
        self.assertNotIn("x-trace-id", reply.headers)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            fixture = root / "data/sales.sqlite3"
            fixture.write_bytes(DATABASE.read_bytes())
            with patch("app.api.ROOT", root):
                client = self.client(model=ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER)))
            fixture.write_bytes(DATABASE.read_bytes() + b" ")
            reply = client.post("/chat", headers=HEADERS, json=body)
            self.assertEqual(reply.status_code, 200)
        self.assertEqual(self.batches[0][0]["inputs"], {"question_length": len(SYNTHETIC_QUESTION)})
        self.assert_tags(self.batches[0], "structural")

    def test_normal_chat_tags_and_exports_approved_content_with_the_real_adapter(self):
        requests = []
        summary = "Aggregate the completed orders by city."
        signature = base64.b64encode(b"opaque-gemini-thought-for-sdk-only").decode()

        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                parts = [
                    {"text": f"{summary} {API_KEY} {TRACE_KEY}", "thought": True},
                    {"functionCall": {"name": "query_sales", "args": {"sql": SQL}}, "thoughtSignature": signature},
                ]
                usage = {"promptTokenCount": 41, "candidatesTokenCount": 9, "thoughtsTokenCount": 6,
                         "cachedContentTokenCount": 3, "totalTokenCount": 56}
            else:
                self.assertEqual(len(requests), 2)
                parts = [{"text": ANSWER}]
                usage = {"promptTokenCount": 58, "candidatesTokenCount": 7, "totalTokenCount": 65}
            return httpx.Response(200, json={
                "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": usage,
            })

        class FixtureCredential(Credentials):
            def refresh(self, request):
                self.token = "fixture-adc-only"

        transport = self.enterContext(httpx.Client(transport=httpx.MockTransport(respond)))

        def sdk_client(**kwargs):
            kwargs["http_options"] = kwargs["http_options"].model_copy(update={"httpx_client": transport})
            return genai.Client(**kwargs)

        with patch("app.api.google.auth.default", return_value=(FixtureCredential(), None)), patch(
            "langchain_google_genai.chat_models.Client", side_effect=sdk_client,
        ):
            reply = self.client(vertex=True).post("/chat", headers=HEADERS, json={"question": SYNTHETIC_QUESTION})

        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()["answer"], ANSWER)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r["generationConfig"]["thinkingConfig"]["include_thoughts"] for r in requests))
        self.assertIn(signature, json.dumps(requests[1]))
        self.assertEqual(len(self.batches), 1)
        root, graph, first, node, tool, last = self.batches[0]
        self.assertEqual([r["name"] for r in self.batches[0]], ["chat", "langgraph", "model", "sql", "query_sales", "model"])
        self.assertEqual(root["id"], reply.headers["x-trace-id"])
        self.assertEqual(root["inputs"]["question"], SYNTHETIC_QUESTION)
        self.assertEqual(root["outputs"]["answer"], ANSWER)
        self.assertEqual(graph["inputs"]["entry_node"], "model")
        self.assert_tags(self.batches[0], "synthetic-content")
        system, user = first["inputs"]["messages"]
        self.assertEqual((system["role"], user["role"]), ("system", "user"))
        self.assertEqual(system["content"][0]["text"], requests[0]["systemInstruction"]["parts"][0]["text"])
        self.assertEqual(user["content"][0]["text"], SYNTHETIC_QUESTION)
        expected_prompt = sha256(system["content"][0]["text"].encode()).hexdigest()
        for run in (first, last):
            metadata = run["extra"]["metadata"]
            self.assertEqual(metadata["ls_model_name"], "gemini-3.5-flash-lite")
            self.assertEqual(metadata["prompt_sha256"], expected_prompt)
        blocks = first["outputs"]["messages"][0]["content"]
        self.assertIn(summary, next(b["text"] for b in blocks if b["type"] == "reasoning"))
        call = next(b for b in blocks if b["type"] == "tool_call")
        self.assertEqual(call["args"], {"sql": SQL})
        self.assertEqual(first["total_tokens"], 56)
        self.assertEqual(first["outputs"]["usage_metadata"], {
            "input_tokens": 41, "output_tokens": 15, "total_tokens": 56,
            "input_token_details": {"cache_read": 3}, "output_token_details": {"reasoning": 6},
        })
        self.assertEqual(tool["parent_run_id"], node["id"])
        self.assertEqual(tool["inputs"]["sql"], SQL)
        self.assertEqual(tool["outputs"]["rows"], [["Temuco", 633700]])
        tool_message = last["inputs"]["messages"][-1]
        self.assertEqual(tool_message["tool_call_id"], call["id"])
        self.assertIn("633700", tool_message["content"][0]["text"])
        self.assertEqual(last["outputs"]["messages"], [{"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}])
        self.assertNotIn("output_token_details", last["outputs"]["usage_metadata"])
        for excluded in (signature, "thoughtSignature", API_KEY, TRACE_KEY, "fixture-adc-only"):
            self.assertNotIn(excluded, json.dumps(self.batches))

    def test_unapproved_question_remains_structural_with_a_caller_trace_header(self):
        client = self.client(model=ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER)))
        question = f"{SYNTHETIC_QUESTION} Caller marker."
        reply = client.post("/chat", headers={**HEADERS, "X-Trace-Mode": "synthetic-content"}, json={"question": question})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(self.batches[0][0]["inputs"], {"question_length": len(question)})
        self.assert_tags(self.batches[0], "structural")
        for run in self.batches[0]:
            self.assertNotIn("messages", run["inputs"])
            self.assertNotIn("messages", run["outputs"])
        encoded = json.dumps(self.batches, ensure_ascii=False)
        for excluded in (question, "Caller marker.", "SELECT", "Temuco", "633700", "trace:synthetic-content"):
            self.assertNotIn(excluded, encoded)
