import json
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import httpx
from fastapi.testclient import TestClient
from google import genai
from google.auth.credentials import Credentials
from google.genai.types import HttpOptions
from langchain_core.messages import AIMessage, ToolMessage

from app.api import Settings, create_app
from app.query import QueryRejected, SalesDatabase
from app.telemetry import LangSmithSink


DATABASE = Path(__file__).resolve().parents[1] / "data" / "sales.sqlite3"
QUESTION = "¿Qué ciudad lideró las ventas completadas entre enero y marzo de 2026?"
SQL = """
SELECT c.city, SUM(i.quantity * i.unit_price_clp) AS sales_clp
FROM customers AS c
JOIN orders AS o ON c.customer_id = o.customer_id
JOIN order_items AS i ON o.order_id = i.order_id
WHERE o.status = 'completed' AND o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'
GROUP BY c.city
ORDER BY sales_clp DESC
LIMIT 1
""".strip()
ANSWER = "Temuco, con CLP 633.700."
API_KEY = "local-test-only"
WORKSPACE = "274ffb31-b9dc-4c84-9b87-93906043740b"
USAGE = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}


def tool_call(sql, call_id="query-1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "query_sales", "args": {"sql": sql}, "id": call_id}],
        usage_metadata=USAGE,
    )


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def bind_tools(self, tools):
        assert [tool.name for tool in tools] == ["query_sales"]
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class QueryTests(unittest.TestCase):
    def test_only_reads_from_sales_tables(self):
        db = SalesDatabase(DATABASE)
        original = DATABASE.read_bytes()
        for invalid in (
            "UPDATE orders SET status = 'canceled'",
            "SELECT name FROM sqlite_schema",
            "SELECT 42",
            "SELECT count(*) FROM orders; SELECT count(*) FROM orders",
        ):
            with self.subTest(statement=invalid), self.assertRaises(QueryRejected):
                db.query(invalid)
        self.assertEqual(db.query("WITH sales AS (SELECT * FROM orders) SELECT count(*) FROM sales")["rows"], [[24]])
        self.assertEqual(DATABASE.read_bytes(), original)

    def test_bounds_rows_and_expensive_aggregates(self):
        db = SalesDatabase(DATABASE)
        statement = "SELECT a.order_id, b.order_id FROM orders a CROSS JOIN orders b"
        exact = db.query(statement + " LIMIT 50")
        self.assertEqual(len(exact["rows"]), 50)
        self.assertFalse(exact["truncated"])
        result = db.query(statement + " LIMIT 51")
        self.assertEqual(len(result["rows"]), 50)
        self.assertTrue(result["truncated"])
        bounded = "SELECT count(*) FROM orders".ljust(4096)
        self.assertEqual(db.query(bounded)["rows"], [[24]])
        with self.assertRaisesRegex(QueryRejected, "too long"):
            db.query(bounded + " ")
        with self.assertRaisesRegex(QueryRejected, "budget"):
            db.query("SELECT count(*) FROM order_items a, order_items b, order_items c, order_items d")


class ChatTests(unittest.TestCase):
    def client(self, model, sink=None):
        client = TestClient(create_app(settings=Settings(api_key=API_KEY), model=model, sink=sink))
        self.addCleanup(client.close)
        return self.enterContext(client)

    def post(self, client):
        return client.post("/chat", headers={"X-API-Key": API_KEY}, json={"question": QUESTION})

    def test_key_required_before_model_or_trace(self):
        model = ScriptedModel(tool_call(SQL))
        hits = []

        def intercept(request):
            hits.append(request)
            return httpx.Response(202)

        sink = LangSmithSink("sql-production-test", WORKSPACE, "trace-dummy", transport=httpx.MockTransport(intercept))
        client = self.client(model, sink)
        self.assertEqual(client.get("/health").json()["configured"], True)
        for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}):
            denied = client.post("/chat", json={"question": QUESTION}, headers=headers)
            self.assertEqual(denied.status_code, 401)
            self.assertNotIn("x-trace-id", denied.headers)
        self.assertEqual(model.prompts, [])
        self.assertEqual(hits, [])

    def test_real_sql_and_full_trace_with_no_content_export(self):
        batches = []

        def intercept(request):
            self.assertEqual(str(request.url), "https://api.smith.langchain.com/runs/batch")
            self.assertEqual(request.headers["x-tenant-id"], WORKSPACE)
            self.assertEqual(request.headers["x-api-key"], "trace-dummy")
            batches.append(json.loads(request.content))
            return httpx.Response(202)

        model = ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER, usage_metadata=USAGE))
        sink = LangSmithSink("sql-production-test", WORKSPACE, "trace-dummy", transport=httpx.MockTransport(intercept))
        with patch.dict("os.environ", {
            "LANGSMITH_TRACING": "true", "K_REVISION": "fastapi-langgraph-sql-test",
        }):
            response = self.post(self.client(model, sink))
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["answer"], ANSWER)
        self.assertEqual(data["status"], "answered")
        self.assertEqual(data["queries"], [{
            "sql": SQL, "columns": ["city", "sales_clp"], "rows": [["Temuco", 633700]], "truncated": False,
        }])
        self.assertIsInstance(model.prompts[1][-1], ToolMessage)
        self.assertIn("633700", model.prompts[1][-1].content)
        self.assertEqual(len(batches), 1)
        batch = batches[0]["post"]
        self.assertEqual([run["name"] for run in batch], ["chat", "langgraph", "model", "sql", "query_sales", "model"])
        root_id = response.headers["x-trace-id"]
        UUID(root_id)
        self.assertEqual(batch[0]["id"], root_id)
        self.assertEqual(batch[1]["parent_run_id"], root_id)
        self.assertTrue(all(batch[i]["parent_run_id"] == batch[1]["id"] for i in (2, 3, 5)))
        self.assertEqual(batch[4]["parent_run_id"], batch[3]["id"])
        self.assertEqual([batch[i]["outputs"]["next_node"] for i in (2, 3, 5)], ["sql", "model", "end"])
        self.assertEqual(batch[1]["inputs"]["entry_node"], "model")
        self.assertTrue(all(run["trace_id"] == root_id and run["session_name"] == "sql-production-test" for run in batch))
        self.assertTrue(all(run["inputs"] and run["outputs"] and run["extra"]["metadata"] for run in batch))
        required = {"service:fastapi-langgraph-sql", "environment:production"}
        self.assertTrue(all(required <= set(run["tags"]) for run in batch))
        self.assertIn("component:fastapi", batch[0]["tags"])
        self.assertIn("node:model", batch[2]["tags"])
        self.assertIn("node:sql", batch[3]["tags"])
        self.assertIn("tool:query_sales", batch[4]["tags"])
        self.assertEqual(batch[2]["outputs"]["usage_metadata"], USAGE)
        self.assertEqual(batch[4]["outputs"]["row_count"], 1)
        encoded = json.dumps(batch, ensure_ascii=False)
        for private in (QUESTION, "SELECT", "Temuco", "633700", API_KEY, "trace-dummy", "Ground the answer"):
            self.assertNotIn(private, encoded)

    def test_no_query_does_not_surface_an_unverified_answer(self):
        response = self.post(self.client(ScriptedModel(AIMessage(content="Invented answer."))))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "no_data", "answer": None, "queries": []})

    def test_sql_rejection_and_budget_do_not_trigger_an_extra_model_call(self):
        model = ScriptedModel(tool_call("SELECT 42", "first"), tool_call(SQL, "second"), tool_call(SQL, "third"))
        response = self.post(self.client(model))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "The SQL agent could not complete this request."})
        self.assertEqual(len(model.prompts), 3)
        self.assertNotIn("SELECT 42", model.prompts[1][-1].content)
        self.assertIn("633700", model.prompts[2][-1].content)

    def test_trace_outage_does_not_replace_a_valid_answer(self):
        def intercept(request):
            return httpx.Response(503)

        model = ScriptedModel(tool_call(SQL), AIMessage(content=ANSWER))
        sink = LangSmithSink("sql-production-test", WORKSPACE, "trace-dummy", transport=httpx.MockTransport(intercept))
        with self.assertLogs("app.telemetry", "WARNING"):
            response = self.post(self.client(model, sink))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], ANSWER)
        self.assertNotIn("x-trace-id", response.headers)

    def test_vertex_failure_has_one_bounded_attempt_and_redacted_diagnostic(self):
        attempts, batches = [], []

        class FixtureCredential(Credentials):
            def refresh(self, request):
                self.token = "fixture-provider-token"

        def reject(request):
            attempts.append(request)
            return httpx.Response(
                503, json={"error": {"code": 503, "message": "unapproved provider details", "status": "UNAVAILABLE"}},
            )

        def capture(request):
            batches.append(json.loads(request.content))
            return httpx.Response(202)

        transport = self.enterContext(httpx.Client(transport=httpx.MockTransport(reject)))

        def sdk_client(**kwargs):
            options = kwargs["http_options"]
            kwargs["http_options"] = options.model_copy(update={"httpx_client": transport})
            return genai.Client(**kwargs)

        sink = LangSmithSink("sql-production-test", WORKSPACE, "trace-dummy", transport=httpx.MockTransport(capture))
        settings = Settings(api_key=API_KEY, vertex_project="fixture-project")
        with patch("app.api.google.auth.default", return_value=(FixtureCredential(), None)), patch(
            "langchain_google_genai.chat_models.Client", side_effect=sdk_client,
        ):
            response = self.post(self.enterContext(TestClient(create_app(settings=settings, sink=sink))))

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].extensions["timeout"]["read"], 15)
        self.assertEqual(len(batches), 1)
        model = next(run for run in batches[0]["post"] if run["name"] == "model")
        self.assertEqual(model["outputs"], {"status": "error", "failure_kind": "provider_http", "provider_status": 503})
        self.assertNotIn("unapproved provider details", json.dumps(batches))
        self.assertNotIn("fixture-provider-token", json.dumps(batches))
