# SQL Agent

Ask a sales question in natural language. The stateless LangGraph model uses one read-only SQLite tool. FastAPI returns the answer, the SQL that ran, and bounded result rows.

Open the local Swagger UI at `http://127.0.0.1:8000/docs` after starting the app below. Use **Authorize** with the key configured for your own deployment. See the [Cloud Run guide](CLOUD_RUN.md) for deployment and secret-handling guidance.

## Run and test

From the repository root with Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Inject runtime configuration using the authorized environment mechanism. [`.env.example`](.env.example) lists the names; the app does not read a dotenv file, copy credentials, or use an API key for Gemini. Set `SQL_AGENT_API_KEY` and `SQL_AGENT_VERTEX_PROJECT` for chat, and run with Application Default Credentials scoped to that project. The default Vertex model is `gemini-3.5-flash-lite` in `global`. Without configured credentials or a project the API does not invoke a model.

- `GET /health` reports local configuration and the presence of the trace sink. It does not invoke a model or certify external credentials.
- `POST /chat` requires `X-API-Key` and JSON such as `{"question":"Which city led completed sales in Q1 2026?"}`. The expected database result is Temuco, CLP 633,700. The response includes each successful statement and its columns, rows and truncation flag. No usable rows returns `no_data` without a fabricated answer.

Each question allows at most three model calls and two tool calls. The tool permits a single statement over the four demo tables, caps results at 50 rows and applies an execution budget. The prompt is versioned in [`app/sales_prompt.txt`](app/sales_prompt.txt).

```sh
.venv/bin/python -m pip install "openpyxl==3.1.5"
.venv/bin/python -m unittest discover -s tests -v
```

Tests run the real SQLite database and FastAPI/LangGraph flow with a scripted model and intercepted trace transport. They do not consume model calls or verify remote ingestion.

## Demo data

[data/synthetic-sales.xlsx](data/synthetic-sales.xlsx) is the source for [data/sales.sqlite3](data/sales.sqlite3). Both contain the same 88 synthetic records. The data covers January through March 2026 and contains no real customer records.

| Sheet / table | Grain | Records |
| --- | --- | ---: |
| `customers` | One row per synthetic customer | 8 |
| `products` | One row per product, with its current list price | 8 |
| `orders` | One row per order, with date, customer, and status | 24 |
| `order_items` | One row per product in an order, with quantity and historical unit price | 48 |

To calculate sales, multiply `order_items.quantity` by `order_items.unit_price_clp` and count only `completed` orders. Amounts are integer Chilean pesos. SQLite stores dates as ISO `YYYY-MM-DD` text. The source workbook uses typed Excel dates and numbers.

## Rebuild the fixture

Run from the repository root with Python 3.12 and SQLite 3.37 or newer:

```sh
.venv/bin/python scripts/import_sales.py data/synthetic-sales.xlsx data/sales-rebuilt.sqlite3
```

The importer requires the workbook's four sheets, enforces key relationships, and refuses to overwrite a database. Tests check the imported rows and a completed-sales aggregate across cities.

## Follow a request in LangSmith

Use the authorized workspace and the agent's production trace destination. Enable explicit tracing with `SQL_AGENT_TRACE_ENABLED=true`, and inject `SQL_AGENT_LANGSMITH_PROJECT` (the resolved tracer session name), `SQL_AGENT_LANGSMITH_WORKSPACE_ID`, and `SQL_AGENT_LANGSMITH_API_KEY`. Automatic content tracing is disabled in the graph.

Open the trace by the `X-Trace-Id` returned by `/chat` after LangSmith accepts export. The header is correlation evidence, not a read-back receipt. One trace contains:

| Run | Inputs and outputs visible in LangSmith |
| --- | --- |
| `chat` | Question length, outcome, query count, service and revision |
| `langgraph` | Entry node, model-call budget, observed model and query counts, outcome |
| `model` node (LLM) | Attempt, message and tool-call counts, selected next node, model/provider and reported input/output/total tokens; safe failure category and HTTP status when available |
| `sql` node (chain) | Attempt, outcome and transition back to `model` |
| `query_sales` tool (inside `sql`) | Attempt, outcome, result count and truncation flag |

A typical query follows `model → sql → model → end`. In LangSmith, expand the `sql` run to see the nested tool.

### Tags on every run

Filter by `service:fastapi-langgraph-sql`, `environment:production` (or `local`), `route:chat`, `dataset:synthetic-sales` and `trace:structural` or `trace:synthetic-content`. Component tags identify FastAPI, LangGraph, Vertex AI and SQLite. Model runs add `node:model` and `provider:vertex-ai`; SQL spans add `node:sql` and the tool adds `tool:query_sales`. Model metadata includes the actual model and the SHA-256 of the system prompt plus schema; Cloud Run revision is recorded when available. Tags never contain caller text.

### Detailed synthetic trace on the normal chat route

By default, `POST /chat` exports counts, outcomes, and metadata, not text. To verify the detailed projection, set `SQL_AGENT_TRACE_SYNTHETIC_CONTENT=true` with LangSmith configured, authorize in Swagger, and make the same `POST /chat` request:

```json
{"question":"¿Cuál fue la ciudad con el mayor total de ventas en pesos chilenos para pedidos completados entre el 1 de enero y el 31 de marzo de 2026? Indicá la ciudad y el importe exacto."}
```

Only this approved synthetic question with the shipped SQLite fingerprint includes content. Other questions still work and stay structural regardless of caller headers. Find the request by `X-Trace-Id` and `trace:synthetic-content`: the root shows the answer; the model runs show the system and user prompts, assistant/tool messages and token details; `sql → query_sales` shows SQL and rows.

Gemini's [optional thought summary](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking) and reasoning-token breakdown are recorded when returned. Hidden reasoning, encrypted thought signatures, raw errors, and configured API key values are not exported. If LangSmith is unavailable, the response can still succeed without a trace ID. Missing usage is not reported as zero.

See LangSmith's [model run and token metadata specification](https://docs.langchain.com/langsmith/log-llm-trace) for how the model spans are rendered.

[Cloud Run deployment and verification](CLOUD_RUN.md)
