# Deploy the SQL agent to Cloud Run

The container serves FastAPI with read-only access to synthetic SQLite data. Deploy it in your own Google Cloud project, using a dedicated runtime service account and your own Vertex AI and LangSmith configuration.

## Configure the service

| Setting | Configuration |
| --- | --- |
| Container | Build the repository's Dockerfile with hash-locked dependencies. Deploy the resulting image by digest. |
| Runtime identity | Grant the runtime service account only the Vertex AI and Secret Manager permissions it needs. |
| API authentication | Supply `SQL_AGENT_API_KEY` as a Secret Manager reference. Clients send it in the `X-API-Key` header, without a `Bearer` prefix. Choose Cloud Run Invoker IAM separately according to your access policy. |
| LangSmith | Configure your own project and workspace using [`.env.example`](.env.example). Keep trace enablement explicit, and supply the LangSmith key through Secret Manager only if tracing is enabled. |

Never put API keys in the repository, image, command arguments, or logs. Cloud Run supplies `PORT` and `K_REVISION`.

## Verify your own deployment

1. With the appropriate Cloud Run access, open `/docs` and check that `/health` returns HTTP 200. Neither step calls the model.
2. Check that `POST /chat` without an application key returns HTTP 401. Then authorize in Swagger and try one approved synthetic sales question. For the shipped Q1 example, the expected result is Temuco, CLP 633,700.
3. If tracing is enabled, use the returned `X-Trace-Id` to check the exact run in your own LangSmith project. A missing trace ID does not imply the model failed. See [the tracing guide](README.md#follow-a-request-in-langsmith) for expected nodes, tools, tokens, and synthetic content controls.
