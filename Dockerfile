FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080

COPY requirements.txt ./
RUN python -m pip --isolated install --no-cache-dir --require-hashes -r requirements.txt

COPY app/__init__.py app/agent.py app/api.py app/query.py app/telemetry.py app/sales_prompt.txt ./app/
COPY data/schema.sql data/sales.sqlite3 ./data/

USER 10001:10001
EXPOSE 8080
CMD ["sh", "-c", "exec python -m uvicorn app.api:app --host 0.0.0.0 --port \"$PORT\" --no-access-log"]
