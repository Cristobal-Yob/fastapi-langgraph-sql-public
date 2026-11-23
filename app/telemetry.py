"""Send only explicitly projected run data to LangSmith."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator
from uuid import UUID, uuid4

import httpx


LOGGER = logging.getLogger(__name__)
API_URL = "https://api.smith.langchain.com"
SERVICE = "fastapi-langgraph-sql"
DATASET = "synthetic-sales"
SPAN_TAGS: dict[str, tuple[str, ...]] = {
    "chat": ("component:fastapi",),
    "langgraph": ("component:langgraph",),
    "model": ("component:vertex-ai", "provider:vertex-ai", "node:model"),
    "sql": ("component:langgraph", "node:sql"),
    "query_sales": ("component:sqlite", "node:sql", "tool:query_sales"),
}


@dataclass
class Span:
    run: dict

    @property
    def id(self) -> str:
        return self.run["id"]

    @property
    def inputs(self) -> dict:
        return self.run["inputs"]

    @property
    def outputs(self) -> dict:
        return self.run["outputs"]

    def record_usage(self, usage: dict | None, *, detailed: bool = False):
        if not usage:
            return
        allowed = ("input_tokens", "output_tokens", "total_tokens")
        counts = {key: usage.get(key) for key in allowed}
        if all(type(value) is int and value >= 0 for value in counts.values()):
            if detailed:
                for group, key in (("input_token_details", "cache_read"), ("output_token_details", "reasoning")):
                    breakdown = usage.get(group)
                    count = breakdown.get(key) if isinstance(breakdown, dict) else None
                    if type(count) is int and count >= 0:
                        counts[group] = {key: count}
            self.outputs["usage_metadata"] = counts
            self.run.update(
                prompt_tokens=counts["input_tokens"],
                completion_tokens=counts["output_tokens"],
                total_tokens=counts["total_tokens"],
            )


class Trace:
    def __init__(self, revision: str | None, *, capture_content: bool = False):
        self.capture_content = capture_content
        self.root_id = str(uuid4())
        self.runs: list[dict] = []
        environment = "production" if revision else "local"
        mode = "synthetic-content" if capture_content else "structural"
        self.metadata = {
            "service": SERVICE, "environment": environment, "trace_mode": mode,
            "route": "/chat", "dataset": DATASET,
        }
        self.tags = (
            f"service:{SERVICE}", f"environment:{environment}", "route:chat",
            f"dataset:{DATASET}", f"trace:{mode}",
        )
        if revision:
            self.metadata["revision"] = revision

    @contextmanager
    def span(self, name: str, kind: str, inputs: dict, *, parent: Span | None = None,
             metadata: dict | None = None) -> Iterator[Span]:
        start = datetime.now(timezone.utc)
        run_id = str(uuid4()) if parent else self.root_id
        order = start.strftime("%Y%m%dT%H%M%S%fZ") + run_id
        run = {
            "id": run_id, "trace_id": self.root_id, "name": name, "run_type": kind,
            "start_time": start.isoformat(), "dotted_order": order,
            "inputs": inputs, "outputs": {}, "extra": {"metadata": self.metadata | (metadata or {})},
            "tags": [*self.tags, *SPAN_TAGS.get(name, ())],
        }
        if parent:
            run.update(parent_run_id=parent.id, dotted_order=f"{parent.run['dotted_order']}.{order}")
        self.runs.append(run)
        try:
            yield Span(run)
        except Exception:
            run["error"] = "operation_failed"
            run["outputs"]["status"] = "error"
            raise
        finally:
            run["end_time"] = datetime.now(timezone.utc).isoformat()


def _mask_values(value, secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_mask_values(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _mask_values(item, secrets) for key, item in value.items()}
    return value


class LangSmithSink:
    def __init__(self, project: str, workspace: str, api_key: str, *, transport=None):
        self.project = project
        self.client = httpx.Client(
            transport=transport, timeout=3, follow_redirects=False, trust_env=False,
            headers={"x-tenant-id": str(UUID(workspace)), "x-api-key": api_key},
        )

    def close(self):
        self.client.close()

    def emit(self, trace: Trace, *, redact: tuple[str, ...] = ()) -> bool:
        runs = [{**run, "session_name": self.project} for run in trace.runs]
        if trace.capture_content:
            secrets = tuple(sorted(filter(None, (*redact, self.client.headers["x-api-key"])), key=len, reverse=True))
            for run in runs:
                run["inputs"] = _mask_values(run["inputs"], secrets)
                run["outputs"] = _mask_values(run["outputs"], secrets)
        try:
            result = self.client.post(f"{API_URL}/runs/batch", json={"post": runs, "patch": []})
            if result.is_success:
                return True
        except httpx.HTTPError:
            pass
        LOGGER.warning("LangSmith trace export failed")
        return False
