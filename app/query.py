"""Bounded, read-only execution against the shipped sales tables."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TypedDict


TABLES = frozenset({"customers", "products", "orders", "order_items"})
FUNCTIONS = frozenset({
    "abs", "avg", "coalesce", "count", "date", "datetime", "dense_rank",
    "group_concat", "ifnull", "instr", "length", "like", "lower", "max", "min",
    "nullif", "rank", "replace", "round", "row_number", "strftime", "substr",
    "sum", "total", "trim", "upper",
})
MAX_SQL_BYTES = 4096
MAX_ROWS = 50
MAX_RESULT_BYTES = 16_384
VM_STEP_INTERVAL = 1_000
VM_STEP_BUDGET = 250_000


class QueryRejected(ValueError):
    """A safe error the model can use to revise its query."""


class QueryResult(TypedDict):
    sql: str
    columns: list[str]
    rows: list[list[str | int | float | None]]
    truncated: bool


class SalesDatabase:
    def __init__(self, path: Path):
        self.uri = path.resolve().as_uri() + "?mode=ro&immutable=1"

    def query(self, sql: str) -> QueryResult:
        if not sql.strip() or len(sql.encode("utf-8")) > MAX_SQL_BYTES:
            raise QueryRejected("The SQL statement is empty or too long.")

        read_sales = False
        remaining = VM_STEP_BUDGET

        def authorize(action, first, second, database, _source):
            nonlocal read_sales
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            # SQLite can omit the database on a column-less table read, such as COUNT(*).
            if action == sqlite3.SQLITE_READ and first in TABLES and (
                database == "main" or (database is None and second == "")
            ):
                read_sales = True
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and second in FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        def progress():
            nonlocal remaining
            remaining -= VM_STEP_INTERVAL
            return remaining <= 0

        try:
            with closing(sqlite3.connect(self.uri, uri=True)) as db:
                db.execute("PRAGMA query_only = ON")
                db.execute("PRAGMA trusted_schema = OFF")
                db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RESULT_BYTES)
                db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_BYTES)
                db.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 32)
                db.set_authorizer(authorize)
                db.set_progress_handler(progress, VM_STEP_INTERVAL)
                cursor = db.execute(sql)
                if not read_sales or cursor.description is None:
                    raise QueryRejected("The SELECT must read at least one sales table.")
                columns = [item[0] for item in cursor.description]
                rows = cursor.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as error:
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                raise QueryRejected("The query exceeded its execution budget.") from None
            raise QueryRejected("Use one read-only SELECT on the supplied sales tables.") from None

        result: QueryResult = {
            "sql": sql,
            "columns": columns,
            "rows": [list(row) for row in rows[:MAX_ROWS]],
            "truncated": len(rows) > MAX_ROWS,
        }
        try:
            length = len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError):
            raise QueryRejected("Return ordinary text or numeric columns.") from None
        if length > MAX_RESULT_BYTES:
            raise QueryRejected("The result is too large; select fewer rows or columns.")
        return result
