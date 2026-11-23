"""Build a new SQLite sales fixture from the companion Excel workbook."""

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import load_workbook


SCHEMA = Path(__file__).resolve().parents[1] / "data" / "schema.sql"
SHEETS = {
    "customers": ("customer_id", "customer_name", "city"),
    "products": ("product_id", "product_name", "category", "list_price_clp"),
    "orders": ("order_id", "customer_id", "order_date", "status"),
    "order_items": ("order_id", "product_id", "quantity", "unit_price_clp"),
}


def sheet_rows(sheet, columns: tuple[str, ...]):
    rows = sheet.iter_rows()
    header = tuple(cell.value for cell in next(rows, ()))
    if header != columns:
        raise ValueError(f"Unexpected columns on sheet {sheet.title}")

    for row in rows:
        values = tuple(cell.value for cell in row)
        if all(value is None for value in values):
            continue
        if len(values) != len(columns) or any(cell.data_type in {"f", "e"} for cell in row):
            raise ValueError(f"Expected literal values on sheet {sheet.title}")
        if "order_date" in columns:
            day = values[2]
            if isinstance(day, datetime):
                day = day.date()
            if not isinstance(day, date):
                raise ValueError("order_date must be an Excel date")
            values = (*values[:2], day.isoformat(), *values[3:])
        yield values


def import_workbook(source: Path, target: Path) -> int:
    """Publish the database only after every row passes its constraints."""
    if target.exists():
        raise FileExistsError("The destination already exists; choose a new SQLite path.")

    with closing(load_workbook(source, read_only=True)) as workbook:
        if set(workbook.sheetnames) != set(SHEETS):
            raise ValueError("The workbook must contain the four sales sheets.")
        with TemporaryDirectory(prefix=".sales-import-", dir=target.parent) as directory:
            staging = Path(directory) / "sales.sqlite3"
            with closing(sqlite3.connect(staging)) as db:
                db.execute("PRAGMA foreign_keys = ON")
                db.executescript(SCHEMA.read_text(encoding="utf-8"))
                with db:
                    count = 0
                    for name, columns in SHEETS.items():
                        placeholders = ", ".join("?" for _ in columns)
                        insert = f"INSERT INTO {name} ({', '.join(columns)}) VALUES ({placeholders})"
                        count += db.executemany(insert, sheet_rows(workbook[name], columns)).rowcount

            # A hard link atomically publishes the completed file without replacing a target.
            os.link(staging, target)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the synthetic XLSX source.")
    parser.add_argument("database", type=Path, help="New SQLite path; an existing file is never overwritten.")
    args = parser.parse_args()
    try:
        count = import_workbook(args.workbook, args.database)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1
    print(f"Imported {count} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
