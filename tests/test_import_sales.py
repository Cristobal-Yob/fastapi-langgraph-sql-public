import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "data" / "synthetic-sales.xlsx"
TABLE_COUNTS = {"customers": 8, "products": 8, "orders": 24, "order_items": 48}


def import_sales(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "import_sales.py"), str(BOOK), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


class ImportSalesTests(unittest.TestCase):
    def test_workbook_rows_and_sales_totals_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sales.sqlite3"
            result = import_sales(database)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("88 records", result.stdout)
            self.assertEqual(result.stderr, "")

            workbook = load_workbook(BOOK, read_only=True)
            self.addCleanup(workbook.close)
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(len(db.execute("PRAGMA foreign_key_list(order_items)").fetchall()), 2)

                for table, expected_count in TABLE_COUNTS.items():
                    workbook_rows = list(workbook[table].values)[1:]
                    if table == "orders":
                        workbook_rows = [
                            (order_id, customer_id, date.date().isoformat(), status)
                            for order_id, customer_id, date, status in workbook_rows
                        ]
                    sql_rows = db.execute(f"SELECT * FROM {table}").fetchall()
                    self.assertEqual(len(sql_rows), expected_count)
                    self.assertCountEqual(sql_rows, workbook_rows)

                sales = db.execute(
                    """
                    SELECT c.city, SUM(i.quantity * i.unit_price_clp) AS sales_clp
                    FROM customers AS c
                    JOIN orders AS o ON o.customer_id = c.customer_id
                    JOIN order_items AS i ON i.order_id = o.order_id
                    WHERE o.status = 'completed'
                      AND o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'
                    GROUP BY c.city
                    ORDER BY sales_clp DESC
                    """
                ).fetchall()
                self.assertEqual(sales, [
                    ("Temuco", 633700),
                    ("Concepción", 499840),
                    ("Valparaíso", 289820),
                    ("Santiago", 179840),
                ])

    def test_import_does_not_replace_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sales.sqlite3"
            database.write_bytes(b"preserve the existing file")
            result = import_sales(database)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("already exists", result.stderr)
            self.assertEqual(database.read_bytes(), b"preserve the existing file")
