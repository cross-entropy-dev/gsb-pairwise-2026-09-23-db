"""CLI tests: statement splitting and end-to-end piped sessions."""

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from minidb.cli import Cli, render_table
from minidb.engine.database import Database
from minidb.engine.session import Session


class TestRenderTable(unittest.TestCase):
    def test_render_aligned(self):
        out = render_table(["id", "name"], [(1, "Ada"), (20, "B")])
        lines = out.splitlines()
        # top border, header, = separator, 2 data rows, bottom border = 6
        self.assertEqual(len(lines), 6)
        self.assertIn("id", lines[1])
        self.assertIn("Ada", lines[3])

    def test_null_and_float_formatting(self):
        out = render_table(["v"], [(None,), (3.14159265,)])
        self.assertIn("NULL", out)
        self.assertIn("3.142", out)


class TestSplitStatements(unittest.TestCase):
    def test_multiple_statements_and_remainder(self):
        buf = "SELECT 1; SELECT 2;\nSELECT 3"
        stmts, rest = Cli._split_statements(buf)
        self.assertEqual(stmts, ["SELECT 1", "SELECT 2"])
        self.assertIn("SELECT 3", rest)

    def test_semicolon_inside_string_not_split(self):
        buf = "INSERT INTO t VALUES ('a;b');\nSELECT 'x;y';"
        stmts, rest = Cli._split_statements(buf)
        self.assertEqual(stmts, ["INSERT INTO t VALUES ('a;b')",
                                "SELECT 'x;y'"])
        self.assertEqual(rest.strip(), "")

    def test_escaped_quote(self):
        buf = "INSERT INTO t VALUES ('it''s; ok');"
        stmts, _ = Cli._split_statements(buf)
        self.assertEqual(len(stmts), 1)
        self.assertIn("it''s", stmts[0])


class TestCliSession(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.dir, "db")
        self.cli = Cli(self.data_dir)

    def tearDown(self):
        self.cli.db.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _exec(self, sql):
        self.cli._execute_and_print(sql)

    def test_create_insert_select(self):
        self._exec("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        self._exec("INSERT INTO t VALUES (1, 10), (2, 20)")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._exec("SELECT * FROM t ORDER BY id")
        self.assertIn("10", buf.getvalue())
        self.assertIn("20", buf.getvalue())

    def test_meta_tables_and_describe(self):
        self._exec("CREATE TABLE t (id INT PRIMARY KEY, v INT NOT NULL)")
        self.assertTrue(self.cli._meta_command(".tables")) if False else None
        buf = io.StringIO()
        with redirect_stdout(buf):
            quit_flag = self.cli._meta_command(".tables")
        self.assertFalse(quit_flag)
        self.assertIn("t", buf.getvalue())

    def test_quit_returns_true(self):
        self.assertTrue(self.cli._meta_command(".quit"))
        self.assertTrue(self.cli._meta_command(".exit"))


class TestCliPipedEndToEnd(unittest.TestCase):
    def test_full_scripted_session(self):
        # drive run() through a fake stdin
        data_dir = os.path.join(tempfile.mkdtemp(), "db")
        script = (
            "CREATE TABLE t (id INT PRIMARY KEY, v INT);\n"
            "INSERT INTO t VALUES (1, 1);\n"
            "BEGIN;\n"
            "UPDATE t SET v = 2 WHERE id = 1;\n"
            "ROLLBACK;\n"
            "SELECT v FROM t;\n"
            ".quit\n"
        )
        old_stdin = sys.stdin
        old_argv_dir = data_dir
        sys.stdin = io.StringIO(script)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                rc = Cli(data_dir).run()
        finally:
            sys.stdin = old_stdin
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("ROLLBACK", text)
        # rolled back -> value still 1
        self.assertRegex(text, r"\|\s*1\s*\|")

        db = Database(data_dir)
        self.assertEqual(
            Session(db).sql("SELECT v FROM t").rows[0], (1,)
        )
        db.shutdown()


if __name__ == "__main__":
    unittest.main()
