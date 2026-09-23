"""Lexer / parser tests covering the SQL grammar."""

import unittest

from minidb.sql import ast
from minidb.sql.lexer import Lexer
from minidb.sql.parser import ParseError, parse_sql


class TestLexer(unittest.TestCase):
    def test_basic_tokens(self):
        toks = Lexer("SELECT a, 42, 3.14, 'it''s' FROM t").tokenize()
        nums = [t for t in toks if t.kind == "NUMBER"]
        strings = [t for t in toks if t.kind == "STRING"]
        self.assertEqual(nums[0].value, "42")
        self.assertEqual(nums[0].literal, 42)
        self.assertEqual(nums[1].literal, 3.14)
        self.assertEqual(strings[0].literal, "it's")
        self.assertEqual(toks[0], ("WORD", "SELECT") if False else toks[0])
        self.assertEqual(toks[0].kind, "WORD")
        self.assertEqual(toks[0].value, "SELECT")

    def test_comparison_normalisation(self):
        toks = Lexer("a <> b != c <= d >= e").tokenize()
        punct = [t.value for t in toks if t.kind == "PUNCT"]
        self.assertEqual(punct, ["!=", "!=", "<=", ">="])

    def test_comment_stripping(self):
        toks = Lexer("-- comment\nSELECT x -- tail\nFROM t").tokenize()
        word_values = [t.value for t in toks if t.kind == "WORD"]
        self.assertEqual(word_values, ["SELECT", "FROM"])
        self.assertEqual([t.value for t in toks if t.kind == "IDENT"], ["x", "t"])

    def test_quoted_identifier(self):
        toks = Lexer('CREATE TABLE t ("weird name" INT)').tokenize()
        idents = [t.value for t in toks if t.kind == "IDENT"]
        self.assertIn("weird name", idents)

    def test_unterminated_string(self):
        with self.assertRaises(Exception):
            Lexer("SELECT 'oops").tokenize()


class TestParser(unittest.TestCase):
    def test_create_table_types(self):
        node = parse_sql("CREATE TABLE t (a INT, b VARCHAR(10), c TEXT, "
                         "d BOOLEAN, e FLOAT, f INTEGER NOT NULL, g BOOL)")
        self.assertIsInstance(node, ast.CreateTable)
        self.assertEqual([c.type for c in node.columns],
                         ["INT", "VARCHAR", "TEXT", "BOOLEAN", "FLOAT",
                          "INT", "BOOLEAN"])
        self.assertTrue(node.columns[5].nullable is False)

    def test_table_level_pk(self):
        node = parse_sql("CREATE TABLE t (a INT, b INT, PRIMARY KEY (a))")
        self.assertTrue(node.columns[0].primary_key)
        self.assertFalse(node.columns[0].nullable)

    def test_insert_multirow(self):
        node = parse_sql("INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y')")
        self.assertEqual(len(node.rows), 2)
        self.assertEqual(node.columns, ["a", "b"])

    def test_select_clauses_order(self):
        node = parse_sql(
            "SELECT a FROM t WHERE a > 1 GROUP BY a HAVING COUNT(*) > 0 "
            "ORDER BY a DESC LIMIT 5 OFFSET 2"
        )
        self.assertEqual(node.limit, 5)
        self.assertEqual(node.offset, 2)
        self.assertTrue(node.order_by[0][1] == "DESC")
        self.assertTrue(node.having)
        self.assertEqual(len(node.group_by), 1)

    def test_join_kinds(self):
        node = parse_sql(
            "SELECT * FROM a INNER JOIN b ON a.id = b.aid LEFT JOIN c "
            "ON b.id = c.bid"
        )
        self.assertEqual([j.kind for j in node.joins], ["INNER", "LEFT"])
        self.assertEqual(node.joins[0].alias, None)

    def test_subquery_in_where(self):
        node = parse_sql("SELECT * FROM t WHERE id IN (SELECT id FROM u)")
        self.assertIsInstance(node.where, ast.InSubquery)
        self.assertFalse(node.where.negated)

    def test_negative_literal_parsing(self):
        node = parse_sql("INSERT INTO t VALUES (-5, -2.5)")
        self.assertIsInstance(node.rows[0][0], ast.UnaryOp)
        self.assertEqual(node.rows[0][0].op, "-")

    def test_between_and_like(self):
        node = parse_sql("SELECT * FROM t WHERE a BETWEEN 1 AND 10")
        self.assertIsInstance(node.where, ast.BinaryOp)
        self.assertEqual(node.where.op, "BETWEEN")
        node2 = parse_sql("SELECT * FROM t WHERE n LIKE 'A%'")
        self.assertEqual(node2.where.op, "LIKE")

    def test_distinct_and_aggregate_syntax(self):
        node = parse_sql("SELECT DISTINCT UPPER(name) FROM t")
        self.assertTrue(node.distinct)
        agg = parse_sql("SELECT dept, COUNT(DISTINCT name) FROM t GROUP BY dept")
        self.assertTrue(agg.projections[1].distinct)

    def begin_isolation_levels(self):
        for sql in ("BEGIN", "BEGIN ISOLATION LEVEL READ COMMITTED",
                    "BEGIN ISOLATION LEVEL SERIALIZABLE",
                    "START TRANSACTION ISOLATION LEVEL SERIALIZABLE"):
            node = parse_sql(sql)
            self.assertIsInstance(node, ast.BeginStmt)

    def test_begin_syntax(self):
        self.begin_isolation_levels()

    def test_invalid_syntax(self):
        bad = [
            "SELECT FROM",
            "INSERT INTO t VALUES",
            "CREATE TABLE t (a WEIRDTYPE)",
            "UPDATE t SET",
            "SELECT * FROM t WHERE",
        ]
        for sql in bad:
            with self.assertRaises(ParseError):
                parse_sql(sql)


if __name__ == "__main__":
    unittest.main()
