"""Tests for the lexer and recursive-descent parser."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb.sql.lexer import tokenize
from minidb.sql.parser import parse, parse_one
from minidb.sql import ast_nodes as ast
from minidb.errors import LexerError, ParseError


class LexerTests(unittest.TestCase):
    def test_simple_tokens(self):
        toks = tokenize("SELECT a FROM t WHERE b = 10")
        self.assertEqual([t.type for t in toks],
                         ["KEYWORD", "IDENT", "KEYWORD", "IDENT",
                          "KEYWORD", "IDENT", "OP", "NUMBER"])

    def test_string_and_number_types(self):
        toks = tokenize("'it''s' 3.14 42 1e3")
        self.assertEqual(toks[0].value, "it's")
        self.assertEqual(toks[1].value, 3.14)
        self.assertEqual(toks[2].value, 42)
        self.assertEqual(toks[3].value, 1000.0)

    def test_operators(self):
        toks = tokenize("a <= b AND c <> d != e")
        self.assertEqual([t.value for t in toks if t.type == "OP"],
                         ["<=", "<>", "!="])

    def test_comments(self):
        toks = tokenize("SELECT 1 -- a comment\n/* block */ + 2")
        self.assertEqual(len(toks), 4)

    def test_unterminated_string(self):
        with self.assertRaises(LexerError):
            tokenize("'never ends")

    def test_bad_character(self):
        with self.assertRaises(LexerError):
            tokenize("SELECT @ FROM t")

    def test_quoted_identifier(self):
        toks = tokenize('SELECT "select" FROM t')
        self.assertEqual(toks[1].type, "IDENT")
        self.assertEqual(toks[1].value, "select")


class ParserTests(unittest.TestCase):
    def test_multiple_statements(self):
        stmts = parse("SELECT 1; SELECT 2;")
        self.assertEqual(len(stmts), 2)

    def test_create_table_types(self):
        node = parse_one(
            "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(50) NOT NULL, "
            "note TEXT, active BOOLEAN DEFAULT TRUE, score FLOAT)")
        self.assertIsInstance(node, ast.CreateTable)
        types = {c.name: c.type_name for c in node.columns}
        self.assertEqual(types, {"id": "INT", "name": "VARCHAR",
                                 "note": "TEXT", "active": "BOOLEAN",
                                 "score": "FLOAT"})
        self.assertTrue(node.columns[0].primary_key)
        self.assertTrue(node.columns[1].not_null)
        self.assertEqual(node.columns[2].length, None)
        self.assertEqual(node.columns[3].default.value, True)

    def test_insert_multi_rows(self):
        node = parse_one("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)")
        self.assertEqual(len(node.rows), 2)
        self.assertEqual(node.columns, ["a", "b"])

    def test_select_clauses(self):
        node = parse_one(
            "SELECT DISTINCT a, COUNT(*) FROM t WHERE a > 1 "
            "GROUP BY a HAVING COUNT(*) > 2 ORDER BY a DESC LIMIT 5 OFFSET 1")
        self.assertTrue(node.distinct)
        self.assertEqual(len(node.items), 2)
        self.assertIsInstance(node.where, ast.BinaryOp)
        self.assertEqual(len(node.group_by), 1)
        self.assertIsNotNone(node.having)
        self.assertTrue(node.order_by[0].desc)
        self.assertEqual(node.limit.value, 5)
        self.assertEqual(node.offset.value, 1)

    def test_operator_precedence(self):
        node = parse_one("SELECT * FROM t WHERE a OR b AND c")
        # OR should be the root, AND its right subtree.
        self.assertEqual(node.where.op, "OR")
        self.assertEqual(node.where.right.op, "AND")

    def test_arithmetic_precedence(self):
        node = parse_one("SELECT 1 + 2 * 3 FROM t")
        expr = node.items[0].expr
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.right.op, "*")

    def test_unary_minus_and_not(self):
        node = parse_one("SELECT * FROM t WHERE NOT a = -5")
        self.assertEqual(node.where.op, "NOT")
        self.assertEqual(node.where.operand.op, "=")
        self.assertIsInstance(node.where.operand.right, ast.UnaryOp)
        self.assertEqual(node.where.operand.right.op, "-")

    def test_join_variants(self):
        node = parse_one(
            "SELECT * FROM a JOIN b ON a.id = b.id "
            "LEFT JOIN c ON c.id = a.id")
        self.assertIsInstance(node.source, ast.Join)
        self.assertEqual(node.source.join_type, "LEFT")
        self.assertIsInstance(node.source.left, ast.Join)
        self.assertEqual(node.source.left.join_type, "INNER")

    def test_between_like_in(self):
        self.assertIsInstance(
            parse_one("SELECT * FROM t WHERE a BETWEEN 1 AND 3").where,
            ast.Between)
        self.assertEqual(
            parse_one("SELECT * FROM t WHERE a LIKE 'x%'").where.op, "LIKE")
        self.assertIsInstance(
            parse_one("SELECT * FROM t WHERE a IN (1, 2, 3)").where,
            ast.InList)

    def test_subqueries(self):
        node = parse_one(
            "SELECT * FROM t WHERE a IN (SELECT id FROM u) "
            "AND EXISTS (SELECT 1 FROM v)")
        self.assertIsInstance(node.where.left, ast.InSubquery)
        self.assertIsInstance(node.where.right, ast.Exists)

    def test_case_expression(self):
        node = parse_one(
            "SELECT CASE WHEN a > 1 THEN 'big' ELSE 'small' END FROM t")
        self.assertIsInstance(node.items[0].expr, ast.CaseExpr)

    def test_aggregate_distinct(self):
        node = parse_one("SELECT COUNT(DISTINCT a), SUM(b) FROM t")
        self.assertTrue(node.items[0].expr.distinct)
        self.assertFalse(node.items[1].expr.distinct)

    def test_transaction_statements(self):
        self.assertIsNone(parse_one("BEGIN").isolation)
        self.assertEqual(parse_one(
            "BEGIN ISOLATION LEVEL SERIALIZABLE").isolation, "SERIALIZABLE")
        self.assertIsInstance(parse_one("COMMIT"), ast.Commit)
        self.assertIsInstance(parse_one("ROLLBACK"), ast.Rollback)

    def test_table_level_primary_key(self):
        node = parse_one("CREATE TABLE t (a INT, b INT, PRIMARY KEY (a, b))")
        self.assertTrue(node.columns[0].primary_key)
        self.assertTrue(node.columns[1].primary_key)

    def test_function_call_expression(self):
        node = parse_one("SELECT * FROM t WHERE a = LENGTH(b)")
        # LENGTH is parsed as an (unknown) function call; resolution happens
        # in the executor where it is rejected as unsupported.
        self.assertIsInstance(node.where.right, ast.FunctionCall)

    def test_invalid_syntax(self):
        with self.assertRaises(ParseError):
            parse_one("SELECT FROM")
        with self.assertRaises(ParseError):
            parse_one("INSERT INTO t VALUES")
        with self.assertRaises(ParseError):
            parse_one("UPDATE t SET")


if __name__ == "__main__":
    unittest.main()
