"""End-to-end SQL execution tests against an in-memory database."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import (Database, READ_COMMITTED, ExecutionError,
                    TypeMismatchError, ConstraintViolationError,
                    TableNotFoundError, ColumnNotFoundError)


class SQLTestBase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def q(self, sql, txn=None):
        return self.db.execute(sql, txn=txn).tuples()


class DDLCreateTests(SQLTestBase):
    def test_all_column_types(self):
        self.db.execute(
            "CREATE TABLE t (i INT, s VARCHAR(10), x TEXT, "
            "b BOOLEAN, f FLOAT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hi', 'long text', TRUE, 2.5)")
        self.assertEqual(self.q("SELECT i, s, x, b, f FROM t"),
                         [(1, "hi", "long text", True, 2.5)])

    def test_type_coercion_float_to_int(self):
        self.db.execute("CREATE TABLE t (a INT, b FLOAT)")
        self.db.execute("INSERT INTO t VALUES (2.0, 3)")
        self.assertEqual(self.q("SELECT a, b FROM t"), [(2, 3.0)])

    def test_type_mismatch(self):
        self.db.execute("CREATE TABLE t (a INT, b BOOLEAN)")
        with self.assertRaises(TypeMismatchError):
            self.db.execute("INSERT INTO t VALUES ('not a number', TRUE)")
        with self.assertRaises(TypeMismatchError):
            self.db.execute("INSERT INTO t VALUES (1, 'maybe')")

    def test_varchar_length_limit(self):
        self.db.execute("CREATE TABLE t (s VARCHAR(3))")
        with self.assertRaises(TypeMismatchError):
            self.db.execute("INSERT INTO t VALUES ('abcd')")

    def test_not_null(self):
        self.db.execute("CREATE TABLE t (a INT NOT NULL)")
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("INSERT INTO t (a) VALUES (NULL)")

    def test_default_value(self):
        self.db.execute(
            "CREATE TABLE t (a INT DEFAULT 10, b VARCHAR(5) DEFAULT 'x')")
        self.db.execute("INSERT INTO t (a) VALUES (1)")
        self.assertEqual(self.q("SELECT a, b FROM t"), [(1, "x")])

    def test_autoincrement(self):
        self.db.execute(
            "CREATE TABLE t (id INT PRIMARY KEY AUTOINCREMENT, name TEXT)")
        self.db.execute("INSERT INTO t (name) VALUES ('a')")
        self.db.execute("INSERT INTO t (name) VALUES ('b')")
        self.assertEqual(self.q("SELECT id FROM t ORDER BY id"), [(1,), (2,)])

    def test_duplicate_table_and_missing_table(self):
        self.db.execute("CREATE TABLE t (a INT)")
        with self.assertRaises(ExecutionError):
            self.db.execute("CREATE TABLE t (a INT)")
        with self.assertRaises(TableNotFoundError):
            self.q("SELECT * FROM missing")
        # IF NOT EXISTS / IF EXISTS are tolerant.
        self.db.execute("CREATE TABLE IF NOT EXISTS t (a INT)")
        self.db.execute("DROP TABLE IF EXISTS missing")

    def test_drop_table(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("DROP TABLE t")
        with self.assertRaises(TableNotFoundError):
            self.q("SELECT * FROM t")


class InsertSelectTests(SQLTestBase):
    def setUp(self):
        super().setUp()
        self.db.executescript("""
            CREATE TABLE emp (id INT PRIMARY KEY, name VARCHAR(20),
                              dept_id INT, salary FLOAT, active BOOLEAN);
            INSERT INTO emp VALUES
                (1, 'Ada', 10, 100.0, TRUE),
                (2, 'Bob', 10, 200.0, TRUE),
                (3, 'Cy',  20, 150.0, FALSE),
                (4, 'Di',  NULL, 90.0, TRUE);
        """)

    def test_multi_insert_and_select_star(self):
        self.assertEqual(len(self.q("SELECT * FROM emp")), 4)

    def test_named_columns_partial_insert(self):
        self.db.execute("INSERT INTO emp (id, name) VALUES (5, 'Eve')")
        row = self.q("SELECT name, dept_id, salary, active FROM emp WHERE id = 5")
        self.assertEqual(row, [("Eve", None, None, None)])

    def test_column_count_mismatch(self):
        with self.assertRaises(ExecutionError):
            self.db.execute("INSERT INTO emp VALUES (9, 'x')")

    def test_where_operators(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE salary >= 150 ORDER BY name"),
            [("Bob",), ("Cy",)])
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE dept_id IS NULL"),
            [("Di",)])
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE dept_id IS NOT NULL "
                   "ORDER BY name"),
            [("Ada",), ("Bob",), ("Cy",)])

    def test_and_or_not(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE "
                   "(dept_id = 10 AND salary > 150) OR active = FALSE"),
            [("Bob",), ("Cy",)])

    def test_between_in_like(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE salary BETWEEN 90 AND 150 "
                   "ORDER BY name"),
            [("Ada",), ("Cy",), ("Di",)])
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE dept_id IN (10, 20) "
                   "ORDER BY name"),
            [("Ada",), ("Bob",), ("Cy",)])
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE name LIKE '%a' ORDER BY name"),
            [("Ada",)])

    def test_order_by_asc_desc_nulls(self):
        rows = self.q("SELECT name FROM emp ORDER BY dept_id ASC, name DESC")
        # NULL sorts first in ASC.
        self.assertEqual(rows, [("Di",), ("Bob",), ("Ada",), ("Cy",)])
        rows = self.q("SELECT name FROM emp ORDER BY dept_id DESC, name DESC")
        self.assertEqual(rows[:2], [("Cy",), ("Bob",)])  # NULL last in DESC

    def test_limit_offset(self):
        self.assertEqual(
            self.q("SELECT name FROM emp ORDER BY id LIMIT 2"),
            [("Ada",), ("Bob",)])
        self.assertEqual(
            self.q("SELECT name FROM emp ORDER BY id LIMIT 2 OFFSET 2"),
            [("Cy",), ("Di",)])

    def test_distinct(self):
        self.assertEqual(
            self.q("SELECT DISTINCT dept_id FROM emp ORDER BY dept_id"),
            [(None,), (10,), (20,)])

    def test_star_qualified(self):
        rows = self.q("SELECT e.name FROM emp e WHERE e.id = 1")
        self.assertEqual(rows, [("Ada",)])

    def test_alias_expression(self):
        self.assertEqual(
            self.q("SELECT salary * 1.1 AS raised FROM emp WHERE id = 1"),
            [(110.00000000000001,)])


class AggregateGroupTests(SQLTestBase):
    def setUp(self):
        super().setUp()
        self.db.executescript("""
            CREATE TABLE sales (id INT PRIMARY KEY, region VARCHAR(10),
                                amount INT);
            INSERT INTO sales VALUES
                (1, 'east', 10), (2, 'east', 30), (3, 'west', 20),
                (4, 'west', NULL), (5, 'north', 40);
        """)

    def test_aggregates_whole_table(self):
        self.assertEqual(
            self.q("SELECT COUNT(*), COUNT(amount), SUM(amount), "
                   "AVG(amount), MIN(amount), MAX(amount) FROM sales"),
            [(5, 4, 100, 25.0, 10, 40)])

    def test_count_on_empty_table(self):
        self.db.execute("CREATE TABLE empty (a INT)")
        self.assertEqual(self.q("SELECT COUNT(*) FROM empty"), [(0,)])
        self.assertEqual(self.q("SELECT SUM(a), AVG(a), MIN(a), MAX(a) "
                                "FROM empty"),
                         [(None, None, None, None)])

    def test_group_by(self):
        self.assertEqual(
            self.q("SELECT region, COUNT(*), SUM(amount) FROM sales "
                   "GROUP BY region ORDER BY region"),
            [("east", 2, 40), ("north", 1, 40), ("west", 2, 20)])

    def test_having(self):
        self.assertEqual(
            self.q("SELECT region, SUM(amount) FROM sales GROUP BY region "
                   "HAVING SUM(amount) >= 30 ORDER BY region"),
            [("east", 40), ("north", 40)])

    def test_aggregate_distinct(self):
        self.db.executescript("""
            CREATE TABLE t (id INT PRIMARY KEY, g INT);
            INSERT INTO t VALUES (1, 1), (2, 1), (3, 2), (4, 2);
        """)
        self.assertEqual(
            self.q("SELECT COUNT(DISTINCT g), SUM(DISTINCT g) FROM t"),
            [(2, 3)])


class JoinTests(SQLTestBase):
    def setUp(self):
        super().setUp()
        self.db.executescript("""
            CREATE TABLE dept (id INT PRIMARY KEY, name VARCHAR(20));
            CREATE TABLE emp (id INT PRIMARY KEY, name VARCHAR(20),
                              dept_id INT);
            INSERT INTO dept VALUES (1, 'Eng'), (2, 'Sales'), (3, 'Empty');
            INSERT INTO emp VALUES
                (1, 'Ada', 1), (2, 'Bob', 1), (3, 'Cy', 2),
                (4, 'Di', NULL);
        """)

    def test_inner_join(self):
        self.assertEqual(
            self.q("SELECT d.name, e.name FROM dept d "
                   "JOIN emp e ON e.dept_id = d.id "
                   "ORDER BY d.name, e.name"),
            [("Eng", "Ada"), ("Eng", "Bob"), ("Sales", "Cy")])

    def test_inner_join_where_and_agg(self):
        self.assertEqual(
            self.q("SELECT d.name, COUNT(*) FROM dept d "
                   "JOIN emp e ON e.dept_id = d.id GROUP BY d.name "
                   "ORDER BY d.name"),
            [("Eng", 2), ("Sales", 1)])

    def test_left_join_preserves_rows(self):
        self.assertEqual(
            self.q("SELECT d.name, e.name FROM dept d "
                   "LEFT JOIN emp e ON e.dept_id = d.id "
                   "ORDER BY d.name, e.name"),
            [("Empty", None), ("Eng", "Ada"), ("Eng", "Bob"),
             ("Sales", "Cy")])

    def test_left_join_with_filter(self):
        self.assertEqual(
            self.q("SELECT d.name, e.name FROM dept d "
                   "LEFT JOIN emp e ON e.dept_id = d.id AND e.name = 'Ada' "
                   "ORDER BY d.name"),
            [("Empty", None), ("Eng", "Ada"), ("Sales", None)])

    def test_join_ambiguous_column_errors(self):
        with self.assertRaises(ExecutionError):
            self.q("SELECT name FROM dept JOIN emp ON TRUE")


class SubqueryTests(SQLTestBase):
    def setUp(self):
        super().setUp()
        self.db.executescript("""
            CREATE TABLE dept (id INT PRIMARY KEY, name VARCHAR(20));
            CREATE TABLE emp (id INT PRIMARY KEY, name VARCHAR(20),
                              dept_id INT, salary INT);
            INSERT INTO dept VALUES (1, 'Eng'), (2, 'Sales');
            INSERT INTO emp VALUES
                (1, 'Ada', 1, 100), (2, 'Bob', 1, 200), (3, 'Cy', 2, 150);
        """)

    def test_in_subquery(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE dept_id IN "
                   "(SELECT id FROM dept WHERE name = 'Eng') ORDER BY name"),
            [("Ada",), ("Bob",)])

    def test_not_in_subquery(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE dept_id NOT IN "
                   "(SELECT id FROM dept WHERE name = 'Eng') ORDER BY name"),
            [("Cy",)])

    def test_scalar_subquery(self):
        self.assertEqual(
            self.q("SELECT name FROM emp WHERE salary > "
                   "(SELECT AVG(salary) FROM emp) ORDER BY name"),
            [("Bob",)])

    def test_exists_correlated(self):
        self.assertEqual(
            self.q("SELECT name FROM dept d WHERE EXISTS "
                   "(SELECT 1 FROM emp e WHERE e.dept_id = d.id) "
                   "ORDER BY name"),
            [("Eng",), ("Sales",)])

    def test_not_exists_correlated(self):
        self.db.execute("INSERT INTO dept VALUES (3, 'Empty')")
        self.assertEqual(
            self.q("SELECT name FROM dept d WHERE NOT EXISTS "
                   "(SELECT 1 FROM emp e WHERE e.dept_id = d.id) "
                   "ORDER BY name"),
            [("Empty",)])


class UpdateDeleteTests(SQLTestBase):
    def setUp(self):
        super().setUp()
        self.db.executescript("""
            CREATE TABLE t (id INT PRIMARY KEY, v INT, tag VARCHAR(10));
            INSERT INTO t VALUES (1, 10, 'a'), (2, 20, 'a'), (3, 30, 'b');
        """)

    def test_update_all_and_filtered(self):
        r = self.db.execute("UPDATE t SET v = v + 1")
        self.assertEqual(r.rowcount, 3)
        self.assertEqual(self.q("SELECT SUM(v) FROM t"), [(63,)])
        self.db.execute("UPDATE t SET tag = 'z' WHERE id = 2")
        self.assertEqual(
            self.q("SELECT tag FROM t WHERE id = 2"), [("z",)])

    def test_delete_filtered_and_all(self):
        r = self.db.execute("DELETE FROM t WHERE tag = 'a'")
        self.assertEqual(r.rowcount, 2)
        self.assertEqual(self.q("SELECT id FROM t"), [(3,)])
        self.db.execute("DELETE FROM t")
        self.assertEqual(self.q("SELECT COUNT(*) FROM t"), [(0,)])

    def test_primary_key_and_unique_constraints(self):
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("INSERT INTO t (id, v) VALUES (1, 0)")
        self.db.execute(
            "CREATE TABLE u (id INT PRIMARY KEY, email VARCHAR(50) UNIQUE)")
        self.db.execute("INSERT INTO u VALUES (1, 'a@x.com')")
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("INSERT INTO u VALUES (2, 'a@x.com')")
        # NULL duplicates are allowed.
        self.db.execute("INSERT INTO u VALUES (2, NULL)")
        self.db.execute("INSERT INTO u VALUES (3, NULL)")

    def test_pk_immutable_update_restriction_is_permissive_here(self):
        # Updating a non-PK column works; PK columns cannot change.
        self.db.execute("UPDATE t SET v = 11 WHERE id = 1")
        self.assertEqual(self.q("SELECT v FROM t WHERE id = 1"), [(11,)])
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("UPDATE t SET id = 99 WHERE id = 1")


class ScriptTransactionTests(SQLTestBase):
    def test_script_commits(self):
        result = self.db.executescript("""
            CREATE TABLE t (id INT PRIMARY KEY);
            INSERT INTO t VALUES (1), (2);
        """)
        self.assertEqual(self.q("SELECT COUNT(*) FROM t"), [(2,)])

    def test_script_rolls_back_on_error(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        with self.assertRaises(Exception):
            self.db.executescript(
                "INSERT INTO t VALUES (1); INSERT INTO t VALUES (1);")
        self.assertEqual(self.q("SELECT COUNT(*) FROM t"), [(0,)])

    def test_select_without_from(self):
        self.assertEqual(self.q("SELECT 1 + 2"), [(3,)])
        self.assertEqual(self.q("SELECT 'hello'"), [("hello",)])

    def test_case_expression(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, score INT)")
        self.db.execute("INSERT INTO t VALUES (1, 90), (2, 40)")
        self.assertEqual(
            self.q("SELECT CASE WHEN score >= 50 THEN 'pass' "
                   "ELSE 'fail' END FROM t ORDER BY id"),
            [("pass",), ("fail",)])


if __name__ == "__main__":
    unittest.main()
