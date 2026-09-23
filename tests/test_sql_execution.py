"""End-to-end SQL execution tests (single-session)."""

import os
import shutil
import tempfile
import unittest

from minidb import Database, Session
from minidb.engine.errors import IntegrityError
from minidb.engine.session import ExplainResult


class SQLFixture:
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.dir, "db"))
        self.s = Session(self.db)
        self.s.sql("CREATE TABLE dept (id INT PRIMARY KEY, name VARCHAR(20))")
        self.s.sql(
            "CREATE TABLE emp (id INT PRIMARY KEY, name VARCHAR(20) NOT NULL, "
            "dept_id INT, salary FLOAT, active BOOLEAN)"
        )
        self.s.sql("INSERT INTO dept VALUES (10, 'Eng'), (20, 'Sales'), (30, 'HR')")
        self.s.sql(
            "INSERT INTO emp VALUES "
            "(1, 'Ada', 10, 100.0, TRUE), "
            "(2, 'Bob', 10, 200.0, TRUE), "
            "(3, 'Cyd', 20, 150.0, FALSE), "
            "(4, 'Dan', NULL, 50.0, TRUE)"
        )

    def tearDown(self):
        self.db.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)


class TestTypesAndDDL(SQLFixture, unittest.TestCase):
    def test_all_types_roundtrip(self):
        self.s.sql("CREATE TABLE wide (id INT PRIMARY KEY, a INT, b FLOAT, "
                   "c VARCHAR(5), d TEXT, e BOOLEAN)")
        self.s.sql("INSERT INTO wide VALUES (1, -7, 2.5, 'abc', 'loooong', TRUE)")
        row = self.s.sql("SELECT a, b, c, d, e FROM wide").rows[0]
        self.assertEqual(row, (-7, 2.5, "abc", "loooong", True))

    def test_varchar_length_enforced(self):
        self.s.sql("CREATE TABLE v (id INT PRIMARY KEY, s VARCHAR(3))")
        with self.assertRaises(IntegrityError):
            self.s.sql("INSERT INTO v VALUES (1, 'toolong')")

    def test_not_null_enforced(self):
        with self.assertRaises(IntegrityError):
            self.s.sql("INSERT INTO emp (id) VALUES (99)")

    def test_duplicate_table(self):
        with self.assertRaises(Exception):
            self.s.sql("CREATE TABLE dept (x INT PRIMARY KEY)")

    def test_describe(self):
        res = self.s.sql("DESCRIBE emp")
        names = [r[0] for r in res.rows]
        self.assertEqual(names, ["id", "name", "dept_id", "salary", "active"])
        pk = [r for r in res.rows if r[0] == "id"][0]
        self.assertEqual(pk[3], "YES")

    def test_show_tables(self):
        self.assertEqual(
            sorted(r[0] for r in self.s.sql("SHOW TABLES").rows),
            ["dept", "emp"],
        )

    def test_type_coercion(self):
        self.s.sql("CREATE TABLE c (id INT PRIMARY KEY, f FLOAT, b BOOLEAN)")
        self.s.sql("INSERT INTO c VALUES (1, '3.5', 'true')")
        self.assertEqual(self.s.sql("SELECT f, b FROM c").rows[0], (3.5, True))


class TestSelect(SQLFixture, unittest.TestCase):
    def test_where_operators(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE salary >= 100 AND active = TRUE "
            "ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Ada",), ("Bob",)])

    def test_null_three_valued_logic(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE dept_id = NULL OR dept_id IS NULL"
        ).rows
        self.assertEqual(rows, [("Dan",)])
        self.assertEqual(
            self.s.sql("SELECT name FROM emp WHERE dept_id IS NOT NULL "
                       "ORDER BY name").rows,
            [("Ada",), ("Bob",), ("Cyd",)],
        )

    def test_like(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE name LIKE 'A%' OR name LIKE '_a%' "
            "ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Ada",), ("Dan",)])

    def test_between_in(self):
        self.assertEqual(
            self.s.sql("SELECT name FROM emp WHERE id BETWEEN 2 AND 3 "
                       "ORDER BY name").rows,
            [("Bob",), ("Cyd",)],
        )
        self.assertEqual(
            self.s.sql("SELECT name FROM emp WHERE id IN (1, 4) "
                       "ORDER BY name").rows,
            [("Ada",), ("Dan",)],
        )

    def test_limit_offset(self):
        rows = self.s.sql("SELECT name FROM emp ORDER BY id LIMIT 2 OFFSET 1").rows
        self.assertEqual(rows, [("Bob",), ("Cyd",)])

    def test_order_by_mixed_and_nulls(self):
        asc = self.s.sql("SELECT dept_id FROM emp ORDER BY dept_id ASC").rows
        # NULLS LAST for ASC
        self.assertEqual(asc[-1], (None,))
        desc = self.s.sql("SELECT dept_id FROM emp ORDER BY dept_id DESC").rows
        # NULLS FIRST for DESC
        self.assertEqual(desc[0], (None,))
        self.assertEqual([r[0] for r in desc[1:]], [20, 10, 10])

    def test_multi_key_order(self):
        rows = self.s.sql(
            "SELECT name, dept_id FROM emp ORDER BY dept_id ASC, name DESC"
        ).rows
        self.assertEqual(rows[:2], [("Bob", 10), ("Ada", 10)])

    def test_distinct(self):
        rows = self.s.sql(
            "SELECT DISTINCT dept_id FROM emp ORDER BY dept_id"
        ).rows
        self.assertEqual(rows, [(10,), (20,), (None,)])

    def test_arithmetic_and_scalar_funcs(self):
        rows = self.s.sql(
            "SELECT UPPER(name), salary + 10 FROM emp WHERE name = 'Ada'"
        ).rows
        self.assertEqual(rows[0], ("ADA", 110.0))


class TestAggregates(SQLFixture, unittest.TestCase):
    def test_scalar_aggregates(self):
        row = self.s.sql(
            "SELECT COUNT(*), COUNT(dept_id), SUM(salary), AVG(salary), "
            "MIN(salary), MAX(salary) FROM emp"
        ).rows[0]
        self.assertEqual(row, (4, 3, 500.0, 125.0, 50.0, 200.0))

    def test_group_by(self):
        rows = self.s.sql(
            "SELECT dept_id, COUNT(*), SUM(salary) FROM emp "
            "WHERE dept_id IS NOT NULL GROUP BY dept_id ORDER BY dept_id"
        ).rows
        self.assertEqual(rows, [(10, 2, 300.0), (20, 1, 150.0)])

    def test_having(self):
        rows = self.s.sql(
            "SELECT dept_id, COUNT(*) FROM emp GROUP BY dept_id "
            "HAVING COUNT(*) >= 2"
        ).rows
        self.assertEqual(rows, [(10, 2)])

    def test_count_distinct(self):
        row = self.s.sql("SELECT COUNT(DISTINCT dept_id) FROM emp").rows[0]
        self.assertEqual(row, (2,))

    def test_aggregate_empty_set(self):
        row = self.s.sql("SELECT COUNT(*), SUM(salary), AVG(salary) "
                         "FROM emp WHERE id > 1000").rows[0]
        self.assertEqual(row, (0, None, None))

    def test_order_by_aggregate(self):
        rows = self.s.sql(
            "SELECT dept_id, SUM(salary) FROM emp WHERE dept_id IS NOT NULL "
            "GROUP BY dept_id ORDER BY SUM(salary) DESC"
        ).rows
        self.assertEqual(rows, [(10, 300.0), (20, 150.0)])


class TestJoins(SQLFixture, unittest.TestCase):
    def test_inner_join(self):
        rows = self.s.sql(
            "SELECT e.name, d.name FROM emp e JOIN dept d "
            "ON e.dept_id = d.id ORDER BY e.name"
        ).rows
        self.assertEqual(
            rows,
            [("Ada", "Eng"), ("Bob", "Eng"), ("Cyd", "Sales")],
        )

    def test_left_join_preserves_unmatched(self):
        rows = self.s.sql(
            "SELECT e.name, d.name FROM emp e LEFT JOIN dept d "
            "ON e.dept_id = d.id ORDER BY e.name"
        ).rows
        self.assertIn(("Dan", None), rows)
        self.assertEqual(len(rows), 4)

    def test_left_join_with_filter_on_left(self):
        rows = self.s.sql(
            "SELECT e.name, d.name FROM emp e LEFT JOIN dept d "
            "ON e.dept_id = d.id WHERE e.active = TRUE ORDER BY e.name"
        ).rows
        self.assertEqual(
            rows,
            [("Ada", "Eng"), ("Bob", "Eng"), ("Dan", None)],
        )

    def test_join_multitable_predicate(self):
        rows = self.s.sql(
            "SELECT e.name FROM emp e JOIN dept d ON e.dept_id = d.id "
            "WHERE d.name = 'Eng' ORDER BY e.name"
        ).rows
        self.assertEqual(rows, [("Ada",), ("Bob",)])


class TestSubqueries(SQLFixture, unittest.TestCase):
    def test_in_subquery(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE dept_id IN "
            "(SELECT id FROM dept WHERE name = 'Eng') ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Ada",), ("Bob",)])

    def test_not_in_subquery(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE dept_id NOT IN "
            "(SELECT id FROM dept WHERE name = 'Eng') "
            "AND dept_id IS NOT NULL ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Cyd",)])

    def test_scalar_subquery_comparison(self):
        rows = self.s.sql(
            "SELECT name FROM emp WHERE salary > "
            "(SELECT AVG(salary) FROM emp) ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Bob",), ("Cyd",)])

    def test_correlated_column_ambient_snapshot(self):
        # subquery shares the outer statement's snapshot
        rows = self.s.sql(
            "SELECT d.name, (SELECT COUNT(*) FROM emp e WHERE e.dept_id = d.id) "
            "FROM dept d ORDER BY d.id"
        ).rows
        self.assertEqual(rows, [("Eng", 2), ("Sales", 1), ("HR", 0)])


class TestDML(SQLFixture, unittest.TestCase):
    def test_insert_partial_columns(self):
        self.s.sql("INSERT INTO emp (id, name) VALUES (9, 'Zed')")
        row = self.s.sql("SELECT name, salary, active FROM emp WHERE id = 9").rows[0]
        self.assertEqual(row, ("Zed", None, None))

    def test_duplicate_pk_rejected(self):
        with self.assertRaises(IntegrityError):
            self.s.sql("INSERT INTO dept VALUES (10, 'Dup')")

    def test_update_matches_where(self):
        n = self.s.sql("UPDATE emp SET salary = 0 WHERE active = FALSE").rows[0][0]
        self.assertEqual(n, 1)
        self.assertEqual(
            self.s.sql("SELECT salary FROM emp WHERE name = 'Cyd'").rows[0], (0,)
        )

    def test_update_using_column(self):
        self.s.sql("UPDATE emp SET salary = salary + 50 WHERE dept_id = 10")
        rows = self.s.sql(
            "SELECT name, salary FROM emp WHERE dept_id = 10 ORDER BY name"
        ).rows
        self.assertEqual(rows, [("Ada", 150.0), ("Bob", 250.0)])

    def test_update_primary_key(self):
        self.s.sql("UPDATE dept SET id = 99 WHERE name = 'HR'")
        self.assertEqual(
            self.s.sql("SELECT name FROM dept WHERE id = 99").rows, [("HR",)]
        )
        self.assertEqual(
            self.s.sql("SELECT COUNT(*) FROM dept").rows[0], (3,)
        )
        with self.assertRaises(IntegrityError):
            self.s.sql("UPDATE dept SET id = 10 WHERE name = 'HR'")

    def test_delete(self):
        n = self.s.sql("DELETE FROM emp WHERE dept_id IS NULL").rows[0][0]
        self.assertEqual(n, 1)
        self.assertEqual(self.s.sql("SELECT COUNT(*) FROM emp").rows[0], (3,))

    def test_delete_all(self):
        n = self.s.sql("DELETE FROM emp").rows[0][0]
        self.assertEqual(n, 4)
        self.assertEqual(self.s.sql("SELECT COUNT(*) FROM emp").rows[0], (0,))


class TestNoFromAndExplain(SQLFixture, unittest.TestCase):
    def test_select_literal(self):
        self.assertEqual(self.s.sql("SELECT 1 + 2").rows, [(3,)])
        self.assertEqual(self.s.sql("SELECT 'hi', TRUE").rows, [("hi", True)])

    def test_explain_shows_pushdown(self):
        plan = self.s.sql(
            "EXPLAIN SELECT * FROM emp e JOIN dept d ON e.dept_id = d.id "
            "WHERE e.salary > 100"
        )
        self.assertIsInstance(plan, ExplainResult)
        self.assertIn("filter pushed", plan.plan)


if __name__ == "__main__":
    unittest.main()
