"""Expression evaluation against an intermediate row dictionary.

Rows produced by the executor map both bare column names (``name``) and
qualified names (``t.name``) to values, so ``Column`` references resolve with
or without a table qualifier.  Aggregate function calls are not evaluated here
- the SELECT executor computes them first and passes them through ``agg_map``.
"""

from ..sql import ast_nodes as ast
from ..errors import ExecutionError, ColumnNotFoundError
from .types import (sql_truth, compare, add, sub, mul, div, like,
                    TypeMismatchError)

AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


class Evaluator:
    def __init__(self, executor, txn, agg_map=None, outer_row=None):
        self.executor = executor
        self.txn = txn
        # id(FunctionCall) -> computed aggregate value
        self.agg_map = agg_map or {}
        # Row of an enclosing query for correlated subquery references.
        self.outer_row = outer_row or {}

    def eval(self, node, row):
        method = self._DISPATCH.get(type(node))
        if method is None:
            raise ExecutionError(f"cannot evaluate {type(node).__name__}")
        return method(self, node, row)

    # ------------------------------------------------------------- literals

    def _literal(self, node, row):
        return node.value

    def _column(self, node, row):
        if node.table is not None:
            key = f"{node.table}.{node.name}"
            if key in row:
                return row[key]
            if key in self.outer_row:
                return self.outer_row[key]
            raise ColumnNotFoundError(f"unknown column {key}")
        if node.name in row:
            return row[node.name]
        if node.name in self.outer_row:
            return self.outer_row[node.name]
        raise ColumnNotFoundError(f"unknown column {node.name}")

    # ---------------------------------------------------------- arithmetic

    def _binary(self, node, row):
        op = node.op
        if op == "AND":
            l = sql_truth(self.eval(node.left, row))
            if l is False:
                return False
            r = sql_truth(self.eval(node.right, row))
            if l is True and r is True:
                return True
            if l is False or r is False:
                return False
            return None
        if op == "OR":
            l = sql_truth(self.eval(node.left, row))
            if l is True:
                return True
            r = sql_truth(self.eval(node.right, row))
            if r is True:
                return True
            if l is None or r is None:
                return None
            return False

        a = self.eval(node.left, row)
        b = self.eval(node.right, row)
        if op in ("=", "<>", "<", ">", "<=", ">="):
            return self._compare(op, a, b)
        if op == "LIKE":
            return like(a, b)
        if op == "NOT LIKE":
            r = like(a, b)
            return None if r is None else not r
        if op == "+":
            return add(a, b)
        if op == "-":
            return sub(a, b)
        if op == "*":
            return mul(a, b)
        if op == "/":
            return div(a, b)
        raise ExecutionError(f"unknown operator {op}")

    def _compare(self, op, a, b):
        c = compare(a, b)
        if c is None:
            return None
        if op == "=":
            return c == 0
        if op == "<>":
            return c != 0
        if op == "<":
            return c < 0
        if op == ">":
            return c > 0
        if op == "<=":
            return c <= 0
        if op == ">=":
            return c >= 0

    def _unary(self, node, row):
        if node.op == "NOT":
            v = sql_truth(self.eval(node.operand, row))
            return None if v is None else not v
        if node.op == "IS_NULL":
            return self.eval(node.operand, row) is None
        if node.op == "IS_NOT_NULL":
            return self.eval(node.operand, row) is not None
        if node.op == "-":
            v = self.eval(node.operand, row)
            return None if v is None else -v
        raise ExecutionError(f"unknown unary operator {node.op}")

    # -------------------------------------------------------- special forms

    def _between(self, node, row):
        v = self.eval(node.expr, row)
        lo = self.eval(node.low, row)
        hi = self.eval(node.high, row)
        if v is None or lo is None or hi is None:
            return None
        c1 = compare(v, lo)
        c2 = compare(v, hi)
        if c1 is None or c2 is None:
            return None
        result = c1 >= 0 and c2 <= 0
        return (not result) if node.negated else result

    def _in_list(self, node, row):
        v = self.eval(node.expr, row)
        if v is None:
            return None
        saw_null = False
        for item in node.values:
            other = self.eval(item, row)
            if other is None:
                saw_null = True
                continue
            if compare(v, other) == 0:
                return not node.negated
        if saw_null:
            return None
        return node.negated  # NOT IN empty/nonmatch -> True; IN -> False/NULL

    def _in_subquery(self, node, row):
        v = self.eval(node.expr, row)
        if v is None:
            return None
        result_rows = self.executor.execute_select(
            node.subquery, self.txn, outer_row=row)
        saw_null = False
        for r in result_rows:
            other = next(iter(r.values()))
            if other is None:
                saw_null = True
                continue
            if compare(v, other) == 0:
                return not node.negated
        if saw_null:
            return None
        return node.negated

    def _exists(self, node, row):
        result_rows = self.executor.execute_select(
            node.subquery, self.txn, outer_row=row)
        found = len(result_rows) > 0
        return (not found) if node.negated else found

    def _subquery(self, node, row):
        rows = self.executor.execute_select(
            node.select, self.txn, outer_row=row)
        if not rows:
            return None
        if len(rows) > 1:
            raise ExecutionError("scalar subquery returned more than one row")
        return next(iter(rows[0].values()))

    def _case(self, node, row):
        if node.operand is not None:
            probe = self.eval(node.operand, row)
            for cond, result in node.whens:
                if compare(probe, self.eval(cond, row)) == 0:
                    return self.eval(result, row)
        else:
            for cond, result in node.whens:
                if sql_truth(self.eval(cond, row)) is True:
                    return self.eval(result, row)
        if node.default is not None:
            return self.eval(node.default, row)
        return None

    def _function(self, node, row):
        if node.name in AGGREGATES:
            if id(node) in self.agg_map:
                return self.agg_map[id(node)]
            raise ExecutionError(
                f"aggregate {node.name} used outside an aggregate context")
        raise ExecutionError(f"unknown function {node.name}")

    def _star(self, node, row):
        raise ExecutionError("'*' cannot be evaluated as a scalar")

    def _qualified_star(self, node, row):
        raise ExecutionError("'*' cannot be evaluated as a scalar")


Evaluator._DISPATCH = {
    ast.Literal: Evaluator._literal,
    ast.Column: Evaluator._column,
    ast.BinaryOp: Evaluator._binary,
    ast.UnaryOp: Evaluator._unary,
    ast.Between: Evaluator._between,
    ast.InList: Evaluator._in_list,
    ast.InSubquery: Evaluator._in_subquery,
    ast.Exists: Evaluator._exists,
    ast.Subquery: Evaluator._subquery,
    ast.CaseExpr: Evaluator._case,
    ast.FunctionCall: Evaluator._function,
    ast.Star: Evaluator._star,
    ast.QualifiedStar: Evaluator._qualified_star,
}


def collect_aggregates(node, out=None):
    """Find every aggregate FunctionCall in an expression tree."""
    if out is None:
        out = []
    if isinstance(node, ast.FunctionCall) and node.name in AGGREGATES:
        out.append(node)
        return out
    for attr in ("left", "right", "operand", "expr", "low", "high",
                 "default"):
        child = getattr(node, attr, None)
        if child is not None and hasattr(child, "__dataclass_fields__", ):
            collect_aggregates(child, out)
    for attr in ("args", "values", "whens", "group_by"):
        children = getattr(node, attr, None)
        if children:
            for child in children:
                if hasattr(child, "__dataclass_fields__"):
                    collect_aggregates(child, out)
                elif isinstance(child, tuple):
                    for part in child:
                        if hasattr(part, "__dataclass_fields__"):
                            collect_aggregates(part, out)
    if hasattr(node, "subquery"):
        # Aggregates inside a subquery belong to the subquery, not here.
        pass
    return out
