"""Expression evaluation with SQL three-valued logic.

A :class:`EvalContext` binds the FROM sources (alias -> schema + current row)
and optionally a subquery executor.  Aggregates are not evaluated here – the
select executor reduces them during grouping.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..sql import ast


class EvalError(Exception):
    pass


# Correlated-subquery support: while a subquery runs, the calling expression
# context is pushed onto this thread-local stack.  Inner EvalContexts fall
# back to it for names they cannot resolve themselves.
_OUTER_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "minidb_outer_ctx_stack", default=()
)


def push_outer_context(ctx: "EvalContext"):
    token = _OUTER_STACK.set(_OUTER_STACK.get() + (ctx,))
    return token


def pop_outer_context(token) -> None:
    _OUTER_STACK.reset(token)


@dataclass
class SourceBinding:
    alias: str
    columns: list[str]            # lower-cased column names
    row: Optional[tuple]          # None for LEFT JOIN unmatched side


class EvalContext:
    def __init__(self, bindings: list[SourceBinding],
                 subquery_runner=None) -> None:
        self.bindings = bindings
        self.subquery_runner = subquery_runner
        self._name_map: dict[str, list[tuple[int, int]]] = {}
        for bi, b in enumerate(bindings):
            for ci, col in enumerate(b.columns):
                self._name_map.setdefault(col.lower(), []).append((bi, ci))
        self._alias_map = {b.alias.lower(): i for i, b in enumerate(bindings)}

    def with_rows(self, rows: list[Optional[tuple]]) -> "EvalContext":
        """Return a context sharing schema bindings with new row values."""
        new = EvalContext(
            [SourceBinding(b.alias, b.columns, row)
             for b, row in zip(self.bindings, rows)],
            self.subquery_runner,
        )
        return new

    def resolve(self, ref: ast.ColumnRef) -> Any:
        if ref.table:
            bi = self._alias_map.get(ref.table.lower())
            if bi is not None:
                b = self.bindings[bi]
                try:
                    ci = b.columns.index(ref.name.lower())
                except ValueError:
                    raise EvalError(f"unknown column {ref.table}.{ref.name}")
                return None if b.row is None else b.row[ci]
            # not local: try outer (correlated) scopes, innermost first
            for outer in reversed(_OUTER_STACK.get()):
                if self._outer_has_alias(outer, ref.table):
                    return outer.resolve(ref)
            raise EvalError(f"unknown table alias {ref.table!r}")
        candidates = self._name_map.get(ref.name.lower())
        if candidates:
            bi, ci = candidates[0]
            b = self.bindings[bi]
            return None if b.row is None else b.row[ci]
        # fall back to outer scopes for correlated references
        for outer in reversed(_OUTER_STACK.get()):
            if ref.name.lower() in outer._name_map:
                return outer.resolve(ref)
        raise EvalError(f"unknown column {ref.name!r}")

    @staticmethod
    def _outer_has_alias(outer: "EvalContext", alias: str) -> bool:
        return alias.lower() in outer._alias_map


# ---------------------------------------------------------------------- #
# public entry point
# ---------------------------------------------------------------------- #
def evaluate(expr: Any, ctx: EvalContext) -> Any:
    if isinstance(expr, ast.Literal):
        return expr.value
    if isinstance(expr, ast.ColumnRef):
        return ctx.resolve(expr)
    if isinstance(expr, ast.UnaryOp):
        return _eval_unary(expr, ctx)
    if isinstance(expr, ast.BinaryOp):
        return _eval_binary(expr, ctx)
    if isinstance(expr, ast.FuncCall):
        return _eval_scalar_func(expr, ctx)
    if isinstance(expr, (ast.Subquery, ast.InSubquery, ast.Exists)):
        return _eval_subquery_expr(expr, ctx)
    raise EvalError(f"cannot evaluate {expr!r}")


def truthy(value: Any) -> bool:
    """SQL truth: only TRUE is true; NULL and FALSE are not."""
    return value is True


# ---------------------------------------------------------------------- #
def _eval_unary(expr: ast.UnaryOp, ctx: EvalContext) -> Any:
    op = expr.op
    if op == "-":
        v = evaluate(expr.operand, ctx)
        return None if v is None else -v
    if op == "NOT":
        v = evaluate(expr.operand, ctx)
        return None if v is None else (not v)
    if op == "IS NULL":
        return evaluate(expr.operand, ctx) is None
    if op == "IS NOT NULL":
        return evaluate(expr.operand, ctx) is not None
    raise EvalError(f"unsupported unary op {op}")


def _eval_binary(expr: ast.BinaryOp, ctx: EvalContext) -> Any:
    op = expr.op
    if op == "AND":
        l = evaluate(expr.left, ctx)
        r = evaluate(expr.right, ctx)
        if l is False or r is False:
            return False
        if l is None or r is None:
            return None
        return bool(l and r)
    if op == "OR":
        l = evaluate(expr.left, ctx)
        r = evaluate(expr.right, ctx)
        if l is True or r is True:
            return True
        if l is None or r is None:
            return None
        return bool(l or r)

    l = evaluate(expr.left, ctx)
    r = evaluate(expr.right, ctx)

    if op == "BETWEEN":
        # right is AND(low, high) built by the parser
        low = evaluate(expr.right.left, ctx)
        high = evaluate(expr.right.right, ctx)
        if l is None or low is None or high is None:
            return None
        return low <= l <= high

    if op in ("LIKE", "NOT LIKE"):
        if l is None or r is None:
            return None
        matched = _like_match(str(l), str(r))
        return matched if op == "LIKE" else not matched

    if l is None or r is None:
        return None  # all remaining ops propagate NULL

    try:
        if op == "=":
            return l == r
        if op == "!=":
            return l != r
        if op == "<":
            return l < r
        if op == "<=":
            return l <= r
        if op == ">":
            return l > r
        if op == ">=":
            return l >= r
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            if r == 0:
                raise EvalError("division by zero")
            return l / r
    except TypeError:
        raise EvalError(f"type mismatch for {op}: {l!r} vs {r!r}")
    raise EvalError(f"unsupported operator {op}")


def _like_match(value: str, pattern: str) -> bool:
    """SQL LIKE: ``%`` matches any run, ``_`` matches one char."""
    pieces = []
    for ch in pattern:
        if ch == "%":
            pieces.append(".*")
        elif ch == "_":
            pieces.append(".")
        else:
            pieces.append(re.escape(ch))
    return re.fullmatch("".join(pieces), value, flags=re.DOTALL) is not None


def _eval_scalar_func(expr: ast.FuncCall, ctx: EvalContext) -> Any:
    if not expr.args:
        raise EvalError(f"function {expr.name} requires an argument")
    v = evaluate(expr.args[0], ctx)
    name = expr.name.upper()
    if name == "UPPER":
        return None if v is None else str(v).upper()
    if name == "LOWER":
        return None if v is None else str(v).lower()
    raise EvalError(f"unknown scalar function {name}")


def _eval_subquery_expr(expr: Any, ctx: EvalContext) -> Any:
    if ctx.subquery_runner is None:
        raise EvalError("subqueries require an execution context")
    runner = ctx.subquery_runner
    token = push_outer_context(ctx)
    try:
        if isinstance(expr, ast.Subquery):
            rows = runner(expr.query)
            if not rows:
                return None
            return rows[0][0]
        if isinstance(expr, ast.InSubquery):
            value = evaluate(expr.expr, ctx)
            rows = runner(expr.query)
            members = {row[0] for row in rows}
            if value is None:
                return None
            result = value in members
            return (not result) if expr.negated else result
        if isinstance(expr, ast.Exists):
            exists = bool(runner(expr.query))
            return (not exists) if expr.negated else exists
        raise EvalError(f"unsupported subquery form {expr!r}")
    finally:
        pop_outer_context(token)


def expr_references(expr: Any) -> set[str]:
    """Lower-cased unqualified column names referenced by an expression.

    Used by the planner for predicate pushdown.
    """
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, ast.ColumnRef):
            names.add(node.name.lower())
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand)
        elif isinstance(node, ast.BinaryOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.FuncCall):
            for a in node.args:
                walk(a)
        # subqueries are treated as opaque for pushdown purposes

    walk(expr)
    return names


def split_conjuncts(expr: Any) -> list[Any]:
    """Flatten a top-level AND chain into a list of predicates."""
    if isinstance(expr, ast.BinaryOp) and expr.op == "AND":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def combine_conjuncts(preds: list[Any]) -> Optional[Any]:
    if not preds:
        return None
    result = preds[0]
    for p in preds[1:]:
        result = ast.BinaryOp("AND", result, p)
    return result
