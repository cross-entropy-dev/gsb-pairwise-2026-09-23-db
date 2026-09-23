"""Tiny rule-based query planner.

The only rewrite performed today is **predicate pushdown**: the conjuncts of
``WHERE`` / ``JOIN ... ON`` that reference columns of a single base table are
pushed into that table's scan, so non-matching rows never enter the join
pipeline.  Conjuncts spanning multiple tables stay in the join/filter stage.

The planner also tags aggregate projections and GROUP BY keys for the
executor, and exposes a human-readable plan (used by ``EXPLAIN`` and tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..sql import ast
from .expressions import expr_references, split_conjuncts


@dataclass
class ScanPlan:
    table: str
    alias: str
    columns: list[str]                      # lower-cased names
    pushed: list[Any] = field(default_factory=list)


@dataclass
class JoinPlan:
    scan: ScanPlan
    kind: str                               # INNER / LEFT
    pushed: list[Any] = field(default_factory=list)  # single-table ON bits
    join_pred: Optional[Any] = None         # cross-table ON remainder


@dataclass
class QueryPlan:
    scans: list[ScanPlan]
    joins: list[JoinPlan]
    where_remainder: list[Any]
    having: Any
    has_aggregates: bool
    description: str

    def explain(self) -> str:
        return self.description


def _alias_columns(plan: ScanPlan) -> set[str]:
    """Column-qualified and bare names the scan can resolve on its own."""
    names = set(plan.columns)
    names.add(plan.alias.lower())
    return names


def _conjunct_is_local(conj: Any, scan: ScanPlan) -> bool:
    """True iff ``conj`` references only columns belonging to ``scan``."""
    if isinstance(conj, ast.Subquery) or _contains_subquery(conj):
        return False
    local_cols = set(scan.columns)
    alias = scan.alias.lower()

    def walk(node: Any) -> Optional[bool]:
        if isinstance(node, ast.ColumnRef):
            if node.table is not None and node.table.lower() != alias:
                return False
            return node.name.lower() in local_cols
        if isinstance(node, ast.UnaryOp):
            return walk(node.operand)
        if isinstance(node, ast.BinaryOp):
            return walk(node.left) and walk(node.right)
        if isinstance(node, ast.FuncCall):
            return all(walk(a) for a in node.args)
        return True  # literals

    return bool(walk(conj))


def _contains_subquery(expr: Any) -> bool:
    found = False

    def walk(node: Any) -> None:
        nonlocal found
        if isinstance(node, (ast.Subquery, ast.InSubquery, ast.Exists)):
            found = True
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand)
        elif isinstance(node, ast.BinaryOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.FuncCall):
            for a in node.args:
                walk(a)

    walk(expr)
    return found


def contains_aggregate(expr: Any) -> bool:
    if isinstance(expr, ast.FuncCall) and expr.name.upper() in (
        "COUNT", "SUM", "AVG", "MIN", "MAX"
    ):
        return True
    found = False
    if isinstance(expr, ast.UnaryOp):
        return contains_aggregate(expr.operand)
    if isinstance(expr, ast.BinaryOp):
        return contains_aggregate(expr.left) or contains_aggregate(expr.right)
    if isinstance(expr, ast.FuncCall):
        found = any(contains_aggregate(a) for a in expr.args)
    return found


def plan_select(stmt: ast.SelectStmt, schemas: dict[str, list[str]]) -> QueryPlan:
    """Build a :class:`QueryPlan`.

    ``schemas`` maps each table name in the query to its lower-cased column
    list.
    """
    base_alias = (stmt.from_alias or stmt.from_table or "").lower()
    base_scan: Optional[ScanPlan]
    if stmt.from_table:
        base_scan = ScanPlan(
            table=stmt.from_table,
            alias=base_alias,
            columns=list(schemas[stmt.from_table]),
        )
        scans = [base_scan]
    else:
        base_scan = None
        scans = []
    join_plans: list[JoinPlan] = []

    for js in stmt.joins:
        j_alias = (js.alias or js.table).lower()
        scan = ScanPlan(
            table=js.table, alias=j_alias, columns=list(schemas[js.table])
        )
        scans.append(scan)
        join_plans.append(JoinPlan(scan=scan, kind=js.kind))

    # distribute WHERE conjuncts
    remaining: list[Any] = []
    for conj in split_conjuncts(stmt.where):
        placed = False
        if not _contains_subquery(conj):
            # LEFT JOIN's right table cannot have WHERE predicates pushed
            # safely in general (they would turn the outer join inner-ish);
            # only push into the base / inner-joined scans.
            for scan in scans:
                is_preserved = any(
                    jp.scan is scan and jp.kind == "LEFT" for jp in join_plans
                )
                if not is_preserved and _conjunct_is_local(conj, scan):
                    scan.pushed.append(conj)
                    placed = True
                    break
        if not placed:
            remaining.append(conj)

    # distribute ON conjuncts
    for js, jp in zip(stmt.joins, join_plans):
        on_remaining: list[Any] = []
        for conj in split_conjuncts(js.on):
            if _conjunct_is_local(conj, jp.scan):
                jp.pushed.append(conj)
            elif (jp.kind == "INNER" and base_scan is not None
                  and _conjunct_is_local(conj, base_scan)):
                base_scan.pushed.append(conj)
            else:
                on_remaining.append(conj)
        jp.join_pred = _and_chain(on_remaining)

    has_aggs = stmt.group_by or any(
        not isinstance(p, ast.Star) and contains_aggregate(p)
        for p in stmt.projections
    ) or (stmt.having is not None)

    description = _describe(base_scan, join_plans, remaining, stmt, has_aggs)
    return QueryPlan(
        scans=scans,
        joins=join_plans,
        where_remainder=remaining,
        having=stmt.having,
        has_aggregates=has_aggs,
        description=description,
    )


def _and_chain(preds: list[Any]) -> Optional[Any]:
    if not preds:
        return None
    out = preds[0]
    for p in preds[1:]:
        out = ast.BinaryOp("AND", out, p)
    return out


def _describe(base, joins, remaining, stmt, has_aggs) -> str:
    lines = []
    if base:
        lines.append(
            f"Scan {base.table} as {base.alias}"
            + (f" | filter pushed: {len(base.pushed)}" if base.pushed else "")
        )
    for jp in joins:
        s = jp.scan
        lines.append(
            f"{jp.kind} JOIN {s.table} as {s.alias}"
            + (f" | filter pushed: {len(jp.pushed)}" if jp.pushed else "")
        )
    if remaining:
        lines.append(f"Filter | {len(remaining)} cross-table/residual predicate(s)")
    if stmt.group_by:
        lines.append(f"GroupBy | {len(stmt.group_by)} key(s)")
    if stmt.having:
        lines.append("Having")
    if stmt.distinct:
        lines.append("Distinct")
    if stmt.order_by:
        lines.append(f"OrderBy | {len(stmt.order_by)} key(s)")
    if stmt.limit is not None:
        lines.append(f"Limit {stmt.limit}"
                     + (f" offset {stmt.offset}" if stmt.offset else ""))
    if has_aggs and not stmt.group_by:
        lines.insert(1, "Aggregate | scalar aggregation")
    return "\n".join(lines)
