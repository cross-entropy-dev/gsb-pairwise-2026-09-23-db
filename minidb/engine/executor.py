"""Statement executor.

Read model
----------
Every statement gets an MVCC snapshot (READ COMMITTED: fresh per statement;
SERIALIZABLE: pinned at first read).  Scans walk B+ tree version chains and
keep only visible rows, so readers never block on writers or on the WAL.
SERIALIZABLE readers additionally take a table-level ``S`` lock (S2PL) which
closes phantoms.

Write model
-----------
Writers take an ``IX`` table lock then ``X`` row locks (escalation lives in
the lock manager).  The flow for UPDATE/DELETE is:

1. snapshot-scan candidate keys;
2. lock them all in a global key order (avirms lock-order deadlocks);
3. *re-snapshot* under READ COMMITTED so rows committed while we waited are
   visible, then re-check the predicate and mutate – this prevents lost
   updates without making readers block;
4. every physical mutation is WAL-logged before it is installed.

Abort needs no undo: new versions carry the aborting txn's id and are
invisible to everyone; GC reclaims them once the txn is old enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..sql import ast
from ..storage.schema import Column, TableSchema, TypeConversionError
from ..storage.table import Version
from ..txn.manager import Isolation, Transaction
from .errors import ExecutionError, IntegrityError
from .expressions import (
    EvalContext, SourceBinding, combine_conjuncts, evaluate, truthy,
)
from .planner import QueryPlan, contains_aggregate, plan_select

AGG_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


@dataclass
class ResultSet:
    columns: list[str]
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def scalar(self) -> Any:
        if not self.rows:
            return None
        return self.rows[0][0]


class Executor:
    def __init__(self, db) -> None:
        self.db = db

    # ================================================================== #
    # dispatch
    # ================================================================== #
    def execute(self, node: Any, txn: Transaction) -> ResultSet:
        if isinstance(node, ast.CreateTable):
            return self._exec_create(node, txn)
        if isinstance(node, ast.Insert):
            return self._exec_insert(node, txn)
        if isinstance(node, ast.Update):
            return self._exec_update(node, txn)
        if isinstance(node, ast.Delete):
            return self._exec_delete(node, txn)
        if isinstance(node, ast.SelectStmt):
            return self._exec_select(node, txn)
        if isinstance(node, ast.ShowTables):
            return ResultSet(["table"], [(n,) for n in self.db.table_names()])
        if isinstance(node, ast.Describe):
            return self._exec_describe(node)
        raise ExecutionError(f"cannot execute {type(node).__name__}")

    # ================================================================== #
    # DDL
    # ================================================================== #
    def _exec_create(self, node: ast.CreateTable, txn: Transaction) -> ResultSet:
        columns = [
            Column(name=c.name, type=c.type, length=c.length,
                   nullable=c.nullable, primary_key=c.primary_key)
            for c in node.columns
        ]
        schema = TableSchema(name=node.name, columns=columns)
        self.db.create_table(txn, schema)  # raises on invalid / duplicate
        return ResultSet(["status"], [("CREATED",)])

    def _exec_describe(self, node: ast.Describe) -> ResultSet:
        table = self.db.get_table(node.table)
        rows = []
        for c in table.schema.columns:
            type_str = c.type + (f"({c.length})" if c.length else "")
            rows.append((c.name, type_str,
                         "NO" if not c.nullable else "YES",
                         "YES" if c.primary_key else ""))
        return ResultSet(["column", "type", "nullable", "primary_key"], rows)

    # ================================================================== #
    # INSERT
    # ================================================================== #
    def _exec_insert(self, node: ast.Insert, txn: Transaction) -> ResultSet:
        table = self.db.get_table(node.table)
        schema = table.schema
        tname = node.table.lower()
        self.db.lock_manager.acquire_table(txn.txid, tname, "IX")
        self.db.txn_manager.ensure_wal_begin(txn)

        wanted: Optional[list[str]] = None
        if node.columns:
            wanted = [c.lower() for c in node.columns]
            unknown = [c for c in wanted if not schema.has_column(c)]
            if unknown:
                raise ExecutionError(
                    f"unknown column {unknown[0]!r} in table {node.table!r}"
                )
            if len(set(wanted)) != len(wanted):
                raise ExecutionError("duplicate column in INSERT column list")

        inserted = 0
        for raw_values in node.rows:
            expected = len(wanted) if wanted else len(schema.columns)
            if len(raw_values) != expected:
                raise ExecutionError(
                    f"column/value count mismatch in INSERT into {node.table!r}"
                )
            values = [evaluate(v, EvalContext([], None)) for v in raw_values]
            if wanted:
                full: list[Any] = [None] * len(schema.columns)
                for name, val in zip(wanted, values):
                    full[schema.index_of(name)] = val
                for col in schema.columns:
                    if not col.primary_key and not col.nullable:
                        if full[schema.index_of(col.name)] is None:
                            raise IntegrityError(
                                f"column {col.name!r} is NOT NULL"
                            )
            else:
                full = values
            try:
                row = schema.validate_and_convert(tuple(full))
            except TypeConversionError as exc:
                raise IntegrityError(str(exc)) from exc

            if table.has_user_pk:
                pk_col = schema.primary_key_columns[0]
                key = row[schema.index_of(pk_col.name)]
                if key is None:
                    raise IntegrityError(
                        f"primary key {pk_col.name!r} cannot be NULL"
                    )
                lm = self.db.lock_manager
                lm.acquire_row(txn.txid, tname, key, "X")
                snap = self.db.txn_manager.snapshot_for(txn)
                if table.visible_version(key, snap) is not None:
                    raise IntegrityError(
                        f"duplicate primary key {pk_col.name}={key!r} "
                        f"in {schema.name!r}"
                    )
            else:
                key = table.allocate_rowid()
                self.db.lock_manager.acquire_row(txn.txid, tname, key, "X")

            self.db.wal.log_insert(txn.txid, node.table, key, row)
            table.insert(txn.txid, row, key=key)
            inserted += 1
        txn.written_tables.add(tname)
        return ResultSet(["inserted"], [(inserted,)])

    # ================================================================== #
    # UPDATE
    # ================================================================== #
    def _exec_update(self, node: ast.Update, txn: Transaction) -> ResultSet:
        table = self.db.get_table(node.table)
        schema = table.schema
        tname = node.table.lower()
        lm = self.db.lock_manager
        lm.acquire_table(txn.txid, tname, "IX")
        self.db.txn_manager.ensure_wal_begin(txn)

        assign_idx = []
        for col_name, _ in node.assignments:
            if not schema.has_column(col_name):
                raise ExecutionError(f"unknown column {col_name!r}")
            assign_idx.append(schema.index_of(col_name))
        pk_idx = (schema.index_of(schema.primary_key_columns[0].name)
                  if table.has_user_pk else None)

        def make_ctx(row, snap):
            return EvalContext(
                [SourceBinding(tname, schema.column_names_lower(), row)],
                self._subquery_runner(txn, snap),
            )

        # 1. candidate keys under the statement snapshot
        snap = self.db.txn_manager.snapshot_for(txn)
        index_keys = self._where_pk_keys(table, node.where)
        if index_keys is not None:
            candidates = []
            for k in index_keys:
                v = table.visible_version(k, snap)
                if v is not None:
                    ctx = make_ctx(v.row, snap)
                    if node.where is None or truthy(evaluate(node.where, ctx)):
                        candidates.append(k)
        else:
            candidates = [
                key for key, row in table.scan(snap)
                if node.where is None
                or truthy(evaluate(node.where, make_ctx(row, snap)))
            ]
        # 2. lock in a stable global order
        candidates.sort(key=lambda k: (type(k).__name__, repr(k)))
        for key in candidates:
            lm.acquire_row(txn.txid, tname, key, "X")
        # 3. fresh snapshot after waiting (RC) so committed updates are seen
        snap = self._post_lock_snapshot(txn, snap)

        updated = 0
        for key in candidates:
            version = table.visible_version(key, snap)
            if version is None:
                continue
            current = version.row
            ctx = make_ctx(current, snap)
            if node.where is not None and not truthy(evaluate(node.where, ctx)):
                continue
            new_row = list(current)
            for idx, (_, expr) in zip(assign_idx, node.assignments):
                new_row[idx] = evaluate(expr, ctx)
            try:
                converted = schema.validate_and_convert(tuple(new_row))
            except TypeConversionError as exc:
                raise IntegrityError(str(exc)) from exc

            if pk_idx is None or converted[pk_idx] == current[pk_idx]:
                self.db.wal.log_update(txn.txid, node.table, key, converted)
                table.update(txn.txid, key, converted)
            else:
                self._move_primary_key(
                    table, txn, node.table, key, converted, snap
                )
            updated += 1
        txn.written_tables.add(tname)
        return ResultSet(["updated"], [(updated,)])

    def _move_primary_key(self, table, txn, tname, old_key, new_row, snap):
        schema = table.schema
        pk_col = schema.primary_key_columns[0]
        new_key = new_row[schema.index_of(pk_col.name)]
        lm = self.db.lock_manager
        lm.acquire_row(txn.txid, tname.lower(), new_key, "X")
        if table.visible_version(new_key, snap) is not None:
            raise IntegrityError(
                f"duplicate primary key {pk_col.name}={new_key!r} in {tname!r}"
            )
        self.db.wal.log_delete(txn.txid, tname, old_key)
        self.db.wal.log_insert(txn.txid, tname, new_key, new_row)
        table.delete(txn.txid, old_key)
        table.insert(txn.txid, new_row, key=new_key)

    # ================================================================== #
    # DELETE
    # ================================================================== #
    def _exec_delete(self, node: ast.Delete, txn: Transaction) -> ResultSet:
        table = self.db.get_table(node.table)
        schema = table.schema
        tname = node.table.lower()
        lm = self.db.lock_manager
        lm.acquire_table(txn.txid, tname, "IX")
        self.db.txn_manager.ensure_wal_begin(txn)

        def make_ctx(row, snap):
            return EvalContext(
                [SourceBinding(tname, schema.column_names_lower(), row)],
                self._subquery_runner(txn, snap),
            )

        snap = self.db.txn_manager.snapshot_for(txn)
        index_keys = self._where_pk_keys(table, node.where)
        if index_keys is not None:
            candidates = []
            for k in index_keys:
                v = table.visible_version(k, snap)
                if v is not None:
                    ctx = make_ctx(v.row, snap)
                    if node.where is None or truthy(evaluate(node.where, ctx)):
                        candidates.append(k)
        else:
            candidates = [
                key for key, row in table.scan(snap)
                if node.where is None
                or truthy(evaluate(node.where, make_ctx(row, snap)))
            ]
        candidates.sort(key=lambda k: (type(k).__name__, repr(k)))
        for key in candidates:
            lm.acquire_row(txn.txid, tname, key, "X")
        snap = self._post_lock_snapshot(txn, snap)

        deleted = 0
        for key in candidates:
            version = table.visible_version(key, snap)
            if version is None:
                continue
            row = version.row
            if node.where is not None and not truthy(
                evaluate(node.where, make_ctx(row, snap))
            ):
                continue
            self.db.wal.log_delete(txn.txid, node.table, key)
            if table.delete(txn.txid, key):
                deleted += 1
        txn.written_tables.add(tname)
        return ResultSet(["deleted"], [(deleted,)])

    def _post_lock_snapshot(self, txn, snap):
        """After acquiring locks, take a fresh statement snapshot under RC so
        rows committed while we waited are visible (prevents lost updates);
        SERIALIZABLE keeps its pinned snapshot (table S lock blocks phantoms).
        """
        if txn.isolation is Isolation.READ_COMMITTED:
            return self.db.txn_manager.take_snapshot(txn.txid)
        return snap

    # ================================================================== #
    # SELECT
    # ================================================================== #
    def _exec_select(self, node, txn, snap_override=None) -> ResultSet:
        if snap_override is not None:
            snap = snap_override
        elif node.from_table is not None and txn.isolation is Isolation.SERIALIZABLE:
            # Acquire the table S locks *before* pinning the snapshot: a
            # reader that had to wait for a writer must see that writer's
            # committed state (S2PL ordering), not a stale pre-wait snapshot.
            tables = [node.from_table] + [j.table for j in node.joins]
            for t in tables:
                self.db.lock_manager.acquire_table(txn.txid, t.lower(), "S")
            snap = self.db.txn_manager.snapshot_for(txn)
        else:
            snap = self.db.txn_manager.snapshot_for(txn)
        runner = self._subquery_runner(txn, snap)

        if node.from_table is None:
            result = self._project_no_from(node, runner)
        else:
            result = self._select_from(node, txn, snap, runner)

        if node.offset:
            result.rows = result.rows[node.offset:]
        if node.limit is not None:
            result.rows = result.rows[:node.limit]
        return result

    def _select_from(self, node, txn, snap, runner) -> ResultSet:
        tables = [node.from_table] + [j.table for j in node.joins]
        schemas = {t: self.db.get_table(t).schema for t in tables}
        plan = plan_select(
            node, {t: s.column_names_lower() for t, s in schemas.items()}
        )

        if txn.isolation is Isolation.SERIALIZABLE:
            for t in tables:
                self.db.lock_manager.acquire_table(txn.txid, t.lower(), "S")

        sources = self._build_sources(node, schemas)
        joined = self._nested_loop_join(node, plan, sources, snap, runner)

        if plan.where_remainder:
            pred = combine_conjuncts(plan.where_remainder)
            joined = [
                r for r in joined
                if truthy(evaluate(pred, EvalContext(sources, runner).with_rows(r)))
            ]

        if plan.has_aggregates:
            out_rows, out_cols, descs = self._group_and_aggregate(
                node, joined, sources, runner
            )
            if node.order_by:
                out_rows = self._order_groups(
                    node, out_rows, out_cols, descs, sources, runner
                )
            if node.distinct:
                out_rows = self._distinct(out_rows)
            return ResultSet(out_cols, out_rows)

        # ---- flat (non-aggregate) path ----
        # Evaluate every ORDER BY expression against the pre-projection
        # source row as a fallback.  Ordinals / output-column names are
        # preferred when they resolve after projection.
        projected_rows: list[tuple] = []
        src_keys: list[tuple] = []
        out_cols: list[str] = []
        for row in joined:
            ctx = EvalContext(sources, runner).with_rows(row)
            vals, cols = self._project(node.projections, ctx)
            if not out_cols:
                out_cols = cols
            projected_rows.append(tuple(vals))
            keys = []
            for expr, _d in node.order_by:
                if isinstance(expr, ast.Literal) and isinstance(expr.value, int):
                    keys.append(None)  # ordinal – resolved post-projection
                else:
                    try:
                        keys.append(evaluate(expr, ctx))
                    except Exception:
                        keys.append(None)
            src_keys.append(tuple(keys))

        if node.distinct:
            projected_rows, src_keys = self._distinct_pairs(projected_rows, src_keys)

        if node.order_by:
            projected_rows = self._order_flat(
                node, projected_rows, out_cols, src_keys
            )
        return ResultSet(out_cols, projected_rows)

    @staticmethod
    def _distinct(rows):
        return list(dict.fromkeys(rows))

    @staticmethod
    def _distinct_pairs(rows, payloads):
        seen: set = set()
        out_rows, out_payload = [], []
        for r, p in zip(rows, payloads):
            if r not in seen:
                seen.add(r)
                out_rows.append(r)
                out_payload.append(p)
        return out_rows, out_payload

    def _order_flat(self, node, rows, out_cols, src_keys):
        # Resolve each ORDER BY entry to either an output-tuple index or a
        # pre-projection source value (src_keys). Stable sort least->most
        # significant key.
        resolved = []
        for i, (expr, direction) in enumerate(node.order_by):
            desc = direction == "DESC"
            idx = None
            use_src = False
            if isinstance(expr, ast.Literal) and isinstance(expr.value, int):
                idx = expr.value - 1
            elif isinstance(expr, ast.ColumnRef) and expr.table is None:
                low = expr.name.lower()
                for j, c in enumerate(out_cols):
                    if c.lower() == low:
                        idx = j
                        break
                if idx is None:
                    use_src = True  # name not projected: use source value
            else:
                use_src = True     # qualified ref or arbitrary expression
            resolved.append((i, idx, use_src, desc))

        per_row: list[list] = [[] for _ in rows]
        for i, idx, use_src, desc in resolved:
            for r, row in enumerate(rows):
                if use_src:
                    v = src_keys[r][i]
                elif idx is not None and idx < len(row):
                    v = row[idx]
                else:
                    v = None
                per_row[r].append(_OrderKey(v, desc))
        order = sorted(range(len(rows)), key=lambda r: tuple(per_row[r]))
        return [rows[r] for r in order]

    # ------------------------------------------------------------------ #
    # joins
    # ------------------------------------------------------------------ #
    def _index_keys(self, table, scan_plan, snap):
        """If the pushed predicates include a primary-key equality
        (``pk = literal``), return the matching key(s); else None to signal
        a full scan.  This turns point lookups into O(log n) tree gets."""
        if not table.has_user_pk:
            return None
        pk_name = table.schema.primary_key_columns[0].name.lower()
        alias = scan_plan.alias
        keys = []
        found = False
        for conj in scan_plan.pushed:
            key = self._pk_eq_key(conj, pk_name, alias, table.schema)
            if key is not None:
                keys.append(key)
                found = True
        if not found:
            return None
        # all equalities must hold – intersect by using the most restrictive
        # (identical keys on same col, otherwise empty result)
        first = keys[0]
        for k in keys[1:]:
            if k != first:
                return []
        return [first]

    def _where_pk_keys(self, table, where):
        """Extract PK equality keys from a WHERE clause (possibly an AND
        chain). Returns None when the clause cannot be reduced to a point
        lookup, a list of keys otherwise (contradictory equalities -> [])."""
        if not table.has_user_pk or where is None:
            return None
        from .expressions import split_conjuncts
        pk_name = table.schema.primary_key_columns[0].name.lower()
        keys = []
        for conj in split_conjuncts(where):
            k = self._pk_eq_key(conj, pk_name,
                                table.schema.name.lower(), table.schema)
            if k is not None:
                keys.append(k)
        if not keys:
            return None
        first = keys[0]
        return [first] if all(k == first for k in keys[1:]) else []

    @staticmethod
    def _pk_eq_key(conj, pk_name, alias, schema):
        if not isinstance(conj, ast.BinaryOp) or conj.op != "=":
            return None

        def match(ref, lit):
            return (
                isinstance(ref, ast.ColumnRef)
                and ref.name.lower() == pk_name
                and (ref.table is None or ref.table.lower() == alias)
                and isinstance(lit, ast.Literal)
            )

        l, r = conj.left, conj.right
        if match(l, r):
            raw = r.value
        elif match(r, l):
            raw = l.value
        else:
            return None
        col = schema.column(pk_name)
        from ..storage.schema import convert_value
        try:
            return convert_value(col, raw)
        except Exception:
            return None

    def _nested_loop_join(self, node, plan, sources, snap, runner):
        def scan(scan_plan):
            table = self.db.get_table(scan_plan.table)
            pred = combine_conjuncts(scan_plan.pushed)
            out = []

            def accept(row):
                if pred is None:
                    return True
                ctx = EvalContext(
                    [SourceBinding(scan_plan.alias, scan_plan.columns, row)],
                    runner,
                )
                return truthy(evaluate(pred, ctx))

            idx_keys = self._index_keys(table, scan_plan, snap)
            if idx_keys is not None:
                for key in idx_keys:
                    version = table.visible_version(key, snap)
                    if version is not None and accept(version.row):
                        out.append(version.row)
            else:
                for _, row in table.scan(snap):
                    if accept(row):
                        out.append(row)
            return out

        left_rows = [(r,) for r in scan(plan.scans[0])]
        for jp in plan.joins:
            right_rows = scan(jp.scan)
            null_right = (None,) * len(jp.scan.columns)
            joined = []
            on_pred = jp.join_pred
            for lrow in left_rows:
                matched = False
                for rrow in right_rows:
                    candidate = lrow + (rrow,)
                    if on_pred is not None:
                        ctx = EvalContext(sources, runner).with_rows(candidate)
                        if not truthy(evaluate(on_pred, ctx)):
                            continue
                    joined.append(candidate)
                    matched = True
                if not matched and jp.kind == "LEFT":
                    joined.append(lrow + (null_right,))
            left_rows = joined
        return left_rows

    # ------------------------------------------------------------------ #
    # grouping / aggregation
    # ------------------------------------------------------------------ #
    def _group_and_aggregate(self, node, rows, sources, runner):
        groups: dict[Any, list[tuple]] = {}
        order: list[Any] = []
        key_exprs = node.group_by
        if key_exprs:
            for row in rows:
                ctx = EvalContext(sources, runner).with_rows(row)
                key = tuple(evaluate(e, ctx) for e in key_exprs)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(row)
        else:
            groups[()] = list(rows)
            order = [()]

        out_cols: list[str] = []
        final_rows: list[tuple] = []
        descs: list[tuple] = []  # aligned 1:1 with final_rows
        for key in order:
            group_rows = groups[key]
            null_row = tuple(None for _ in sources)
            first_ctx = (
                EvalContext(sources, runner).with_rows(group_rows[0])
                if group_rows
                else EvalContext(sources, runner).with_rows(null_row)
            )
            if node.having is not None and not truthy(self._eval_having(
                node.having, first_ctx, group_rows, sources, runner
            )):
                continue

            vals = []
            cols = []
            for proj in node.projections:
                val, name = self._project_one_agg(
                    proj, key_exprs, key, group_rows, first_ctx,
                    sources, runner,
                )
                vals.append(val)
                cols.append(name)
            if not out_cols:
                out_cols = cols
            final_rows.append(tuple(vals))
            descs.append((key, group_rows, first_ctx))
        return final_rows, out_cols, descs

    def _project_one_agg(self, proj, key_exprs, key, group_rows, first_ctx,
                         sources, runner):
        if isinstance(proj, ast.Star):
            raise ExecutionError("* cannot be combined with aggregation")
        if isinstance(proj, ast.FuncCall) and proj.name.upper() in AGG_FUNCS:
            return self._agg_value(proj, group_rows, sources, runner), agg_name(proj)
        if key_exprs:
            for i, e in enumerate(key_exprs):
                if expr_equiv(e, proj):
                    return key[i], expr_name(proj)
            return evaluate(proj, first_ctx), expr_name(proj)
        if not group_rows:
            return None, expr_name(proj)
        return evaluate(proj, first_ctx), expr_name(proj)

    def _eval_having(self, expr, first_ctx, group_rows, sources, runner):
        """Evaluate HAVING: aggregates computed over the group, everything
        else evaluated against the group's representative row."""
        if isinstance(expr, ast.FuncCall) and expr.name.upper() in AGG_FUNCS:
            return self._agg_value(expr, group_rows, sources, runner)
        if isinstance(expr, ast.UnaryOp):
            v = self._eval_having(
                expr.operand, first_ctx, group_rows, sources, runner
            )
            if expr.op == "NOT":
                return None if v is None else not v
            if expr.op == "-":
                return None if v is None else -v
            return v
        if isinstance(expr, ast.BinaryOp):
            l = self._eval_having(
                expr.left, first_ctx, group_rows, sources, runner
            )
            r = self._eval_having(
                expr.right, first_ctx, group_rows, sources, runner
            )
            return _apply_binary(expr.op, l, r)
        return evaluate(expr, first_ctx)

    def _agg_value(self, func, group_rows, sources, runner):
        name = func.name.upper()
        if func.star:
            if name != "COUNT":
                raise ExecutionError(f"{name}(*) is not valid")
            return len(group_rows)
        arg = func.args[0]
        values = []
        for row in group_rows:
            v = evaluate(arg, EvalContext(sources, runner).with_rows(row))
            if v is not None:
                values.append(v)
        if func.distinct:
            values = list(dict.fromkeys(values))
        if name == "COUNT":
            return len(values)
        if not values:
            return None
        if name == "SUM":
            return sum(values)
        if name == "AVG":
            return sum(values) / len(values)
        if name == "MIN":
            return min(values)
        if name == "MAX":
            return max(values)
        raise ExecutionError(f"unsupported aggregate {name}")

    # ------------------------------------------------------------------ #
    # ORDER BY for aggregate queries
    # ------------------------------------------------------------------ #
    def _order_groups(self, node, rows, out_cols, descs, sources, runner):
        resolved = []
        for i, (expr, direction) in enumerate(node.order_by):
            desc = direction == "DESC"
            idx = None
            special = False
            if isinstance(expr, ast.Literal) and isinstance(expr.value, int):
                idx = expr.value - 1
            elif isinstance(expr, ast.ColumnRef) and expr.table is None:
                low = expr.name.lower()
                for j, c in enumerate(out_cols):
                    if c.lower() == low:
                        idx = j
                        break
                if idx is None:
                    special = True
            else:
                special = True  # aggregate or group-key expression
            resolved.append((i, idx, special, desc))

        def special_value(i, expr, desc_tuple):
            gkey, group_rows, first_ctx = desc_tuple
            if isinstance(expr, ast.FuncCall) and expr.name.upper() in AGG_FUNCS:
                return self._agg_value(expr, group_rows, sources, runner)
            for k, e in enumerate(node.group_by):
                if expr_equiv(e, expr):
                    return gkey[k]
            try:
                return evaluate(expr, first_ctx)
            except Exception:
                return None

        per_row = [[] for _ in rows]
        for i, idx, special, desc in resolved:
            for r, row in enumerate(rows):
                if special:
                    v = special_value(i, node.order_by[i][0], descs[r])
                elif idx is not None and idx < len(row):
                    v = row[idx]
                else:
                    v = None
                per_row[r].append(_OrderKey(v, desc))
        order = sorted(range(len(rows)), key=lambda r: tuple(per_row[r]))
        return [rows[r] for r in order]

    # ------------------------------------------------------------------ #
    # projection
    # ------------------------------------------------------------------ #
    def _project(self, projections, ctx: EvalContext):
        vals: list[Any] = []
        cols: list[str] = []
        for proj in projections:
            if isinstance(proj, ast.Star):
                for b in ctx.bindings:
                    if proj.table is not None and b.alias != proj.table.lower():
                        continue
                    for ci, cname in enumerate(b.columns):
                        vals.append(None if b.row is None else b.row[ci])
                        cols.append(cname)
                continue
            vals.append(evaluate(proj, ctx))
            cols.append(expr_name(proj))
        return vals, cols

    def _project_no_from(self, node, runner) -> ResultSet:
        ctx = EvalContext([], runner)
        vals, cols = self._project(node.projections, ctx)
        return ResultSet(cols, [tuple(vals)])

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_sources(self, node, schemas) -> list[SourceBinding]:
        sources = [SourceBinding(
            (node.from_alias or node.from_table).lower(),
            schemas[node.from_table].column_names_lower(), None,
        )]
        for js in node.joins:
            sources.append(SourceBinding(
                (js.alias or js.table).lower(),
                schemas[js.table].column_names_lower(), None,
            ))
        return sources

    def _subquery_runner(self, txn, snap):
        def run(query: ast.SelectStmt) -> list[tuple]:
            return self._exec_select(query, txn, snap_override=snap).rows
        return run


# ---------------------------------------------------------------------- #
# small utilities
# ---------------------------------------------------------------------- #
class _OrderKey:
    """Sort wrapper: ASC -> NULLS LAST, DESC -> NULLS FIRST, mixed types
    fall back to (type-name, str-value) ordering."""

    __slots__ = ("key",)

    def __init__(self, v, desc: bool) -> None:
        isnull = v is None
        if desc:
            null_rank = 0 if isnull else 1          # NULLS FIRST
        else:
            null_rank = 1 if isnull else 0          # NULLS LAST
        self.key = (null_rank, _Cmp(v, desc))

    def __lt__(self, other: "_OrderKey") -> bool:
        return self.key < other.key

    def __eq__(self, other) -> bool:
        return isinstance(other, _OrderKey) and self.key == other.key


class _Cmp:
    __slots__ = ("v", "desc")

    def __init__(self, v, desc: bool) -> None:
        self.v = v
        self.desc = desc

    def _cmp(self, other: "_Cmp") -> int:
        a, b = self.v, other.v
        if a is None or b is None:
            return 0
        try:
            if a < b:
                return -1
            if a > b:
                return 1
            return 0
        except TypeError:
            sa, sb = f"{type(a).__name__}:{a}", f"{type(b).__name__}:{b}"
            if sa < sb:
                return -1
            if sa > sb:
                return 1
            return 0

    def __lt__(self, other: "_Cmp") -> bool:
        c = self._cmp(other)
        return c > 0 if self.desc else c < 0

    def __eq__(self, other) -> bool:
        return self._cmp(other) == 0


def _apply_binary(op: str, l, r):
    if op == "AND":
        if l is False or r is False:
            return False
        if l is None or r is None:
            return None
        return bool(l and r)
    if op == "OR":
        if l is True or r is True:
            return True
        if l is None or r is None:
            return None
        return bool(l or r)
    if l is None or r is None:
        return None
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
            raise ExecutionError("division by zero")
        return l / r
    raise ExecutionError(f"unsupported operator in HAVING: {op}")


def expr_name(expr) -> str:
    if isinstance(expr, ast.ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if isinstance(expr, ast.FuncCall):
        inner = "*" if expr.star else ", ".join(expr_name(a) for a in expr.args)
        prefix = "DISTINCT " if expr.distinct else ""
        return f"{expr.name}({prefix}{inner})"
    return "expr"


def agg_name(func: ast.FuncCall) -> str:
    if func.star:
        return f"{func.name}(*)"
    return f"{func.name}({expr_name(func.args[0])})"


def expr_equiv(a, b) -> bool:
    if isinstance(a, ast.ColumnRef) and isinstance(b, ast.ColumnRef):
        return (a.name.lower() == b.name.lower()
                and (a.table or "").lower() == (b.table or "").lower())
    return a == b
