"""SQL statement executor.

SELECT pipeline::

    bind names -> plan scans (predicate pushdown / PK point lookup)
        -> nested-loop joins (PK equi-join => B+ tree index lookup)
        -> WHERE  -> GROUP BY / aggregates -> HAVING
        -> SELECT list -> DISTINCT -> ORDER BY -> OFFSET / LIMIT

Every read is an MVCC snapshot read.  Under SERIALIZABLE the transaction also
takes S row/table locks (X table lock for predicate UPDATE/DELETE), which makes
schedules conflict-serializable including phantom protection.
"""

from ..sql import ast_nodes as ast
from ..errors import (ExecutionError, TableNotFoundError, ColumnNotFoundError,
                      DuplicateColumnError, ConstraintViolationError)
from ..storage.table import Schema, Column as StoreColumn
from .evaluator import Evaluator, AGGREGATES
from .optimizer import Optimizer, split_conjunction
from .types import (sql_truth, coerce_value, aggregate_init, aggregate_step,
                    aggregate_final)


class QueryResult:
    def __init__(self, columns=None, rows=None, rowcount=0, message=None):
        self.columns = columns or []
        self.rows = rows or []            # list[dict] label -> value
        self.rowcount = rowcount
        self.message = message

    def dicts(self):
        return [{c: r.get(c) for c in self.columns} for r in self.rows]

    def tuples(self):
        return [tuple(r.get(c) for c in self.columns) for r in self.rows]


class QueryExecutor:
    def __init__(self, db):
        self.db = db
        self.tm = db.tm
        self.optimizer = Optimizer()

    # ============================================================== dispatch

    def execute(self, node, txn):
        if isinstance(node, ast.CreateTable):
            return self._create_table(node, txn)
        if isinstance(node, ast.DropTable):
            return self._drop_table(node, txn)
        if isinstance(node, ast.Insert):
            return self._insert(node, txn)
        if isinstance(node, ast.Update):
            return self._update(node, txn)
        if isinstance(node, ast.Delete):
            return self._delete(node, txn)
        if isinstance(node, ast.Select):
            rows, labels = self.execute_select(node, txn, with_labels=True)
            return QueryResult(columns=labels, rows=rows, rowcount=len(rows))
        raise ExecutionError(f"unsupported statement {type(node).__name__}")

    # =================================================================== DDL

    def _create_table(self, node, txn):
        if node.if_not_exists and node.name in self.db.visible_tables(txn):
            return QueryResult(
                message=f"CREATE TABLE {node.name} (skipped, already exists)")
        names = [c.name for c in node.columns]
        if len(set(names)) != len(names):
            raise DuplicateColumnError("duplicate column name in CREATE TABLE")
        cols = []
        pk_cols = []
        for cd in node.columns:
            default = None
            if cd.default is not None:
                if not isinstance(cd.default, ast.Literal):
                    raise ExecutionError("DEFAULT only supports constants")
                default = cd.default.value
            col = StoreColumn(
                name=cd.name, type_name=cd.type_name, length=cd.length,
                not_null=cd.not_null, primary_key=cd.primary_key,
                unique=cd.unique or cd.primary_key, default=default,
                autoincrement=cd.autoincrement)
            cols.append(col)
            if cd.primary_key:
                pk_cols.append(cd.name)
        if len(pk_cols) > 1 and any(c.autoincrement for c in cols):
            raise ExecutionError("AUTOINCREMENT requires a single-column PK")
        schema = Schema(name=node.name, columns=cols, pk_columns=pk_cols)
        self.tm.create_table(txn, schema)
        return QueryResult(message=f"CREATE TABLE {node.name}")

    def _drop_table(self, node, txn):
        if node.name not in self.db.visible_tables(txn):
            if node.if_exists:
                return QueryResult(message="DROP TABLE (skipped)")
            raise TableNotFoundError(f"unknown table {node.name!r}")
        self.tm.drop_table(txn, node.name)
        return QueryResult(message=f"DROP TABLE {node.name}")

    # ================================================================ INSERT

    def _insert(self, node, txn):
        table = self.tm.require_table(txn, node.table)
        count = 0
        for value_exprs in node.rows:
            row = self._build_insert_row(table, node, value_exprs, txn)
            self.tm.insert_version(txn, table, table.make_key(row), row)
            count += 1
        return QueryResult(
            rowcount=count,
            message=f"INSERT {count} row{'s' if count != 1 else ''}")

    def _build_insert_row(self, table, node, value_exprs, txn):
        schema = table.schema
        target_cols = node.columns or schema.column_names
        if node.columns is not None:
            unknown = [c for c in target_cols if schema.column(c) is None]
            if unknown:
                raise ColumnNotFoundError(
                    f"unknown column {unknown[0]!r} in table {table.name!r}")
        if len(value_exprs) != len(target_cols):
            raise ExecutionError(
                f"{table.name}: expected {len(target_cols)} values, "
                f"got {len(value_exprs)}")

        evaluator = Evaluator(self, txn)
        raw = {}
        for col_name, expr in zip(target_cols, value_exprs):
            raw[col_name] = evaluator.eval(expr, {})

        row = {}
        for col in schema.columns:
            if col.name in raw:
                value = raw[col.name]
            elif col.autoincrement:
                value = table.alloc_auto(col.name)
            else:
                value = col.default
            row[col.name] = coerce_value(value, col)

        if not schema.pk_columns:
            row["_rowid_"] = table.alloc_rowid()

        for col in schema.columns:
            if col.not_null and row.get(col.name) is None:
                raise ConstraintViolationError(
                    f"null value in NOT NULL column {col.name!r}")
        return row

    # ================================================================ UPDATE

    def _update(self, node, txn):
        table = self.tm.require_table(txn, node.table)
        snapshot = self.tm.snapshot_for_statement(txn)
        evaluator = Evaluator(self, txn)

        unknown = [c for c, _ in node.assignments
                   if table.schema.column(c) is None]
        if unknown:
            raise ColumnNotFoundError(f"unknown column {unknown[0]!r}")

        # PK equality predicate -> B+ tree point access instead of a scan
        # (SERIALIZABLE then needs only IX/X row locks, no predicate lock).
        point_keys = self._point_target_keys(table, node.where, evaluator)
        if point_keys is None and txn.serializable:
            # Full/predicate write: X table lock blocks concurrent inserts
            # (phantom protection under SERIALIZABLE).
            self.tm.lm.lock_table(txn.txn_id, table.name, "X")

        candidate_keys = point_keys
        if candidate_keys is None:
            candidate_keys = []
            for key, version in table.scan_visible(snapshot, txn.txn_id):
                row = self._make_row(table, table.name, version.data)
                if node.where is None or \
                        sql_truth(evaluator.eval(node.where, row)) is True:
                    candidate_keys.append(key)

        count = 0
        for key in candidate_keys:
            # Acquire the X lock first, then perform a *current read*: under
            # READ COMMITTED the row may have been changed by a transaction
            # that committed while we located it. Evaluating against anything
            # but the latest committed version would lose updates.
            self.tm._prepare_write(txn, table, key)
            current = self.tm.current_version(txn, table, key)
            if current is None:
                continue
            cur_row = self._make_row(table, table.name, current.data)
            if node.where is not None and \
                    sql_truth(evaluator.eval(node.where, cur_row)) is not True:
                continue
            data = dict(current.data)
            for col_name, expr in node.assignments:
                value = evaluator.eval(expr, cur_row)
                data[col_name] = coerce_value(
                    value, table.schema.column(col_name))
            for pk in table.pk_columns:
                if data.get(pk) != current.data.get(pk):
                    raise ConstraintViolationError(
                        "primary key values cannot be updated")
            self.tm.update_version(txn, table, key, data, wal=True)
            count += 1
        return QueryResult(
            rowcount=count,
            message=f"UPDATE {count} row{'s' if count != 1 else ''}")

    def _point_target_keys(self, table, where, evaluator):
        """Return PK values for a constant PK equality/IN predicate.

        Returns ``None`` when the WHERE is not a pure key predicate and the
        caller must fall back to a scan.  The residual WHERE is re-checked on
        the current row by callers, so extracting key conjuncts is safe even
        when extra conditions are present.
        """
        if where is None or not table.pk_columns:
            return None
        conjuncts = split_conjunction(where)
        values = {}  # pk column -> set of constant values
        for cond in conjuncts:
            if isinstance(cond, ast.BinaryOp) and cond.op == "=":
                for col_expr, val_expr in ((cond.left, cond.right),
                                           (cond.right, cond.left)):
                    if isinstance(col_expr, ast.Column) and \
                            col_expr.name in table.pk_columns and \
                            isinstance(val_expr, ast.Literal):
                        values.setdefault(col_expr.name, set()).add(
                            val_expr.value)
            elif isinstance(cond, ast.InList) and not cond.negated and \
                    isinstance(cond.expr, ast.Column) and \
                    cond.expr.name in table.pk_columns and \
                    all(isinstance(v, ast.Literal) for v in cond.values):
                values.setdefault(cond.expr.name, set()).update(
                    v.value for v in cond.values)
            # Any other conjunct (non-PK filter, OR, ...) is simply left in
            # the WHERE and re-evaluated per fetched row.

        if not all(pk in values for pk in table.pk_columns):
            return None
        product_size = 1
        for pk in table.pk_columns:
            product_size *= len(values[pk])
        if product_size == 0 or product_size > 100:
            return None
        if len(table.pk_columns) == 1:
            return list(values[table.pk_columns[0]])
        # Composite PK: cartesian product in declared column order.
        result = [()]
        for pk in table.pk_columns:
            result = [base + (v,) for base in result
                      for v in values[pk]]
        return result

    # ================================================================ DELETE

    def _delete(self, node, txn):
        table = self.tm.require_table(txn, node.table)
        snapshot = self.tm.snapshot_for_statement(txn)
        evaluator = Evaluator(self, txn)

        point_keys = self._point_target_keys(table, node.where, evaluator)
        if point_keys is None and txn.serializable:
            # Full/predicate write: X table lock blocks concurrent inserts
            # (phantom protection under SERIALIZABLE).
            self.tm.lm.lock_table(txn.txn_id, table.name, "X")

        if point_keys is not None:
            keys = point_keys
        else:
            keys = []
            for key, version in table.scan_visible(snapshot, txn.txn_id):
                row = self._make_row(table, table.name, version.data)
                if node.where is None or \
                        sql_truth(evaluator.eval(node.where, row)) is True:
                    keys.append(key)

        count = 0
        for key in keys:
            if point_keys is not None:
                # Current read + residual WHERE re-check under the X lock.
                self.tm._prepare_write(txn, table, key)
                current = self.tm.current_version(txn, table, key)
                if current is None:
                    continue
                if node.where is not None:
                    cur_row = self._make_row(table, table.name, current.data)
                    if sql_truth(evaluator.eval(node.where, cur_row)) is not True:
                        continue
            if self.tm.delete_version(txn, table, key):
                count += 1
        return QueryResult(
            rowcount=count,
            message=f"DELETE {count} row{'s' if count != 1 else ''}")

    # ================================================================ SELECT

    def execute_select(self, select, txn, with_labels=False, outer_row=None):
        evaluator = Evaluator(self, txn, outer_row=outer_row or {})
        self.optimizer.optimize(select, evaluator)

        bindings = []
        pushed = {}
        if select.source is not None:
            bindings = self._bind_source(select.source, txn)
            qmap = self._qualifier_map(bindings)
            self._resolve_query_columns(select, qmap, bindings)
            pushed = self._plan_pushdown(select, bindings)

        snapshot = self.tm.snapshot_for_statement(txn)

        if select.source is None:
            input_rows = [{}]
        else:
            input_rows = list(
                self._run_source(select.source, txn, snapshot, pushed,
                                 evaluator))

        # WHERE (residual after pushdown)
        if isinstance(select.where, ast.Literal):
            if sql_truth(select.where.value) is False:
                input_rows = []
        elif select.where is not None:
            input_rows = [r for r in input_rows
                          if sql_truth(evaluator.eval(select.where, r)) is True]

        items, labels = self._expand_items(select.items, bindings)
        aggregates = self._find_aggregates(items, select.having)

        if aggregates or select.group_by:
            input_rows = self._aggregate(
                items, labels, aggregates, select, input_rows, evaluator)
        else:
            projected = []
            for r in input_rows:
                out = {label: evaluator.eval(item.expr, r)
                       for item, label in zip(items, labels)}
                projected.append(self._merge(r, out))
            input_rows = projected

        if select.distinct:
            seen, deduped = set(), []
            for r in input_rows:
                sig = tuple(_hashable(r.get(label)) for label in labels)
                if sig not in seen:
                    seen.add(sig)
                    deduped.append(r)
            input_rows = deduped

        if select.order_by:
            input_rows = self._order(
                select.order_by, input_rows, labels, evaluator, bindings)

        offset = int(evaluator.eval(select.offset, {})) \
            if select.offset is not None else 0
        limit = int(evaluator.eval(select.limit, {})) \
            if select.limit is not None else None
        if offset:
            input_rows = input_rows[offset:]
        if limit is not None:
            input_rows = input_rows[:limit]

        rows = [{label: r.get(label) for label in labels} for r in input_rows]
        return (rows, labels) if with_labels else rows

    # ------------------------------------------------------- source binding

    def _bind_source(self, source, txn):
        refs = []

        def walk(node):
            if isinstance(node, ast.TableRef):
                refs.append(node)
            elif isinstance(node, ast.Join):
                walk(node.left)
                walk(node.right)

        walk(source)
        bindings = []
        for ref in refs:
            bindings.append(
                (ref.alias or ref.name,
                 self.tm.require_table(txn, ref.name), ref))
        return bindings

    def _qualifier_map(self, bindings):
        qmap, names = {}, set()
        for alias, table, ref in bindings:
            if alias in names:
                raise ExecutionError(f"duplicate table alias {alias!r}")
            names.add(alias)
            qmap[alias] = alias
            if ref.alias is None and table.name not in names:
                qmap[table.name] = alias
                names.add(table.name)
        return qmap

    def _resolve_query_columns(self, select, qmap, bindings):
        output_labels = {
            (i.alias or self._default_label(i.expr)).lower()
            for i in select.items
            if not isinstance(i.expr, (ast.Star, ast.QualifiedStar))}

        def owners(name):
            return [alias for alias, table, _ in bindings
                    if table.schema.column(name) is not None]

        def resolve(node, allow_output_label=False):
            if isinstance(node, ast.Column):
                if node.table is not None:
                    if node.table not in qmap:
                        # Qualified name from an enclosing (correlated) query.
                        return
                    node.table = qmap[node.table]
                    return
                if allow_output_label and node.name.lower() in output_labels \
                        and not owners(node.name):
                    return  # keep bare name; resolves against output row
                found = owners(node.name)
                if len(found) == 1:
                    node.table = found[0]
                elif len(found) > 1:
                    raise ExecutionError(
                        f"column {node.name!r} is ambiguous; qualify it")
                # Zero owners: unqualified correlated column, left bare for
                # run-time resolution against the outer row.

        def walk(node):
            if isinstance(node, (ast.Subquery, ast.Exists, ast.InSubquery)):
                return
            if isinstance(node, ast.Column):
                resolve(node)
                return
            for field_name in getattr(node, "__dataclass_fields__", {}):
                val = getattr(node, field_name)
                if hasattr(val, "__dataclass_fields__"):
                    walk(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, "__dataclass_fields__"):
                            walk(item)
                        elif isinstance(item, tuple):
                            for part in item:
                                if hasattr(part, "__dataclass_fields__"):
                                    walk(part)

        if select.where is not None:
            walk(select.where)
        if select.having is not None:
            walk(select.having)
        for item in select.items:
            walk(item.expr)
        for g in select.group_by:
            walk(g)
        for ob in select.order_by:
            if isinstance(ob.expr, ast.Column) and ob.expr.table is None \
                    and ob.expr.name.lower() in output_labels:
                continue
            walk(ob.expr)
        self._walk_joins(select.source, walk)

    def _walk_joins(self, source, walk):
        if isinstance(source, ast.Join):
            if source.on is not None:
                walk(source.on)
            self._walk_joins(source.left, walk)
            self._walk_joins(source.right, walk)

    # -------------------------------------------------- predicate pushdown

    def _plan_pushdown(self, select, bindings):
        preserved = self._preserved_aliases(select.source)
        pushed = {alias: [] for alias, _, _ in bindings}
        if select.where is not None:
            for cond in split_conjunction(select.where):
                if isinstance(cond, ast.Literal):
                    continue
                aliases = {t for t, _ in self._column_refs(cond)}
                if len(aliases) == 1:
                    alias = next(iter(aliases))
                    if alias in preserved:
                        pushed[alias].append(cond)
        self._push_on_predicates(select.source, pushed)
        return pushed

    def _push_on_predicates(self, source, pushed):
        if not isinstance(source, ast.Join):
            return
        right_aliases = set()
        self._collect_aliases(source.right, right_aliases)
        if source.on is not None:
            for cond in split_conjunction(source.on):
                aliases = {t for t, _ in self._column_refs(cond)}
                if len(aliases) == 1 and aliases <= right_aliases:
                    pushed.setdefault(next(iter(aliases)), []).append(cond)
        self._push_on_predicates(source.left, pushed)
        self._push_on_predicates(source.right, pushed)

    def _collect_aliases(self, node, out):
        if isinstance(node, ast.TableRef):
            out.add(node.alias or node.name)
        elif isinstance(node, ast.Join):
            self._collect_aliases(node.left, out)
            self._collect_aliases(node.right, out)

    def _preserved_aliases(self, source):
        out = set()

        def walk(node, preserved):
            if isinstance(node, ast.TableRef):
                if preserved:
                    out.add(node.alias or node.name)
            elif isinstance(node, ast.Join):
                if node.join_type == "LEFT":
                    walk(node.left, preserved)
                    walk(node.right, False)
                elif node.join_type == "RIGHT":
                    walk(node.left, False)
                    walk(node.right, preserved)
                else:
                    walk(node.left, preserved)
                    walk(node.right, preserved)

        walk(source, True)
        return out

    def _column_refs(self, node):
        refs = set()

        def walk(n):
            if isinstance(n, ast.Column):
                refs.add((n.table, n.name))
                return
            if isinstance(n, (ast.Subquery, ast.Exists, ast.InSubquery)):
                return
            for field_name in getattr(n, "__dataclass_fields__", {}):
                val = getattr(n, field_name)
                if hasattr(val, "__dataclass_fields__"):
                    walk(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, "__dataclass_fields__"):
                            walk(item)
                        elif isinstance(item, tuple):
                            for part in item:
                                if hasattr(part, "__dataclass_fields__"):
                                    walk(part)

        walk(node)
        return refs

    # ------------------------------------------------------------- scanning

    def _run_source(self, source, txn, snapshot, pushed, evaluator):
        if isinstance(source, ast.TableRef):
            alias = source.alias or source.name
            table = self.tm.require_table(txn, source.name)
            yield from self._scan_table(
                txn, table, alias, snapshot, pushed.get(alias, []),
                evaluator, outer_row={})
            return
        if isinstance(source, ast.Join):
            yield from self._run_join(source, txn, snapshot, pushed, evaluator)
            return
        raise ExecutionError("invalid FROM source")

    def _scan_table(self, txn, table, alias, snapshot, filters, evaluator,
                    outer_row, pk_param=None):
        pk_col = table.pk_columns[0] if len(table.pk_columns) == 1 else None
        const_key = None
        remaining = []
        for f in filters:
            key_expr = self._pk_equality(f, alias, pk_col, allow_columns=False)
            if key_expr is not None:
                const_key = evaluator.eval(key_expr, outer_row)
            else:
                remaining.append(f)

        def make_version_row(key, version):
            row = self._make_row(table, alias, version.data)
            if all(sql_truth(evaluator.eval(f, row)) is True
                   for f in remaining):
                return row
            return None

        if pk_param is not None:
            # Index nested-loop join: one point lookup per outer row.
            key = evaluator.eval(pk_param, outer_row)
            if key is None:
                return
            self.tm.lock_for_read(txn, table, key)
            version = table.visible_version(key, snapshot, txn.txn_id)
            if version is not None:
                row = make_version_row(key, version)
                if row is not None:
                    yield row
            return

        if const_key is not None:
            self.tm.lock_for_read(txn, table, const_key)
            version = table.visible_version(const_key, snapshot, txn.txn_id)
            if version is not None:
                row = make_version_row(const_key, version)
                if row is not None:
                    yield row
            return

        self.tm.lock_predicate_read(txn, table)
        for key, version in table.scan_visible(snapshot, txn.txn_id):
            row = make_version_row(key, version)
            if row is not None:
                yield row

    def _pk_equality(self, cond, alias, pk_col, allow_columns):
        if pk_col is None or not isinstance(cond, ast.BinaryOp) \
                or cond.op != "=":
            return None

        def match(col, expr):
            if isinstance(col, ast.Column) and col.table == alias \
                    and col.name == pk_col:
                if allow_columns:
                    return expr
                return None if self._column_refs(expr) else expr
            return None

        return match(cond.left, cond.right) or match(cond.right, cond.left)

    # ---------------------------------------------------------------- joins

    def _run_join(self, join, txn, snapshot, pushed, evaluator):
        # Parser builds left-deep joins, so the right side is a base table.
        right_ref = join.right if isinstance(join.right, ast.TableRef) else None
        pk_param = None
        if right_ref is not None and join.on is not None:
            right_alias = right_ref.alias or right_ref.name
            right_table = self.tm.require_table(txn, right_ref.name)
            pk_col = right_table.pk_columns[0] \
                if len(right_table.pk_columns) == 1 else None
            for cond in split_conjunction(join.on):
                expr = self._pk_equality(cond, right_alias, pk_col,
                                         allow_columns=True)
                if expr is not None and \
                        not self._refs_alias(expr, right_alias):
                    pk_param = expr
                    break

        for lrow in self._run_source(join.left, txn, snapshot, pushed,
                                     evaluator):
            if right_ref is not None:
                right_alias = right_ref.alias or right_ref.name
                right_table = self.tm.require_table(txn, right_ref.name)
                matches = list(self._scan_table(
                    txn, right_table, right_alias, snapshot,
                    pushed.get(right_alias, []), evaluator, lrow,
                    pk_param=pk_param))
            else:
                matches = list(self._run_source(
                    join.right, txn, snapshot, pushed, evaluator))
                matches = [self._merge(lrow, r) for r in matches]

            any_match = False
            for rrow in matches:
                merged = self._merge(lrow, rrow)
                if join.on is None or \
                        sql_truth(evaluator.eval(join.on, merged)) is True:
                    any_match = True
                    yield merged
            if not any_match and join.join_type == "LEFT":
                yield self._null_pad(join.right, lrow, txn)

    def _refs_alias(self, node, alias):
        return any(t == alias for t, _ in self._column_refs(node))

    def _null_pad(self, source, row, txn):
        out = dict(row)

        def walk(node):
            if isinstance(node, ast.TableRef):
                table = self.tm.require_table(txn, node.name)
                alias = node.alias or node.name
                for col in table.schema.columns:
                    out[f"{alias}.{col.name}"] = None
                    out[col.name] = None
            elif isinstance(node, ast.Join):
                walk(node.left)
                walk(node.right)

        walk(source)
        return out

    # --------------------------------------------------------------- rows

    def _make_row(self, table, alias, data):
        """Qualified + bare column keys for one physical row."""
        row = {}
        for col_name, value in data.items():
            if col_name.startswith("_") and col_name.endswith("_"):
                continue
            row[f"{alias}.{col_name}"] = value
            row[col_name] = value
        return row

    @staticmethod
    def _merge(left, right):
        merged = dict(left)
        merged.update(right)
        return merged

    # ---------------------------------------------------------- projection

    def _expand_items(self, items, bindings):
        expanded, labels, used = [], [], set()

        def add(expr, label):
            candidate, n = label, 1
            while candidate.lower() in used:
                n += 1
                candidate = f"{label}_{n}"
            used.add(candidate.lower())
            expanded.append(ast.SelectItem(expr=expr, alias=candidate))
            labels.append(candidate)

        for item in items:
            if isinstance(item.expr, ast.Star):
                for alias, table, _ in bindings:
                    for col in table.schema.columns:
                        add(ast.Column(name=col.name, table=alias), col.name)
            elif isinstance(item.expr, ast.QualifiedStar):
                target = next(((a, t) for a, t, _ in bindings
                               if a == item.expr.table), None)
                if target is None:
                    raise ColumnNotFoundError(
                        f"unknown table {item.expr.table!r} in SELECT *")
                alias, table = target
                for col in table.schema.columns:
                    add(ast.Column(name=col.name, table=alias), col.name)
            else:
                add(item.expr, item.alias or self._default_label(item.expr))
        return expanded, labels

    def _default_label(self, expr):
        if isinstance(expr, ast.Column):
            return expr.name
        if isinstance(expr, ast.FunctionCall):
            if expr.star:
                args = "*"
            else:
                args = ", ".join(self._default_label(a) for a in expr.args)
            label = f"{expr.name.lower()}({args})"
            if expr.distinct:
                label = f"{expr.name.lower()}(distinct {args})"
            return label
        if isinstance(expr, ast.Literal):
            return str(expr.value)
        if isinstance(expr, ast.BinaryOp):
            return f"({self._default_label(expr.left)} {expr.op} " \
                   f"{self._default_label(expr.right)})"
        return type(expr).__name__.lower()

    # ---------------------------------------------------------- aggregation

    def _find_aggregates(self, items, having):
        found = {}

        def walk(node):
            if isinstance(node, (ast.Subquery, ast.Exists, ast.InSubquery)):
                return
            if isinstance(node, ast.FunctionCall) and node.name in AGGREGATES:
                found[id(node)] = node
                return
            for field_name in getattr(node, "__dataclass_fields__", {}):
                val = getattr(node, field_name)
                if hasattr(val, "__dataclass_fields__"):
                    walk(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, "__dataclass_fields__"):
                            walk(item)
                        elif isinstance(item, tuple):
                            for part in item:
                                if hasattr(part, "__dataclass_fields__"):
                                    walk(part)

        for item in items:
            walk(item.expr)
        if having is not None:
            walk(having)
        return list(found.values())

    def _aggregate(self, items, labels, aggregates, select, rows, evaluator):
        groups, order = {}, []
        if select.group_by:
            for r in rows:
                key = tuple(_hashable(evaluator.eval(g, r))
                            for g in select.group_by)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(r)
        else:
            groups[()] = rows
            order.append(())

        out_rows = []
        for key in order:
            members = groups[key]
            representative = members[0] if members else {}
            states, distinct_sets = {}, {}
            for agg in aggregates:
                states[id(agg)] = aggregate_init(agg.name)
                if agg.distinct:
                    distinct_sets[id(agg)] = set()

            for r in members:
                for agg in aggregates:
                    if agg.star:
                        states[id(agg)] += 1
                        continue
                    value = evaluator.eval(agg.args[0], r)
                    states[id(agg)] = aggregate_step(
                        agg.name, states[id(agg)], value,
                        distinct_sets.get(id(agg)))

            agg_map = {}
            for agg in aggregates:
                agg_map[id(agg)] = aggregate_final(
                    agg.name, states[id(agg)])

            group_eval = Evaluator(self, evaluator.txn, agg_map)
            if select.having is not None and sql_truth(
                    group_eval.eval(select.having, representative)) is not True:
                continue
            out = {label: group_eval.eval(item.expr, representative)
                   for item, label in zip(items, labels)}
            out_rows.append(self._merge(representative, out))
        return out_rows

    # ------------------------------------------------------------- ordering

    def _order(self, order_by, rows, labels, evaluator, bindings):
        sort_keys = []
        for ob in order_by:
            if isinstance(ob.expr, ast.Column) and ob.expr.table is None \
                    and ob.expr.name in labels:
                sort_keys.append((ob.expr.name, ob.desc))
            else:
                sort_keys.append((ob.expr, ob.desc))

        def sort_value(row):
            # SQL default: NULLs sort first in ASC and last in DESC.
            out = []
            for key, desc in sort_keys:
                v = row.get(key) if isinstance(key, str) \
                    else evaluator.eval(key, row)
                if v is None:
                    out.append((0 if not desc else 2,))
                else:
                    sk = _SortKey(v)
                    out.append((1, _DescKey(sk) if desc else sk))
            return out

        return sorted(rows, key=sort_value)


class _SortKey:
    """Comparable wrapper: numbers among themselves, strings among themselves,
    incomparable types fall back to type-name ordering."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _cmp_key(self):
        v = self.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (0, float(v))
        if isinstance(v, str):
            return (1, v)
        if isinstance(v, bool):
            return (2, v)
        return (3, type(v).__name__, str(v))

    def __lt__(self, other):
        return self._cmp_key() < other._cmp_key()

    def __eq__(self, other):
        return self._cmp_key() == other._cmp_key()


class _DescKey:
    """Inverts a :class:`_SortKey` for DESC ordering."""
    __slots__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner

    def __lt__(self, other):
        return other.inner < self.inner

    def __eq__(self, other):
        return self.inner == other.inner


def _hashable(value):
    if isinstance(value, (int, float, str, bool, tuple)) or value is None:
        return value
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return str(value)
