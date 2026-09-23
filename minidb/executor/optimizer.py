"""A tiny rule-based query optimizer.

Rules applied here at plan time:

1. **Constant folding** - literal-only expressions (``1 + 1``,
   ``'a' = 'b'``) are evaluated once and replaced by their value.
2. **Tautology elimination** - ``WHERE ... AND TRUE`` drops the TRUE;
   a constant FALSE short-circuits the whole predicate.

Predicate pushdown (the other standard rule) needs schema knowledge to decide
which base table owns an unqualified column, so it is performed by the
executor while binding table scans - see
``QueryExecutor._plan_scan_filters``.
"""

from ..sql import ast_nodes as ast
from .types import sql_truth


def split_conjunction(node):
    if isinstance(node, ast.BinaryOp) and node.op == "AND":
        return split_conjunction(node.left) + split_conjunction(node.right)
    return [node]


def make_conjunction(parts):
    if not parts:
        return None
    result = parts[0]
    for p in parts[1:]:
        result = ast.BinaryOp("AND", result, p)
    return result


def referenced_tables(node):
    """Set of table qualifiers used by an expression (unqualified -> None).

    Subqueries are opaque: columns inside them belong to a different query.
    """
    tables = set()

    def walk(n):
        if isinstance(n, ast.Column):
            tables.add(n.table)
        for field_name in getattr(n, "__dataclass_fields__", {}):
            val = getattr(n, field_name)
            if hasattr(val, "__dataclass_fields__"):
                if not isinstance(val, (ast.Subquery, ast.Exists, ast.InSubquery)):
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
    return tables


def fold_constants(node, evaluator):
    if not hasattr(node, "__dataclass_fields__"):
        return node
    if isinstance(node, (ast.Literal, ast.Column, ast.Star, ast.QualifiedStar)):
        return node
    if isinstance(node, (ast.Subquery, ast.Exists)):
        return node
    if isinstance(node, ast.InSubquery):
        return node
    if isinstance(node, ast.FunctionCall):
        node.args = [fold_constants(a, evaluator) for a in node.args]
        return node

    folded_fields = {}
    for field_name in node.__dataclass_fields__:
        val = getattr(node, field_name)
        if hasattr(val, "__dataclass_fields__"):
            folded_fields[field_name] = fold_constants(val, evaluator)
        elif isinstance(val, list):
            folded_fields[field_name] = [
                _fold_list_item(item, evaluator) for item in val]
        else:
            folded_fields[field_name] = val
    new_node = type(node)(**folded_fields)

    if isinstance(new_node, (ast.BinaryOp, ast.UnaryOp, ast.Between,
                             ast.InList)) and _is_literal_tree(new_node):
        try:
            return ast.Literal(evaluator.eval(new_node, {}))
        except Exception:
            return new_node
    return new_node


def _fold_list_item(item, evaluator):
    if hasattr(item, "__dataclass_fields__"):
        return fold_constants(item, evaluator)
    if isinstance(item, tuple):
        return tuple(fold_constants(p, evaluator)
                     if hasattr(p, "__dataclass_fields__") else p
                     for p in item)
    return item


def _is_literal_tree(node):
    if isinstance(node, ast.Literal):
        return True
    # Any other leaf expression (column, star, subquery, ...) makes the
    # whole expression non-constant, even though these nodes carry no
    # expression children of their own.
    if not hasattr(node, "__dataclass_fields__"):
        return False
    if isinstance(node, (ast.Column, ast.Star, ast.QualifiedStar,
                         ast.FunctionCall, ast.Subquery, ast.Exists,
                         ast.InSubquery)):
        return False
    for field_name in node.__dataclass_fields__:
        val = getattr(node, field_name)
        if hasattr(val, "__dataclass_fields__"):
            if not _is_literal_tree(val):
                return False
        elif isinstance(val, list):
            for item in val:
                if hasattr(item, "__dataclass_fields__") and \
                        not _is_literal_tree(item):
                    return False
    return True


class Optimizer:
    def optimize(self, select, evaluator):
        if select.where is not None:
            select.where = fold_constants(select.where, evaluator)
            select.where = self._eliminate_constants(select.where)
        if select.having is not None:
            select.having = fold_constants(select.having, evaluator)
        self._fold_joins(select.source, evaluator)
        return select

    def _fold_joins(self, source, evaluator):
        if isinstance(source, ast.Join):
            if source.on is not None:
                source.on = fold_constants(source.on, evaluator)
            self._fold_joins(source.left, evaluator)
            self._fold_joins(source.right, evaluator)

    def _eliminate_constants(self, where):
        parts = []
        for p in split_conjunction(where):
            if isinstance(p, ast.Literal):
                if sql_truth(p.value) is False:
                    return ast.Literal(False)
                continue  # TRUE drops out
            parts.append(p)
        return make_conjunction(parts)
