"""Recursive-descent parser producing the AST in :mod:`minidb.sql.ast`.

Grammar (simplified)::

    stmt      := select | insert | update | delete | create | txnctl | show
    select    := SELECT [DISTINCT] proj
                 FROM table [alias] (join)*
                 [WHERE expr] [GROUP BY expr (, expr)* [HAVING expr]]
                 [ORDER BY order_item (, order_item)*]
                 [LIMIT n [OFFSET n]]
    join      := [INNER] JOIN table [alias] ON expr
               | LEFT [OUTER] JOIN table [alias] ON expr
    expr precedence (low->high):
      OR -> AND -> NOT -> comparison(= <> < <= > >= LIKE IS IN BETWEEN)
      -> additive -> multiplicative -> unary -> primary
    primary   := literal | column | func(args) | (expr) | (select)
"""

from __future__ import annotations

from typing import Any, Optional

from . import ast
from .lexer import Lexer, Token


class ParseError(Exception):
    pass


# Clause keywords that must never be accepted where a column is expected.
_RESERVED_AS_NAME = frozenset({
    "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET",
    "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "ON", "AS", "AND", "OR",
    "IS", "IN", "LIKE", "BETWEEN", "BY", "ASC", "DESC", "SELECT",
    "VALUES", "SET", "INTO", "TABLE",
})


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens = Lexer(text).tokenize()
        self.pos = 0

    # ------------------------------------------------------------------ #
    # token helpers
    # ------------------------------------------------------------------ #
    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def check(self, kind: str, value: Any = None) -> bool:
        tok = self.cur
        if tok.kind != kind:
            return False
        return value is None or tok.value == value

    def match(self, kind: str, value: Any = None) -> Optional[Token]:
        if self.check(kind, value):
            return self.advance()
        return None

    def expect(self, kind: str, value: Any = None) -> Token:
        if not self.check(kind, value):
            want = value if value is not None else kind
            raise ParseError(
                f"expected {want!r} but got {self.cur.value!r} "
                f"(at position {self.cur.pos})"
            )
        return self.advance()

    def match_word(self, *words: str) -> bool:
        if self.check("WORD") and self.cur.value in words:
            self.advance()
            return True
        return False

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def parse(self) -> Any:
        if self.check("EOF"):
            raise ParseError("empty statement")
        word = self.cur.value if self.cur.kind == "WORD" else None
        if word == "SELECT":
            node = self.parse_select()
        elif word == "INSERT":
            node = self.parse_insert()
        elif word == "UPDATE":
            node = self.parse_update()
        elif word in ("DELETE",):
            node = self.parse_delete()
        elif word == "CREATE":
            node = self.parse_create()
        elif word in ("BEGIN", "START"):
            node = self.parse_begin()
        elif word == "COMMIT":
            self.advance()
            self.match_word("WORK")
            node = ast.CommitStmt()
        elif word == "ROLLBACK":
            self.advance()
            self.match_word("WORK")
            node = ast.RollbackStmt()
        elif word == "SHOW":
            node = self.parse_show()
        elif word == "DESCRIBE" or (self.check("WORD", "DESC")
                                    and self._peek_is_ident()):
            self.advance()
            name = self._parse_ident()
            node = ast.Describe(table=name)
        else:
            raise ParseError(f"unsupported statement starting with {word!r}")
        self.expect("EOF")
        return node

    def _peek_is_ident(self) -> bool:
        return self.tokens[self.pos + 1].kind == "IDENT"

    # ------------------------------------------------------------------ #
    # SELECT
    # ------------------------------------------------------------------ #
    def parse_select(self) -> ast.SelectStmt:
        self.expect("WORD", "SELECT")
        distinct = self.match_word("DISTINCT")
        projections = self._parse_projections()

        from_table = from_alias = None
        joins: list[ast.JoinSpec] = []
        if self.match_word("FROM"):
            from_table = self._parse_ident()
            from_alias = self._parse_optional_alias(from_table)
            joins = self._parse_joins()

        where = None
        if self.match_word("WHERE"):
            where = self.parse_expr()

        group_by: list[Any] = []
        having = None
        if self.match_word("GROUP"):
            self.expect("WORD", "BY")
            group_by.append(self.parse_expr())
            while self.match("PUNCT", ","):
                group_by.append(self.parse_expr())
            if self.match_word("HAVING"):
                having = self.parse_expr()

        order_by: list[tuple[Any, str]] = []
        if self.match_word("ORDER"):
            self.expect("WORD", "BY")
            order_by.append(self._parse_order_item())
            while self.match("PUNCT", ","):
                order_by.append(self._parse_order_item())

        limit = offset = None
        if self.match_word("LIMIT"):
            limit = self._parse_nonneg_int()
            if self.match_word("OFFSET"):
                offset = self._parse_nonneg_int()
        elif self.match_word("OFFSET"):
            offset = self._parse_nonneg_int()
            if self.match_word("LIMIT"):
                limit = self._parse_nonneg_int()

        return ast.SelectStmt(
            projections=projections,
            from_table=from_table,
            from_alias=from_alias,
            joins=joins,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=bool(distinct),
        )

    def _parse_projections(self) -> list[Any]:
        items = [self._parse_projection()]
        while self.match("PUNCT", ","):
            items.append(self._parse_projection())
        return items

    def _parse_projection(self) -> Any:
        if self.check("PUNCT", "*"):
            self.advance()
            return ast.Star()
        if self._is_qualified_star():
            table = self._parse_ident()
            self.expect("PUNCT", ".")
            self.expect("PUNCT", "*")
            return ast.Star(table=table)
        expr = self.parse_expr()
        return expr

    def _is_qualified_star(self) -> bool:
        return (
            self.cur.kind == "IDENT"
            and self.tokens[self.pos + 1].kind == "PUNCT"
            and self.tokens[self.pos + 1].value == "."
            and self.tokens[self.pos + 2].kind == "PUNCT"
            and self.tokens[self.pos + 2].value == "*"
        )

    def _parse_order_item(self) -> tuple[Any, str]:
        expr = self.parse_expr()
        direction = "ASC"
        if self.match_word("ASC"):
            direction = "ASC"
        elif self.match_word("DESC"):
            direction = "DESC"
        return expr, direction

    def _parse_optional_alias(self, table: str) -> Optional[str]:
        # AS alias  | bare identifier alias (but not a keyword we need)
        if self.match_word("AS"):
            return self._parse_ident()
        if self.check("IDENT"):
            return self._parse_ident()
        return None

    def _parse_joins(self) -> list[ast.JoinSpec]:
        joins: list[ast.JoinSpec] = []
        while True:
            kind: Optional[str]
            if self.match_word("INNER"):
                self.expect("WORD", "JOIN")
                kind = "INNER"
            elif self.match_word("LEFT"):
                self.match_word("OUTER")
                self.expect("WORD", "JOIN")
                kind = "LEFT"
            elif self.match_word("JOIN"):
                kind = "INNER"
            else:
                break
            table = self._parse_ident()
            alias = self._parse_optional_alias(table)
            self.expect("WORD", "ON")
            on = self.parse_expr()
            joins.append(ast.JoinSpec(table=table, alias=alias, kind=kind, on=on))
        return joins

    # ------------------------------------------------------------------ #
    # INSERT
    # ------------------------------------------------------------------ #
    def parse_insert(self) -> ast.Insert:
        self.expect("WORD", "INSERT")
        self.expect("WORD", "INTO")
        table = self._parse_ident()
        columns: Optional[list[str]] = None
        if self.match("PUNCT", "("):
            columns = [self._parse_ident()]
            while self.match("PUNCT", ","):
                columns.append(self._parse_ident())
            self.expect("PUNCT", ")")
        self.expect("WORD", "VALUES")
        rows = [self._parse_value_row()]
        while self.match("PUNCT", ","):
            rows.append(self._parse_value_row())
        return ast.Insert(table=table, columns=columns, rows=rows)

    def _parse_value_row(self) -> list[Any]:
        self.expect("PUNCT", "(")
        values = [self.parse_expr()]
        while self.match("PUNCT", ","):
            values.append(self.parse_expr())
        self.expect("PUNCT", ")")
        return values

    # ------------------------------------------------------------------ #
    # UPDATE / DELETE
    # ------------------------------------------------------------------ #
    def parse_update(self) -> ast.Update:
        self.expect("WORD", "UPDATE")
        table = self._parse_ident()
        alias = self._parse_optional_alias(table)
        if alias is not None:
            raise ParseError("UPDATE does not support table aliases")
        self.expect("WORD", "SET")
        assignments = [self._parse_assignment()]
        while self.match("PUNCT", ","):
            assignments.append(self._parse_assignment())
        where = self.parse_expr() if self.match_word("WHERE") else None
        return ast.Update(table=table, assignments=assignments, where=where)

    def _parse_assignment(self) -> tuple[str, Any]:
        col = self._parse_ident()
        self.expect("PUNCT", "=")
        value = self.parse_expr()
        return col, value

    def parse_delete(self) -> ast.Delete:
        self.expect("WORD", "DELETE")
        self.expect("WORD", "FROM")
        table = self._parse_ident()
        where = self.parse_expr() if self.match_word("WHERE") else None
        return ast.Delete(table=table, where=where)

    # ------------------------------------------------------------------ #
    # CREATE TABLE
    # ------------------------------------------------------------------ #
    def parse_create(self) -> ast.CreateTable:
        self.expect("WORD", "CREATE")
        self.expect("WORD", "TABLE")
        name = self._parse_ident()
        self.expect("PUNCT", "(")
        columns = [self._parse_column_def()]
        while self.match("PUNCT", ","):
            # table-level PRIMARY KEY (a, b)
            if self.check("WORD", "PRIMARY"):
                self._consume_table_pk(columns)
            else:
                columns.append(self._parse_column_def())
        self.expect("PUNCT", ")")
        return ast.CreateTable(name=name, columns=columns)

    def _consume_table_pk(self, columns: list[ast.ColumnDef]) -> None:
        self.expect("WORD", "PRIMARY")
        self.expect("WORD", "KEY")
        self.expect("PUNCT", "(")
        pk_cols = [self._parse_ident()]
        while self.match("PUNCT", ","):
            pk_cols.append(self._parse_ident())
        self.expect("PUNCT", ")")
        by_name = {c.name.lower(): c for c in columns}
        for name in pk_cols:
            col = by_name.get(name.lower())
            if col is None:
                raise ParseError(f"PRIMARY KEY references unknown column {name!r}")
            col.primary_key = True
            col.nullable = False

    def _parse_column_def(self) -> ast.ColumnDef:
        name = self._parse_ident_raw()
        type_name, length = self._parse_type()
        nullable = True
        primary_key = False
        while True:
            if self.match_word("NOT"):
                self.expect("WORD", "NULL")
                nullable = False
            elif self.match_word("NULL"):
                nullable = True
            elif self.match_word("PRIMARY"):
                self.expect("WORD", "KEY")
                primary_key = True
                nullable = False
            else:
                break
        return ast.ColumnDef(
            name=name, type=type_name, length=length,
            nullable=nullable, primary_key=primary_key,
        )

    def _parse_type(self) -> tuple[str, Optional[int]]:
        tok = self.expect("WORD")
        mapping = {
            "INT": "INT", "INTEGER": "INT", "BIGINT": "INT",
            "SMALLINT": "INT", "TINYINT": "INT",
            "FLOAT": "FLOAT", "DOUBLE": "FLOAT", "REAL": "FLOAT",
            "VARCHAR": "VARCHAR", "CHAR": "VARCHAR",
            "TEXT": "TEXT", "STRING": "TEXT",
            "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN",
        }
        if tok.value not in mapping:
            raise ParseError(f"unknown type {tok.value!r}")
        type_name = mapping[tok.value]
        length = None
        if self.match("PUNCT", "("):
            length = self._parse_nonneg_int()
            self.expect("PUNCT", ")")
        return type_name, length

    # ------------------------------------------------------------------ #
    # transaction control / SHOW
    # ------------------------------------------------------------------ #
    def parse_begin(self) -> ast.BeginStmt:
        if self.match_word("START"):
            self.expect("WORD", "TRANSACTION")
        else:
            self.expect("WORD", "BEGIN")
        isolation: Optional[str] = None
        if self.match_word("ISOLATION"):
            self.expect("WORD", "LEVEL")
            isolation = self._parse_isolation_level()
        elif self.match_word("READ"):
            if self.match_word("COMMITTED"):
                isolation = "READ COMMITTED"
            else:
                self.expect("WORD", "UNCOMMITTED")
                raise ParseError("READ UNCOMMITTED is not supported")
        elif self.match_word("SERIALIZABLE"):
            isolation = "SERIALIZABLE"
        return ast.BeginStmt(isolation=isolation)

    def _parse_isolation_level(self) -> str:
        if self.match_word("READ"):
            if self.match_word("COMMITTED"):
                return "READ COMMITTED"
            self.expect("WORD", "UNCOMMITTED")
            raise ParseError("READ UNCOMMITTED is not supported")
        if self.match_word("REPEATABLE"):
            self.expect("WORD", "READ")
            raise ParseError("REPEATABLE READ is not supported")
        self.expect("WORD", "SERIALIZABLE")
        return "SERIALIZABLE"

    def parse_show(self) -> Any:
        self.expect("WORD", "SHOW")
        if self.match_word("TABLES"):
            return ast.ShowTables()
        raise ParseError("only SHOW TABLES is supported")

    # ------------------------------------------------------------------ #
    # expressions
    # ------------------------------------------------------------------ #
    def parse_expr(self) -> Any:
        return self._parse_or()

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self.match_word("OR"):
            right = self._parse_and()
            left = ast.BinaryOp("OR", left, right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_not()
        while self.match_word("AND"):
            right = self._parse_not()
            left = ast.BinaryOp("AND", left, right)
        return left

    def _parse_not(self) -> Any:
        if self.match_word("NOT"):
            return ast.UnaryOp("NOT", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> Any:
        left = self._parse_additive()
        while True:
            negated = False
            if self.match_word("NOT"):
                # could be NOT LIKE / NOT IN / NOT BETWEEN
                negated = True
            if self.check("PUNCT") and self.cur.value in ("=", "!=", "<", "<=", ">", ">="):
                op = self.advance().value
                right = self._parse_additive()
                left = ast.BinaryOp(op, left, right)
                continue
            if self.match_word("LIKE"):
                right = self._parse_additive()
                left = ast.BinaryOp("NOT LIKE" if negated else "LIKE", left, right)
                continue
            if self.match_word("IN"):
                left = self._parse_in_tail(left, negated)
                continue
            if self.match_word("BETWEEN"):
                low = self._parse_additive()
                self.expect("WORD", "AND")
                high = self._parse_additive()
                if negated:
                    left = ast.UnaryOp(
                        "NOT",
                        ast.BinaryOp("BETWEEN", left, ast.BinaryOp("AND", low, high)),
                    )
                else:
                    left = ast.BinaryOp("BETWEEN", left, ast.BinaryOp("AND", low, high))
                continue
            if not negated and self.match_word("IS"):
                if self.match_word("NOT"):
                    self.expect("WORD", "NULL")
                    left = ast.UnaryOp("IS NOT NULL", left)
                else:
                    self.expect("WORD", "NULL")
                    left = ast.UnaryOp("IS NULL", left)
                continue
            if negated:
                raise ParseError("expected LIKE/IN/BETWEEN after NOT")
            break
        return left

    def _parse_in_tail(self, left: Any, negated: bool) -> Any:
        if self.match("PUNCT", "("):
            if self.check("WORD", "SELECT"):
                query = self.parse_select()
                self.expect("PUNCT", ")")
                return ast.InSubquery(expr=left, query=query, negated=negated)
            items = [self._parse_additive()]
            while self.match("PUNCT", ","):
                items.append(self._parse_additive())
            self.expect("PUNCT", ")")
            # fold an explicit list into a chain of OR = comparisons
            cond = None
            for item in items:
                cmp_ = ast.BinaryOp("=", left, item)
                cond = cmp_ if cond is None else ast.BinaryOp("OR", cond, cmp_)
            if negated:
                cond = ast.UnaryOp("NOT", cond)
            return cond
        raise ParseError("expected ( after IN")

    def _parse_additive(self) -> Any:
        left = self._parse_multiplicative()
        while self.check("PUNCT") and self.cur.value in ("+", "-"):
            op = self.advance().value
            right = self._parse_multiplicative()
            left = ast.BinaryOp(op, left, right)
        return left

    def _parse_multiplicative(self) -> Any:
        left = self._parse_unary()
        while self.check("PUNCT") and self.cur.value in ("*", "/"):
            op = self.advance().value
            right = self._parse_unary()
            left = ast.BinaryOp(op, left, right)
        return left

    def _parse_unary(self) -> Any:
        if self.check("PUNCT", "-"):
            self.advance()
            return ast.UnaryOp("-", self._parse_unary())
        if self.check("PUNCT", "+"):
            self.advance()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        tok = self.cur
        if tok.kind == "NUMBER":
            self.advance()
            return ast.Literal(tok.literal)
        if tok.kind == "STRING":
            self.advance()
            return ast.Literal(tok.literal)
        if self.match_word("NULL"):
            return ast.Literal(None)
        if self.match_word("TRUE"):
            return ast.Literal(True)
        if self.match_word("FALSE"):
            return ast.Literal(False)
        if self.match("PUNCT", "("):
            if self.check("WORD", "SELECT"):
                query = self.parse_select()
                self.expect("PUNCT", ")")
                return ast.Subquery(query=query)
            expr = self.parse_expr()
            self.expect("PUNCT", ")")
            return expr
        if tok.kind == "WORD" and tok.value in (
            "COUNT", "SUM", "AVG", "MIN", "MAX", "UPPER", "LOWER"
        ):
            return self._parse_func()
        if tok.kind in ("IDENT", "WORD"):
            # bare-word keywords used as columns (e.g. a column named "level")
            return self._parse_column_ref()
        raise ParseError(f"unexpected token {tok.value!r} at position {tok.pos}")

    def _parse_func(self) -> ast.FuncCall:
        name = self.advance().value
        self.expect("PUNCT", "(")
        distinct = bool(self.match_word("DISTINCT"))
        star = False
        args: list[Any] = []
        if self.match("PUNCT", "*"):
            star = True
        elif not self.check("PUNCT", ")"):
            args.append(self.parse_expr())
            while self.match("PUNCT", ","):
                args.append(self.parse_expr())
        self.expect("PUNCT", ")")
        return ast.FuncCall(name=name, args=args, distinct=distinct, star=star)

    def _parse_column_ref(self) -> ast.ColumnRef:
        first_tok = self.cur
        first = self.advance().value
        # reserved clause keywords may never be used as a column reference
        if first_tok.kind == "WORD" and first in _RESERVED_AS_NAME:
            raise ParseError(
                f"unexpected keyword {first!r} where a column was expected "
                f"(at position {first_tok.pos})"
            )
        if self.match("PUNCT", "."):
            second = self.advance()
            if second.kind not in ("IDENT", "WORD"):
                raise ParseError("expected column name after '.'")
            return ast.ColumnRef(name=second.value, table=first)
        return ast.ColumnRef(name=first)

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _parse_ident(self) -> str:
        tok = self.advance()
        if tok.kind not in ("IDENT", "WORD"):
            raise ParseError(
                f"expected identifier but got {tok.value!r} at position {tok.pos}"
            )
        return tok.value

    def _parse_ident_raw(self) -> str:
        """Column name in DDL must be an IDENT (or quoted), not a keyword."""
        tok = self.advance()
        if tok.kind != "IDENT":
            raise ParseError(
                f"expected column name but got {tok.value!r} at position {tok.pos}"
            )
        return tok.value

    def _parse_nonneg_int(self) -> int:
        tok = self.expect("NUMBER")
        if not isinstance(tok.literal, int) or tok.literal < 0:
            raise ParseError(f"expected non-negative integer, got {tok.value}")
        return tok.literal


def parse_sql(text: str) -> Any:
    return Parser(text).parse()
