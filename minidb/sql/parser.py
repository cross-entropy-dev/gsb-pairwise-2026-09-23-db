"""Recursive-descent SQL parser.

Produces the AST nodes in :mod:`minidb.sql.ast_nodes`.  Grammar precedence
(lowest to highest):

    OR  <  AND  <  NOT  <  comparison (=, <>, IS, LIKE, IN, BETWEEN ...)
        <  + -  <  * /  <  unary -  <  primary
"""

from . import ast_nodes as ast
from .lexer import tokenize
from ..errors import ParseError

TYPE_ALIASES = {
    "INT": "INT", "INTEGER": "INT", "BIGINT": "INT", "SMALLINT": "INT",
    "TINYINT": "INT",
    "VARCHAR": "VARCHAR", "CHAR": "VARCHAR",
    "TEXT": "TEXT",
    "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN",
    "FLOAT": "FLOAT", "DOUBLE": "FLOAT", "REAL": "FLOAT",
}

# Keywords that terminate an expression / can never be column names.
EXPRESSION_STOPWORDS = frozenset({
    "FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "AS",
    "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON", "UNION",
    "THEN", "WHEN", "ELSE", "END", "BY", "SET", "VALUES", "INTO", "AND",
    "OR", "IS", "LIKE", "BETWEEN", "IN", "NOT", "ASC", "DESC", "DISTINCT",
    "ALL", "COMMIT", "TRANSACTION", "ISOLATION", "LEVEL", "TABLE",
})


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    # ------------------------------------------------------------ primitives

    @property
    def cur(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def advance(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_end(self):
        return self.i >= len(self.tokens)

    def is_kw(self, *words):
        tok = self.cur
        return tok is not None and tok.type == "KEYWORD" and tok.value in words

    def is_op(self, *ops):
        tok = self.cur
        return tok is not None and tok.type == "OP" and tok.value in ops

    def eat_kw(self, word):
        if self.is_kw(word):
            self.advance()
            return True
        return False

    def eat_op(self, op):
        if self.is_op(op):
            self.advance()
            return True
        return False

    def expect_op(self, op):
        if not self.eat_op(op):
            got = self.cur.value if self.cur else "EOF"
            raise ParseError(f"expected {op!r} but got {got!r}")

    def expect_kw(self, word):
        if not self.eat_kw(word):
            got = self.cur.value if self.cur else "EOF"
            raise ParseError(f"expected {word} but got {got!r}")

    def read_name(self):
        """Read an identifier; keywords are accepted where names are expected."""
        tok = self.cur
        if tok is None:
            raise ParseError("expected an identifier but reached EOF")
        if tok.type in ("IDENT", "KEYWORD"):
            self.advance()
            return tok.value if tok.type == "IDENT" else tok.value.lower()
        raise ParseError(f"expected an identifier but got {tok.value!r}")

    # -------------------------------------------------------------- entry pt

    def parse_statements(self):
        stmts = []
        while not self.at_end():
            self.eat_op(";")
            if self.at_end():
                break
            stmts.append(self.parse_statement())
            self.eat_op(";")
        return stmts

    def parse_statement(self):
        if self.is_kw("SELECT"):
            return self.parse_select()
        if self.is_kw("INSERT"):
            return self.parse_insert()
        if self.is_kw("UPDATE"):
            return self.parse_update()
        if self.is_kw("DELETE"):
            return self.parse_delete()
        if self.is_kw("CREATE"):
            return self.parse_create()
        if self.is_kw("DROP"):
            return self.parse_drop()
        if self.is_kw("BEGIN", "START"):
            return self.parse_begin()
        if self.is_kw("COMMIT"):
            self.advance()
            return ast.Commit()
        if self.is_kw("ROLLBACK"):
            self.advance()
            return ast.Rollback()
        raise ParseError(f"unsupported statement starting with {self.cur.value!r}")

    # ------------------------------------------------------------- DDL

    def parse_create(self):
        self.expect_kw("CREATE")
        self.expect_kw("TABLE")
        if_not_exists = False
        if self.eat_kw("IF"):
            self.expect_kw("NOT")
            self.expect_kw("EXISTS")
            if_not_exists = True
        name = self.read_name()
        self.expect_op("(")
        columns = []
        while True:
            if self.is_kw("PRIMARY"):  # table-level PRIMARY KEY (a, b)
                self.advance()
                self.expect_kw("KEY")
                self.expect_op("(")
                pk_cols = []
                while True:
                    pk_cols.append(self.read_name())
                    if not self.eat_op(","):
                        break
                self.expect_op(")")
                for col in columns:
                    if col.name in pk_cols:
                        col.primary_key = True
                        col.not_null = True
            else:
                columns.append(self.parse_column_def())
            if not self.eat_op(","):
                break
        self.expect_op(")")
        return ast.CreateTable(name=name, columns=columns, if_not_exists=if_not_exists)

    def parse_column_def(self):
        name = self.read_name()
        type_tok = self.cur
        if type_tok is None or type_tok.type != "KEYWORD" or type_tok.value not in TYPE_ALIASES:
            raise ParseError(f"column {name!r} needs a supported data type")
        self.advance()
        type_name = TYPE_ALIASES[type_tok.value]
        length = None
        if self.eat_op("("):
            length_tok = self.cur
            if length_tok is None or length_tok.type != "NUMBER" or not isinstance(length_tok.value, int):
                raise ParseError("expected an integer length in type declaration")
            length = length_tok.value
            self.advance()
            self.expect_op(")")

        col = ast.ColumnDef(name=name, type_name=type_name, length=length)
        # Constraint clauses, in any order.
        while True:
            if self.eat_kw("PRIMARY"):
                self.expect_kw("KEY")
                col.primary_key = True
                col.not_null = True
                if self.eat_kw("AUTOINCREMENT"):
                    col.autoincrement = True
            elif self.eat_kw("NOT"):
                self.expect_kw("NULL")
                col.not_null = True
            elif self.eat_kw("NULL"):
                col.not_null = False
            elif self.eat_kw("UNIQUE"):
                col.unique = True
            elif self.eat_kw("AUTOINCREMENT"):
                col.autoincrement = True
            elif self.eat_kw("DEFAULT"):
                col.default = self.parse_expression()
            else:
                break
        return col

    def parse_drop(self):
        self.expect_kw("DROP")
        self.expect_kw("TABLE")
        if_exists = False
        if self.eat_kw("IF"):
            self.expect_kw("EXISTS")
            if_exists = True
        return ast.DropTable(name=self.read_name(), if_exists=if_exists)

    # ------------------------------------------------------------- DML

    def parse_insert(self):
        self.expect_kw("INSERT")
        self.expect_kw("INTO")
        table = self.read_name()
        columns = None
        if self.eat_op("("):
            columns = []
            while True:
                columns.append(self.read_name())
                if not self.eat_op(","):
                    break
            self.expect_op(")")
        self.expect_kw("VALUES")
        rows = []
        while True:
            self.expect_op("(")
            values = []
            if not self.is_op(")"):
                while True:
                    values.append(self.parse_expression())
                    if not self.eat_op(","):
                        break
            self.expect_op(")")
            rows.append(values)
            if not self.eat_op(","):
                break
        return ast.Insert(table=table, columns=columns, rows=rows)

    def parse_update(self):
        self.expect_kw("UPDATE")
        table = self.read_name()
        self.expect_kw("SET")
        assignments = []
        while True:
            col = self.read_name()
            self.expect_op("=")
            expr = self.parse_expression()
            assignments.append((col, expr))
            if not self.eat_op(","):
                break
        where = self.parse_where()
        return ast.Update(table=table, assignments=assignments, where=where)

    def parse_delete(self):
        self.expect_kw("DELETE")
        self.expect_kw("FROM")
        table = self.read_name()
        return ast.Delete(table=table, where=self.parse_where())

    def parse_where(self):
        if self.eat_kw("WHERE"):
            return self.parse_expression()
        return None

    # ----------------------------------------------------------- SELECT

    def parse_select(self):
        self.expect_kw("SELECT")
        distinct = self.eat_kw("DISTINCT")
        self.eat_kw("ALL")
        items = self.parse_select_items()
        source = None
        if self.eat_kw("FROM"):
            source = self.parse_table_source()
        where = self.parse_where()
        group_by = []
        having = None
        if self.eat_kw("GROUP"):
            self.expect_kw("BY")
            while True:
                group_by.append(self.parse_expression())
                if not self.eat_op(","):
                    break
            if self.eat_kw("HAVING"):
                having = self.parse_expression()
        order_by = []
        if self.eat_kw("ORDER"):
            self.expect_kw("BY")
            while True:
                expr = self.parse_expression()
                desc = False
                if self.eat_kw("ASC"):
                    desc = False
                elif self.eat_kw("DESC"):
                    desc = True
                order_by.append(ast.OrderByItem(expr=expr, desc=desc))
                if not self.eat_op(","):
                    break
        limit = None
        offset = None
        if self.eat_kw("LIMIT"):
            limit = self.parse_expression()
            if self.eat_kw("OFFSET"):
                offset = self.parse_expression()
        elif self.eat_kw("OFFSET"):
            offset = self.parse_expression()
            self.eat_kw("ROWS")
        return ast.Select(items=items, source=source, where=where,
                          group_by=group_by, having=having, order_by=order_by,
                          limit=limit, offset=offset, distinct=distinct)

    def parse_select_items(self):
        items = []
        while True:
            if self.is_op("*"):
                self.advance()
                items.append(ast.SelectItem(expr=ast.Star()))
            elif self.cur and self.cur.type in ("IDENT", "KEYWORD") \
                    and self._ahead_is(1, "OP", ".") and self._ahead_is(2, "OP", "*"):
                table = self.read_name()
                self.advance()  # .
                self.advance()  # *
                items.append(ast.SelectItem(expr=ast.QualifiedStar(table=table)))
            else:
                expr = self.parse_expression()
                alias = None
                if self.eat_kw("AS"):
                    alias = self.read_name()
                elif self.cur and self.cur.type == "IDENT":
                    alias = self.advance().value
                items.append(ast.SelectItem(expr=expr, alias=alias))
            if not self.eat_op(","):
                break
        return items

    def _ahead_is(self, offset, type_, value=None):
        j = self.i + offset
        if j >= len(self.tokens):
            return False
        tok = self.tokens[j]
        return tok.type == type_ and (value is None or tok.value == value)

    def parse_table_source(self):
        left = self.parse_table_ref()
        while True:
            if self.eat_kw("INNER"):
                self.expect_kw("JOIN")
                right = self.parse_table_ref()
                on = self.parse_on()
                left = ast.Join("INNER", left, right, on)
            elif self.eat_kw("LEFT"):
                self.eat_kw("OUTER")
                self.expect_kw("JOIN")
                right = self.parse_table_ref()
                on = self.parse_on()
                left = ast.Join("LEFT", left, right, on)
            elif self.eat_kw("RIGHT"):
                self.eat_kw("OUTER")
                self.expect_kw("JOIN")
                right = self.parse_table_ref()
                on = self.parse_on()
                left = ast.Join("RIGHT", left, right, on)
            elif self.eat_kw("JOIN"):
                right = self.parse_table_ref()
                on = self.parse_on()
                left = ast.Join("INNER", left, right, on)
            elif self.eat_op(","):
                right = self.parse_table_ref()
                left = ast.Join("CROSS", left, right, None)
            else:
                break
        return left

    def parse_table_ref(self):
        name = self.read_name()
        alias = None
        if self.eat_kw("AS"):
            alias = self.read_name()
        elif self.cur and self.cur.type == "IDENT":
            alias = self.advance().value
        return ast.TableRef(name=name, alias=alias)

    def parse_on(self):
        if self.eat_kw("ON"):
            return self.parse_expression()
        return None

    # ------------------------------------------------------- transactions

    def parse_begin(self):
        if self.eat_kw("START"):
            self.expect_kw("TRANSACTION")
        else:
            self.expect_kw("BEGIN")
            self.eat_kw("TRANSACTION")
        isolation = None
        if self.eat_kw("ISOLATION"):
            self.expect_kw("LEVEL")
            if self.eat_kw("READ"):
                if self.eat_kw("COMMITTED"):
                    isolation = "READ COMMITTED"
                else:
                    self.expect_kw("UNCOMMITTED")
                    isolation = "READ UNCOMMITTED"
            elif self.eat_kw("REPEATABLE"):
                self.expect_kw("READ")
                isolation = "REPEATABLE READ"
            elif self.eat_kw("SERIALIZABLE"):
                isolation = "SERIALIZABLE"
        return ast.Begin(isolation=isolation)

    # ------------------------------------------------------- expressions

    def parse_expression(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self.eat_kw("OR"):
            right = self._parse_and()
            left = ast.BinaryOp("OR", left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self.eat_kw("AND"):
            right = self._parse_not()
            left = ast.BinaryOp("AND", left, right)
        return left

    def _parse_not(self):
        if self.eat_kw("NOT"):
            return ast.UnaryOp("NOT", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_additive()
        while True:
            negated = False
            if self.eat_kw("NOT"):
                negated = True
            if self.is_op("=", "<>", "!=", "<", ">", "<=", ">="):
                op = self.advance().value
                if op == "!=":
                    op = "<>"
                left = ast.BinaryOp(op, left, self._parse_additive())
            elif self.eat_kw("IS"):
                is_not = self.eat_kw("NOT")
                self.expect_kw("NULL")
                left = ast.UnaryOp("IS_NOT_NULL" if is_not else "IS_NULL", left)
            elif self.eat_kw("LIKE"):
                left = ast.BinaryOp("NOT LIKE" if negated else "LIKE",
                                    left, self._parse_additive())
                negated = False
            elif self.eat_kw("BETWEEN"):
                low = self._parse_additive()
                self.expect_kw("AND")
                high = self._parse_additive()
                left = ast.Between(expr=left, low=low, high=high, negated=negated)
                negated = False
            elif self.eat_kw("IN"):
                left = self._parse_in_tail(left, negated)
                negated = False
            else:
                if negated:
                    raise ParseError("expected LIKE / IN / BETWEEN after NOT")
                break
        return left

    def _parse_in_tail(self, expr, negated):
        self.expect_op("(")
        if self.is_kw("SELECT"):
            sub = self.parse_select()
            self.expect_op(")")
            return ast.InSubquery(expr=expr, subquery=sub, negated=negated)
        values = []
        while True:
            values.append(self.parse_expression())
            if not self.eat_op(","):
                break
        self.expect_op(")")
        return ast.InList(expr=expr, values=values, negated=negated)

    def _parse_additive(self):
        left = self._parse_multiplicative()
        while self.is_op("+", "-"):
            op = self.advance().value
            left = ast.BinaryOp(op, left, self._parse_multiplicative())
        return left

    def _parse_multiplicative(self):
        left = self._parse_unary()
        while self.is_op("*", "/"):
            op = self.advance().value
            left = ast.BinaryOp(op, left, self._parse_unary())
        return left

    def _parse_unary(self):
        if self.eat_op("-"):
            return ast.UnaryOp("-", self._parse_unary())
        if self.eat_op("+"):
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self):
        tok = self.cur
        if tok is None:
            raise ParseError("unexpected end of statement in expression")

        if tok.type == "NUMBER":
            self.advance()
            return ast.Literal(tok.value)
        if tok.type == "STRING":
            self.advance()
            return ast.Literal(tok.value)
        if self.eat_kw("NULL"):
            return ast.Literal(None)
        if self.eat_kw("TRUE"):
            return ast.Literal(True)
        if self.eat_kw("FALSE"):
            return ast.Literal(False)

        if self.is_op("("):
            self.advance()
            if self.is_kw("SELECT"):
                sub = self.parse_select()
                self.expect_op(")")
                return ast.Subquery(sub)
            expr = self.parse_expression()
            self.expect_op(")")
            return expr

        if self.eat_kw("CASE"):
            return self._parse_case()

        if self.eat_kw("EXISTS"):
            self.expect_op("(")
            sub = self.parse_select()
            self.expect_op(")")
            return ast.Exists(subquery=sub, negated=False)

        # Identifier / qualified name / function call.
        if tok.type in ("IDENT", "KEYWORD"):
            # Clause-introducing keywords can never be column names; other
            # keywords (COUNT, ...) are accepted as identifiers here.
            if tok.type == "KEYWORD" and tok.value in EXPRESSION_STOPWORDS:
                raise ParseError(
                    f"unexpected keyword {tok.value!r} in expression")
            name = self.read_name()
            if self.eat_op("."):
                if self.eat_op("*"):
                    return ast.QualifiedStar(table=name)
                col = self.read_name()
                return ast.Column(name=col, table=name)
            if self.is_op("("):
                return self._parse_function_call(name)
            return ast.Column(name=name)

        raise ParseError(f"unexpected token {tok.value!r} in expression")

    def _parse_function_call(self, name):
        self.expect_op("(")
        upper = name.upper()
        distinct = False
        star = False
        args = []
        if self.is_op("*"):
            self.advance()
            star = True
        elif not self.is_op(")"):
            if self.eat_kw("DISTINCT"):
                distinct = True
            while True:
                args.append(self.parse_expression())
                if not self.eat_op(","):
                    break
        self.expect_op(")")
        return ast.FunctionCall(name=upper, args=args, distinct=distinct, star=star)

    def _parse_case(self):
        operand = None
        if not self.is_kw("WHEN"):
            operand = self.parse_expression()
        whens = []
        while self.eat_kw("WHEN"):
            cond = self.parse_expression()
            self.expect_kw("THEN")
            result = self.parse_expression()
            whens.append((cond, result))
        default = None
        if self.eat_kw("ELSE"):
            default = self.parse_expression()
        self.expect_kw("END")
        return ast.CaseExpr(operand=operand, whens=whens, default=default)


def parse(sql):
    """Parse one or many ``;``-separated statements into AST nodes."""
    return Parser(tokenize(sql)).parse_statements()


def parse_one(sql):
    stmts = parse(sql)
    if len(stmts) != 1:
        raise ParseError(f"expected exactly one statement, got {len(stmts)}")
    return stmts[0]
