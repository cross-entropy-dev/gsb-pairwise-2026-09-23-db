"""SQL lexer: turns a SQL string into a stream of tokens.

Token types:
    KEYWORD     reserved word (value stored upper-cased)
    IDENT       identifier or unreserved word (value as written)
    NUMBER      integer or float literal
    STRING      string literal
    OP          operator or punctuation (``=``, ``<=``, ``,`` ...)

String literals use single quotes with ``''`` as an escaped quote.
Identifiers may be quoted with double quotes.
"""

from ..errors import LexerError

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DROP", "PRIMARY", "KEY", "NOT", "NULL",
    "DEFAULT", "INT", "INTEGER", "BIGINT", "VARCHAR", "TEXT", "BOOLEAN",
    "BOOL", "FLOAT", "DOUBLE", "REAL", "AND", "OR", "IS", "LIKE", "IN",
    "BETWEEN", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "FULL", "ON",
    "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "AS",
    "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "BEGIN", "START",
    "TRANSACTION", "COMMIT", "ROLLBACK", "ISOLATION", "LEVEL", "READ",
    "COMMITTED", "SERIALIZABLE", "REPEATABLE", "UNCOMMITTED", "EXISTS",
    "IF",
    "CASE", "WHEN", "THEN", "ELSE", "END", "UNION", "ALL", "TRUE", "FALSE",
    "AUTOINCREMENT", "UNIQUE",
}

# Multi-character operators, longest first.
OPERATORS = ("<>", "<=", ">=", "!=", "=", "<", ">", "+", "-", "*", "/",
             "(", ")", ",", ";", ".")


class Token:
    __slots__ = ("type", "value", "pos", "line")

    def __init__(self, type_, value, pos, line):
        self.type = type_
        self.value = value
        self.pos = pos
        self.line = line

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        if isinstance(other, tuple):
            return (self.type, self.value) == other
        return NotImplemented


class Lexer:
    def __init__(self, text):
        self.text = text
        self.n = len(text)
        self.i = 0
        self.line = 1
        self.tokens = []

    def tokenize(self):
        while self.i < self.n:
            ch = self.text[self.i]
            if ch in " \t\r":
                self.i += 1
            elif ch == "\n":
                self.line += 1
                self.i += 1
            elif ch == "-" and self._peek(1) == "-":
                self._consume_line_comment()
            elif ch == "/" and self._peek(1) == "*":
                self._consume_block_comment()
            elif ch == "'":
                self._read_string()
            elif ch == '"':
                self._read_quoted_identifier()
            elif ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                self._read_number()
            elif ch.isalpha() or ch == "_":
                self._read_word()
            else:
                self._read_operator()
        return self.tokens

    def _peek(self, offset):
        j = self.i + offset
        return self.text[j] if j < self.n else ""

    def _consume_line_comment(self):
        while self.i < self.n and self.text[self.i] != "\n":
            self.i += 1

    def _consume_block_comment(self):
        self.i += 2
        while self.i < self.n:
            if self.text[self.i] == "*" and self._peek(1) == "/":
                self.i += 2
                return
            if self.text[self.i] == "\n":
                self.line += 1
            self.i += 1
        raise LexerError("unterminated block comment")

    def _read_string(self):
        start = self.i
        start_line = self.line
        self.i += 1  # opening quote
        chars = []
        while self.i < self.n:
            ch = self.text[self.i]
            if ch == "'":
                if self._peek(1) == "'":
                    chars.append("'")
                    self.i += 2
                    continue
                self.i += 1
                self.tokens.append(Token("STRING", "".join(chars), start, start_line))
                return
            if ch == "\n":
                self.line += 1
            chars.append(ch)
            self.i += 1
        raise LexerError(f"unterminated string literal at line {start_line}")

    def _read_quoted_identifier(self):
        start = self.i
        start_line = self.line
        self.i += 1
        chars = []
        while self.i < self.n:
            ch = self.text[self.i]
            if ch == '"':
                if self._peek(1) == '"':
                    chars.append('"')
                    self.i += 2
                    continue
                self.i += 1
                self.tokens.append(Token("IDENT", "".join(chars), start, start_line))
                return
            chars.append(ch)
            self.i += 1
        raise LexerError(f"unterminated quoted identifier at line {start_line}")

    def _read_number(self):
        start = self.i
        start_line = self.line
        is_float = False
        while self.i < self.n and self.text[self.i].isdigit():
            self.i += 1
        if self.i < self.n and self.text[self.i] == ".":
            # Avoid swallowing a trailing dot used as a table separator.
            if self._peek(1).isdigit():
                is_float = True
                self.i += 1
                while self.i < self.n and self.text[self.i].isdigit():
                    self.i += 1
        if self.i < self.n and self.text[self.i] in "eE":
            j = self.i + 1
            if j < self.n and self.text[j] in "+-":
                j += 1
            if j < self.n and self.text[j].isdigit():
                is_float = True
                self.i = j
                while self.i < self.n and self.text[self.i].isdigit():
                    self.i += 1
        raw = self.text[start:self.i]
        value = float(raw) if is_float else int(raw)
        self.tokens.append(Token("NUMBER", value, start, start_line))

    def _read_word(self):
        start = self.i
        start_line = self.line
        while self.i < self.n and (self.text[self.i].isalnum() or self.text[self.i] == "_"):
            self.i += 1
        word = self.text[start:self.i]
        upper = word.upper()
        if upper in KEYWORDS:
            self.tokens.append(Token("KEYWORD", upper, start, start_line))
        else:
            self.tokens.append(Token("IDENT", word, start, start_line))

    def _read_operator(self):
        start = self.i
        start_line = self.line
        for op in OPERATORS:
            if self.text.startswith(op, self.i):
                self.tokens.append(Token("OP", op, start, start_line))
                self.i += len(op)
                return
        raise LexerError(f"unexpected character {self.text[self.i]!r} at line {self.line}")


def tokenize(sql):
    return Lexer(sql).tokenize()
