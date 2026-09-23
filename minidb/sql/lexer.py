"""SQL lexer: turns a SQL string into a token stream.

Tokens produced:

* keywords / identifiers: ``(WORD, value)`` (case-insensitive keywords are
  normalised to upper case; identifiers keep their original case but are
  compared case-insensitively by the parser/executor);
* numbers: ``NUMBER`` (int or float in ``literal``);
* strings: ``STRING`` (single quotes, ``''`` escapes a quote);
* punctuation: ``( ) , ; . * + - / = <> != <= >= < >``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "INT", "INTEGER", "BIGINT", "SMALLINT",
    "TINYINT", "FLOAT", "DOUBLE", "REAL", "VARCHAR", "CHAR", "TEXT",
    "STRING", "BOOLEAN", "BOOL", "PRIMARY", "KEY", "NOT", "NULL", "TRUE",
    "FALSE", "AND", "OR", "IN", "IS", "LIKE", "BETWEEN", "ORDER", "BY",
    "ASC", "DESC", "LIMIT", "OFFSET", "GROUP", "HAVING", "JOIN", "INNER",
    "LEFT", "RIGHT", "OUTER", "ON", "AS", "COUNT", "SUM", "AVG", "MIN",
    "MAX", "DISTINCT", "UPPER", "LOWER", "BEGIN", "START", "TRANSACTION", "COMMIT", "ROLLBACK",
    "SERIALIZABLE", "COMMITTED", "UNCOMMITTED", "REPEATABLE", "ISOLATION",
    "LEVEL", "READ", "WORK", "SHOW", "TABLES", "DESCRIBE", "DESC",
    "EXPLAIN",
}


@dataclass
class Token:
    kind: str          # WORD, NUMBER, STRING, PUNCT, EOF
    value: Any
    pos: int
    literal: Any = None  # parsed value for NUMBER/STRING

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"Token({self.kind}, {self.value!r})"


class LexerError(Exception):
    def __init__(self, msg: str, pos: int) -> None:
        self.pos = pos
        super().__init__(f"SQL parse error at position {pos}: {msg}")


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.i = 0

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.i < self.n:
            c = self.text[self.i]
            if c in " \t\r\n":
                self.i += 1
                continue
            if c == "-" and self._peek(1) == "-":
                while self.i < self.n and self.text[self.i] != "\n":
                    self.i += 1
                continue
            if c == "/" and self._peek(1) == "*":
                self.i += 2
                while self.i < self.n and not (
                    self.text[self.i] == "*" and self._peek(1) == "/"
                ):
                    self.i += 1
                self.i += 2
                continue
            start = self.i
            if c.isalpha() or c == "_":
                tokens.append(self._read_word(start))
            elif c.isdigit() or (c == "." and self._peek(1).isdigit()):
                tokens.append(self._read_number(start))
            elif c == "'":
                tokens.append(self._read_string(start))
            elif c == '"':
                tokens.append(self._read_quoted_ident(start))
            else:
                tokens.append(self._read_punct(start))
        tokens.append(Token("EOF", None, self.i))
        return tokens

    # ------------------------------------------------------------------ #
    def _peek(self, offset: int = 0) -> str:
        j = self.i + offset
        return self.text[j] if j < self.n else ""

    def _read_word(self, start: int) -> Token:
        while self.i < self.n and (
            self.text[self.i].isalnum() or self.text[self.i] == "_"
        ):
            self.i += 1
        word = self.text[start:self.i]
        upper = word.upper()
        if upper in KEYWORDS:
            return Token("WORD", upper, start)
        return Token("IDENT", word, start)

    def _read_number(self, start: int) -> Token:
        is_float = False
        while self.i < self.n and self.text[self.i].isdigit():
            self.i += 1
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self.i += 1
            while self.i < self.n and self.text[self.i].isdigit():
                self.i += 1
        if self._peek() in ("e", "E"):
            is_float = True
            self.i += 1
            if self._peek() in ("+", "-"):
                self.i += 1
            while self.i < self.n and self.text[self.i].isdigit():
                self.i += 1
        raw = self.text[start:self.i]
        value = float(raw) if is_float else int(raw)
        return Token("NUMBER", raw, start, value)

    def _read_string(self, start: int) -> Token:
        self.i += 1  # opening quote
        chars: list[str] = []
        while self.i < self.n:
            c = self.text[self.i]
            if c == "'":
                if self._peek(1) == "'":
                    chars.append("'")
                    self.i += 2
                    continue
                self.i += 1
                return Token("STRING", self.text[start:self.i], start, "".join(chars))
            chars.append(c)
            self.i += 1
        raise LexerError("unterminated string literal", start)

    def _read_quoted_ident(self, start: int) -> Token:
        self.i += 1
        chars: list[str] = []
        while self.i < self.n:
            c = self.text[self.i]
            if c == '"':
                self.i += 1
                return Token("IDENT", "".join(chars), start)
            chars.append(c)
            self.i += 1
        raise LexerError("unterminated quoted identifier", start)

    _TWO_CHAR = {"<=", ">=", "!=", "<>"}

    def _read_punct(self, start: int) -> Token:
        two = self.text[self.i:self.i + 2]
        if two in self._TWO_CHAR:
            self.i += 2
            norm = "!=" if two == "<>" else two
            return Token("PUNCT", norm, start)
        c = self.text[self.i]
        if c in "(),;.*+-/=":
            self.i += 1
            return Token("PUNCT", c, start)
        if c in "<>":
            self.i += 1
            return Token("PUNCT", c, start)
        raise LexerError(f"unexpected character {c!r}", start)
