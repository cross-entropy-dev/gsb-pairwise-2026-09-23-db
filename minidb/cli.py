"""Interactive SQL shell for minidb.

Run::

    python -m minidb.cli [data_dir]

Features:

* multi-line statements terminated by ``;``
* ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` (with optional isolation level)
* ``.tables`` / ``.describe <table>`` / ``.explain <select>`` meta-commands
* ``.quit`` / ``.exit`` (or Ctrl-D / Ctrl-C)
* result rendering in an aligned ASCII table
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from .engine.database import Database
from .engine.session import ExplainResult, Session
from .txn.manager import Isolation

PROMPT = "minidb> "
CONTINUE = "      -> "
BANNER = (
    "minidb -- MVCC in-memory SQL engine.  "
    "Type SQL ending with ';', or .help for commands."
)


def render_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    str_rows = [[fmt(v) for v in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(
            c.ljust(widths[i]) for i, c in enumerate(cells)
        ) + " |"

    out = [line(), render_row(columns), line("=")]
    out.extend(render_row(r) for r in str_rows)
    out.append(line())
    return "\n".join(out)


class Cli:
    def __init__(self, data_dir: str) -> None:
        self.db = Database(data_dir)
        self.session = Session(self.db)

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        print(BANNER)
        buffer = ""
        prompt = PROMPT
        try:
            while True:
                try:
                    line = input(prompt)
                except EOFError:
                    print()
                    break
                stripped = line.strip()
                if not buffer and stripped.startswith("."):
                    if self._meta_command(stripped):
                        break
                    continue
                buffer += line + "\n"
                if stripped.endswith(";") or stripped.upper() in (
                    "BEGIN", "COMMIT", "ROLLBACK"
                ):
                    statements, remainder = self._split_statements(buffer)
                    buffer = remainder if remainder.strip() else ""
                    prompt = PROMPT if not buffer else CONTINUE
                    for stmt in statements:
                        self._execute_and_print(stmt)
                else:
                    prompt = CONTINUE
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self.session.close()
            self.db.shutdown()
        return 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_statements(buffer: str) -> tuple[list[str], str]:
        # honour single-quoted strings when splitting on ';'
        stmts: list[str] = []
        cur: list[str] = []
        in_str = False
        i = 0
        while i < len(buffer):
            c = buffer[i]
            if c == "'":
                cur.append(c)
                if in_str and i + 1 < len(buffer) and buffer[i + 1] == "'":
                    cur.append(buffer[i + 1])
                    i += 2
                    continue
                in_str = not in_str
            elif c == ";" and not in_str:
                stmt = "".join(cur).strip()
                if stmt:
                    stmts.append(stmt)
                cur = []
            else:
                cur.append(c)
            i += 1
        return stmts, "".join(cur)

    def _execute_and_print(self, sql: str) -> None:
        try:
            result = self.session.sql(sql)
        except Exception as exc:
            print(f"ERROR: {exc}")
            # auto-abort an explicit transaction only on user request? keep
            # it open so the user can fix/rollback; autocommit already rolled.
            return

        if isinstance(result, ExplainResult):
            print("Query plan:")
            print(result.plan)
            return
        if isinstance(result, str):
            print(result)
            return
        if not result.rows:
            print("(no rows)")
            return
        print(render_table(result.columns, result.rows))
        noun = "row" if len(result.rows) == 1 else "rows"
        print(f"({len(result.rows)} {noun})")

    # ------------------------------------------------------------------ #
    def _meta_command(self, line: str) -> bool:
        """Handle a ``.command``. Returns True if the CLI should quit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip().rstrip(";").strip() if len(parts) > 1 else ""
        if cmd in (".quit", ".exit"):
            return True
        if cmd == ".help":
            print(
                "commands:\n"
                "  .tables                list tables\n"
                "  .describe <table>      show table schema\n"
                "  .explain <select>      show query plan\n"
                "  .txn                   show current transaction state\n"
                "  .quit                  exit (clean shutdown + checkpoint)"
            )
        elif cmd == ".tables":
            for name in self.db.table_names():
                print(name)
        elif cmd == ".describe":
            if not arg:
                print("usage: .describe <table>")
            else:
                try:
                    res = self.session.sql(f"DESCRIBE {arg}")
                    print(render_table(res.columns, res.rows))
                except Exception as exc:
                    print(f"ERROR: {exc}")
        elif cmd == ".explain":
            if not arg:
                print("usage: .explain <select>")
            else:
                try:
                    res = self.session.sql(f"EXPLAIN {arg}")
                    print(res)
                except Exception as exc:
                    print(f"ERROR: {exc}")
        elif cmd == ".txn":
            if self.session.in_transaction:
                t = self.session._txn
                print(f"active txn {t.txid}, isolation={t.isolation.value}")
            else:
                print("autocommit")
        else:
            print(f"unknown command: {cmd} (try .help)")
        return False


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    data_dir = argv[0] if argv else os.path.join(os.getcwd(), "minidb_data")
    return Cli(data_dir).run()


if __name__ == "__main__":
    raise SystemExit(main())
