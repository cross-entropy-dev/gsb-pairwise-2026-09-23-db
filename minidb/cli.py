"""Interactive command-line shell for minidb.

Run::

    python -m minidb.cli [data_directory]

With no directory the database is purely in-memory.  Inside the shell:

* SQL statements end with ``;`` and may span multiple lines.
* ``.tables`` lists tables, ``.schema <table>`` shows a table definition.
* ``.indexes`` is reserved, ``.isolation <rc|serializable>`` sets the level
  used for ``BEGIN``.
* ``.checkpoint`` forces a checkpoint, ``.exit`` / ``.quit`` leaves.
"""

import argparse
import sys

from . import Database, READ_COMMITTED, SERIALIZABLE, MiniDBError
from .sql.parser import parse


HELP = """\
Commands:
  .tables                       list tables
  .schema <table>               show CREATE TABLE definition
  .isolation rc|serializable    default isolation for new transactions
  .checkpoint                   force a checkpoint to disk
  .help                         show this help
  .exit / .quit                 leave the shell
Anything else ending with ';' is executed as SQL.
"""


def format_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


def render_table(columns, rows):
    matrix = [[format_value(r.get(c)) for c in columns] for r in rows]
    widths = [len(c) for c in columns]
    for r in matrix:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(ch="-"):
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    parts = [line(),
             "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |",
             line()]
    for r in matrix:
        parts.append("| " + " | ".join(cell.ljust(w)
                                       for cell, w in zip(r, widths)) + " |")
    parts.append(line())
    parts.append(f"({len(matrix)} row"
                 f"{'s' if len(matrix) != 1 else ''})")
    return "\n".join(parts)


class Shell:
    def __init__(self, db):
        self.db = db
        self.txn = None
        self.isolation = READ_COMMITTED

    # ----------------------------------------------------------------- run

    def run_stream(self, stream):
        buffer = ""
        for raw in stream:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("."):
                if buffer.strip():
                    self._run_sql(buffer)
                    buffer = ""
                self._dot_command(stripped)
                continue
            buffer += line + "\n"
            if stripped.endswith(";"):
                self._run_sql(buffer)
                buffer = ""
        if buffer.strip():
            self._run_sql(buffer)
        self._close_txn_if_needed()

    def run_interactive(self):
        print("minidb interactive shell  (type .help for help, .exit to quit)")
        buffer = ""
        while True:
            prompt = "minidb> " if not buffer else "    ...> "
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            stripped = line.strip()
            if not buffer and stripped.startswith("."):
                self._dot_command(stripped)
                if self._exit_requested:
                    break
                continue
            buffer += line + "\n"
            if stripped.endswith(";"):
                self._run_sql(buffer)
                buffer = ""
        if buffer.strip():
            self._run_sql(buffer)
        self._close_txn_if_needed()

    _exit_requested = False

    # ------------------------------------------------------------- commands

    def _dot_command(self, text):
        parts = text.split()
        cmd = parts[0].lower()
        try:
            if cmd in (".exit", ".quit"):
                self._close_txn_if_needed()
                self._exit_requested = True
            elif cmd == ".help":
                print(HELP)
            elif cmd == ".tables":
                names = self.db.table_names()
                print("\n".join(names) if names else "(no tables)")
            elif cmd == ".schema":
                self._show_schema(parts[1] if len(parts) > 1 else None)
            elif cmd == ".isolation":
                if len(parts) < 2 or parts[1].lower() in ("rc", "read_committed",
                                                          "readcommitted"):
                    self.isolation = READ_COMMITTED
                elif parts[1].lower() == "serializable":
                    self.isolation = SERIALIZABLE
                else:
                    print("unknown isolation level")
                    return
                print(f"default isolation: {self.isolation}")
            elif cmd == ".checkpoint":
                ts = self.db.checkpoint()
                print(f"checkpoint written (max ts {ts})")
            else:
                print(f"unknown command {cmd}")
        except MiniDBError as exc:
            print(f"error: {exc}")

    def _show_schema(self, name):
        if name is None:
            print("usage: .schema <table>")
            return
        table = self.db.catalog.get(name)
        if table is None:
            print(f"unknown table {name}")
            return
        cols = []
        for c in table.schema.columns:
            bits = [c.name, c.type_name +
                    (f"({c.length})" if c.length else "")]
            if c.primary_key:
                bits.append("PRIMARY KEY")
            if c.not_null and not c.primary_key:
                bits.append("NOT NULL")
            if c.unique and not c.primary_key:
                bits.append("UNIQUE")
            if c.autoincrement:
                bits.append("AUTOINCREMENT")
            cols.append("    " + " ".join(bits))
        print(f"CREATE TABLE {name} (\n" + ",\n".join(cols) + "\n)")

    # ----------------------------------------------------------------- sql

    def _run_sql(self, text):
        try:
            statements = parse(text)
        except MiniDBError as exc:
            print(f"parse error: {exc}")
            return
        for node in statements:
            try:
                result = self.db._run(node, self.txn, self.isolation)
            except MiniDBError as exc:
                print(f"error: {exc}")
                # Drop any failed explicit transaction context for the user;
                # autocommit txns are already rolled back by _run.
                if self.txn is not None and self.txn.status != "active":
                    self.txn = None
                return
            self._after_result(node, result)

    def _after_result(self, node, result):
        from .sql import ast_nodes as ast
        if isinstance(node, ast.Begin):
            self.txn = result
            print(f"BEGIN (txn {result.txn_id}, {result.isolation})")
        elif isinstance(node, ast.Commit):
            print("COMMIT")
            self.txn = None
        elif isinstance(node, ast.Rollback):
            print("ROLLBACK")
            self.txn = None
        elif result.columns:
            print(render_table(result.columns, result.rows))
        elif result.message:
            print(result.message)

    def _close_txn_if_needed(self):
        if self.txn is not None and self.txn.status == "active":
            self.db.rollback(self.txn)
            self.txn = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="minidb interactive shell")
    parser.add_argument("directory", nargs="?", default=":memory:",
                        help="data directory (default: in-memory)")
    args = parser.parse_args(argv)
    db = Database(args.directory)
    shell = Shell(db)
    if not sys.stdin.isatty():
        shell.run_stream(sys.stdin)
    else:
        shell.run_interactive()
    db.close()


if __name__ == "__main__":
    main()
