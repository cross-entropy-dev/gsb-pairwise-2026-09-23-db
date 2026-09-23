"""Write-Ahead Log.

Records are JSON objects, one per line, appended in transaction order::

    {"ty": "begin",  "txid": 7}
    {"ty": "create", "txid": 7, "schema": {...}}
    {"ty": "insert", "txid": 7, "table": "t", "key": 1, "row": [...]}
    {"ty": "update", "txid": 7, "table": "t", "key": 1, "row": [...]}
    {"ty": "delete", "txid": 7, "table": "t", "key": 1}
    {"ty": "commit", "txid": 7}

A dedicated background thread owns the file handle.  Mutations are handed to
it through a queue and a ``commit`` request carries an event that is set only
after ``flush()`` + ``os.fsync()``, so:

* readers never touch the WAL (logging does not block reads);
* a transaction reports success only after its commit record is durable;
* recovery replays records of *committed* transactions only, which gives
  atomicity across a crash (an ``abort`` record or a missing ``commit`` both
  mean the transaction never happened).
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any, Optional


class WALError(Exception):
    pass


def encode_key(key: Any) -> Any:
    """Make a row key JSON-serialisable (composite PKs are tuples)."""
    if isinstance(key, tuple):
        return {"__tuple__": [encode_key(k) for k in key]}
    return key


def decode_key(key: Any) -> Any:
    if isinstance(key, dict) and "__tuple__" in key:
        return tuple(decode_key(k) for k in key["__tuple__"])
    return key


class _CommitReq:
    __slots__ = ("txid", "event", "error")

    def __init__(self, txid: int) -> None:
        self.txid = txid
        self.event = threading.Event()
        self.error: Optional[BaseException] = None


class WAL:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._truncate_torn_tail()
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._file = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()  # only serialises commit bookkeeping
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="wal-writer", daemon=True
        )
        self._thread.start()

    def _truncate_torn_tail(self) -> None:
        """Remove a final non-newline-terminated record.

        A crash can leave a partial append (torn write).  Keeping it would
        make the very next append land on the same physical line and corrupt
        both records, so truncate back to the last complete line.
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return
        with open(self.path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            f.seek(max(0, size - block))
            tail = f.read()
            if tail.endswith(b"\n"):
                return
            last_nl = tail.rfind(b"\n")
            if last_nl == -1:
                f.seek(0)
                f.truncate()
            else:
                f.seek(max(0, size - block) + last_nl + 1)
                f.truncate()

    # ------------------------------------------------------------------ #
    # logging API (cheap, non-blocking for callers)
    # ------------------------------------------------------------------ #
    def _put(self, record: dict[str, Any]) -> None:
        self._q.put(record)

    def log_begin(self, txid: int) -> None:
        self._put({"ty": "begin", "txid": txid})

    def log_create_table(self, txid: int, schema: dict[str, Any]) -> None:
        self._put({"ty": "create", "txid": txid, "schema": schema})

    def log_insert(self, txid: int, table: str, key: Any, row: tuple) -> None:
        self._put(
            {"ty": "insert", "txid": txid, "table": table,
             "key": encode_key(key), "row": list(row)}
        )

    def log_update(self, txid: int, table: str, key: Any, row: tuple) -> None:
        self._put(
            {"ty": "update", "txid": txid, "table": table,
             "key": encode_key(key), "row": list(row)}
        )

    def log_delete(self, txid: int, table: str, key: Any) -> None:
        self._put(
            {"ty": "delete", "txid": txid, "table": table,
             "key": encode_key(key)}
        )

    def log_abort(self, txid: int) -> None:
        self._put({"ty": "abort", "txid": txid})

    def commit(self, txid: int) -> None:
        """Append the commit record and block until it is fsync'd."""
        req = _CommitReq(txid)
        self._q.put({"ty": "commit", "txid": txid, "_req": req})
        req.event.wait()
        if req.error is not None:
            raise WALError(f"failed to persist commit for txn {txid}") from req.error

    # ------------------------------------------------------------------ #
    # background writer
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                    return
                req = item.pop("_req", None)
                self._file.write(json.dumps(item, separators=(",", ":")) + "\n")
                if req is not None:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                    req.event.set()
            except BaseException as exc:  # pragma: no cover - disk failure
                if req is not None:
                    req.error = exc
                    req.event.set()

    def close(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._q.put(None)
        self._thread.join(timeout=10)
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def reset(self) -> None:
        """Truncate the log (used after a durable checkpoint)."""
        self.close()
        with open(self.path, "w", encoding="utf-8"):
            pass
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")
        self._q = queue.Queue()
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name="wal-writer", daemon=True
        )
        self._thread.start()


# ---------------------------------------------------------------------- #
# recovery
# ---------------------------------------------------------------------- #
def read_wal(path: str) -> tuple[set[int], list[dict[str, Any]]]:
    """Parse a WAL file.

    Returns ``(committed_txids, data_records)`` where ``data_records`` only
    contains records belonging to committed transactions.
    """
    committed: set[int] = set()
    aborted: set[int] = set()
    all_records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return committed, all_records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A torn append from a crash: ignore the partial tail.
                break
            ty = rec.get("ty")
            if ty == "commit":
                committed.add(rec["txid"])
            elif ty == "abort":
                aborted.add(rec["txid"])
            else:
                all_records.append(rec)
    committed -= aborted
    records = [r for r in all_records if r.get("txid") in committed]
    return committed, records
