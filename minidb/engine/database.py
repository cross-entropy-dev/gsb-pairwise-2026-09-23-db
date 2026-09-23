"""Database kernel: catalog, wiring and crash recovery.

On-disk layout under ``data_dir``::

    wal.log         append-only write-ahead log
    checkpoint.json last consistent snapshot (written on clean shutdown)

Recovery combines the checkpoint (a snapshot taken at a quiescent point)
with WAL records of transactions that committed afterwards.  Records of
transactions without a commit record – the signature of a crash mid-flight –
are discarded, which is what restores atomicity after a crash.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

from ..storage.schema import SchemaError, TableSchema
from ..storage.table import Table, Version
from ..txn.locks import LockManager
from ..txn.manager import Isolation, Transaction, TransactionManager
from ..txn.wal import WAL, decode_key, encode_key, read_wal


class CatalogError(Exception):
    pass


class Database:
    def __init__(self, data_dir: str, lock_escalation: int = 100) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.wal_path = os.path.join(data_dir, "wal.log")
        self.checkpoint_path = os.path.join(data_dir, "checkpoint.json")

        self.tables: dict[str, Table] = {}
        self._catalog_lock = threading.RLock()

        self.wal = WAL(self.wal_path)
        self.lock_manager = LockManager(escalation_threshold=lock_escalation)
        self.txn_manager = TransactionManager(self.wal, self.lock_manager)

        self._recover()

        # background GC of dead MVCC versions
        self._gc_stop = threading.Event()
        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="mvcc-gc", daemon=True
        )
        self._gc_thread.start()

    # ------------------------------------------------------------------ #
    # catalog
    # ------------------------------------------------------------------ #
    def create_table(self, txn: Transaction, schema: TableSchema) -> Table:
        if not schema.columns:
            raise SchemaError(f"table {schema.name!r} needs at least one column")
        names = [c.name.lower() for c in schema.columns]
        if len(set(names)) != len(names):
            raise SchemaError(f"duplicate column names in table {schema.name!r}")
        pks = schema.primary_key_columns
        if len(pks) > 1:
            raise SchemaError("only single-column PRIMARY KEY is supported")
        with self._catalog_lock:
            if schema.name.lower() in self.tables:
                raise CatalogError(f"table {schema.name!r} already exists")
            table = Table(schema)
            # log first, install after (crash between the two is harmless:
            # recovery performs the install itself)
            self.txn_manager.ensure_wal_begin(txn)
            self.wal.log_create_table(txn.txid, schema.to_dict())
            self.tables[schema.name.lower()] = table
            txn.did_ddl = True
            txn.written_tables.add(schema.name.lower())
            return table

    def get_table(self, name: str) -> Table:
        with self._catalog_lock:
            try:
                return self.tables[name.lower()]
            except KeyError:
                raise CatalogError(f"no such table: {name}")

    def has_table(self, name: str) -> bool:
        return name.lower() in self.tables

    def table_names(self) -> list[str]:
        with self._catalog_lock:
            return sorted(t.schema.name for t in self.tables.values())

    # ------------------------------------------------------------------ #
    # recovery
    # ------------------------------------------------------------------ #
    def _recover(self) -> None:
        max_txid = 0

        # 1. checkpoint (only exists from a clean shutdown)
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for td in data["tables"]:
                schema = TableSchema.from_dict(td["schema"])
                table = Table(schema)
                table.next_rowid = td["next_rowid"]
                for rd in td["rows"]:
                    key = decode_key(rd["key"])
                    table.tree.insert(key, [Version(row=tuple(rd["row"]), xmin=0)])
                self.tables[schema.name.lower()] = table
        # 2. committed WAL records
        committed, records = read_wal(self.wal_path)
        max_txid = max(committed, default=0)
        for rec in records:
            max_txid = max(max_txid, rec["txid"])
            self._apply_replay_record(rec)

        self.txn_manager.bootstrap_state(max_txid)

    def _apply_replay_record(self, rec: dict) -> None:
        ty = rec["ty"]
        txid = rec["txid"]
        if ty == "begin":
            return
        if ty == "create":
            schema = TableSchema.from_dict(rec["schema"])
            if schema.name.lower() not in self.tables:
                self.tables[schema.name.lower()] = Table(schema)
            return
        table = self.tables[rec["table"].lower()]
        key = decode_key(rec["key"])
        if ty == "insert":
            row = tuple(rec["row"])
            chain = table.tree.get(key, None)
            version = Version(row=row, xmin=txid)
            if chain is None:
                table.tree.insert(key, [version])
            else:
                chain.append(version)
            table.bump_rowid(key)
        elif ty == "update":
            row = tuple(rec["row"])
            chain = table.tree[key]
            chain[-1].xmax = txid
            chain.append(Version(row=row, xmin=txid))
        elif ty == "delete":
            chain = table.tree[key]
            chain[-1].xmax = txid

    # ------------------------------------------------------------------ #
    # GC + shutdown
    # ------------------------------------------------------------------ #
    def _gc_loop(self) -> None:
        while not self._gc_stop.wait(5.0):
            try:
                self.txn_manager.gc_tick(self.tables)
            except Exception:  # pragma: no cover - GC must never kill the db
                pass

    def checkpoint(self) -> None:
        """Write a consistent snapshot and truncate the WAL.

        Callers must ensure no transactions are active (``shutdown`` aborts
        all of them first).
        """
        if self.txn_manager.active_count() > 0:
            raise CatalogError("cannot checkpoint while transactions are active")
        # GC first so the snapshot contains only live, committed rows.
        self.txn_manager.gc_tick(self.tables)
        payload = {"tables": []}
        for table in self.tables.values():
            rows = []
            for key, chain in table.tree.items():
                v = chain[-1]
                rows.append({"key": encode_key(key), "row": list(v.row)})
            payload["tables"].append({
                "schema": table.schema.to_dict(),
                "next_rowid": table.next_rowid,
                "rows": rows,
            })
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.checkpoint_path)
        self.wal.reset()

    def shutdown(self, checkpoint: bool = True) -> None:
        self.txn_manager.shutdown()  # abort stragglers
        self._gc_stop.set()
        self._gc_thread.join(timeout=5)
        if checkpoint:
            try:
                self.checkpoint()
            except Exception:
                pass
        self.wal.close()
