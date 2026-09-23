"""Transaction manager: MVCC snapshots, commit protocol and crash recovery.

Concurrency model
-----------------
* **READ COMMITTED** - every statement gets a fresh snapshot of committed
  data.  Readers take *no* row locks and are never blocked by writers or by
  WAL appends.  Writers take X row locks (with IX table intention locks), so
  lost updates are prevented first-updater-wins: after blocking on a row the
  writer re-reads the latest committed version.
* **SERIALIZABLE** - strict two-phase locking on top of MVCC.  Point reads
  take S row locks, predicate/scan reads take an S table lock (phantom
  protection), writes take IX/X.  Every lock is held until commit/abort.

Durability
----------
DML frames are appended (flushed) *before* a version becomes visible in
memory.  The ``COMMIT`` frame is the single fsync point of the transaction.
Recovery loads the last checkpoint and replays committed WAL transactions in
commit order; frames of transactions without a COMMIT frame are discarded.
"""

import threading
from itertools import count

from ..errors import (TransactionError, ConstraintViolationError,
                      TableNotFoundError, TableExistsError)
from ..storage.table import Table, Schema, Version, BOOTSTRAP_TS

READ_COMMITTED = "READ COMMITTED"
SERIALIZABLE = "SERIALIZABLE"


class Transaction:
    def __init__(self, txn_id, isolation, watermark):
        self.txn_id = txn_id
        self.isolation = isolation
        self.status = "active"
        self.snapshot_ts = watermark   # current statement snapshot
        self.begin_ts = watermark      # watermark when the txn began
        self.touched_keys = []         # (table, key) written, in order
        self.wal_started = False
        self.created_tables = {}       # name -> Table, not yet committed
        self.dropped_tables = set()    # names dropped inside this txn

    @property
    def serializable(self):
        return self.isolation == SERIALIZABLE


class TransactionManager:
    def __init__(self, catalog, wal, lock_manager, clock_start=BOOTSTRAP_TS):
        self.catalog = catalog             # committed tables: name -> Table
        self.wal = wal
        self.lm = lock_manager
        self._clock = clock_start          # highest assigned commit timestamp
        self._txn_seq = count(1)
        self.active = {}                   # txn_id -> Transaction
        self._mutex = threading.RLock()
        self.commit_lock = threading.Lock()   # global fsync order

    # ------------------------------------------------------------- lifecycle

    def begin(self, isolation=READ_COMMITTED):
        with self._mutex:
            txn_id = next(self._txn_seq)
            txn = Transaction(txn_id, isolation, self._clock)
            self.active[txn_id] = txn
            return txn

    def require_table(self, txn, name):
        """Resolve a table visible to ``txn`` (its own new tables included)."""
        if name in txn.dropped_tables:
            raise TableNotFoundError(f"unknown table {name!r}")
        table = txn.created_tables.get(name) or self.catalog.get(name)
        if table is None:
            raise TableNotFoundError(f"unknown table {name!r}")
        return table

    def snapshot_for_statement(self, txn):
        """Refresh a READ COMMITTED statement snapshot."""
        if txn.isolation == READ_COMMITTED:
            with self._mutex:
                txn.snapshot_ts = self._clock
        return txn.snapshot_ts

    def oldest_snapshot(self):
        with self._mutex:
            if not self.active:
                return self._clock
            return min(t.snapshot_ts for t in self.active.values())

    # ------------------------------------------------------------------ DDL

    def create_table(self, txn, schema):
        with self._mutex:
            if schema.name in self.catalog or schema.name in txn.created_tables:
                raise TableExistsError(
                    f"table {schema.name!r} already exists")
            table = Table(schema)
            txn.created_tables[schema.name] = table   # staged until commit
            self._wal(txn, {"type": "ddl", "txn": txn.txn_id,
                            "op": "create", "schema": schema.to_dict()})
            return table

    def drop_table(self, txn, name):
        table = self.require_table(txn, name)
        if name in txn.created_tables:
            txn.created_tables.pop(name)
        else:
            txn.dropped_tables.add(name)
        self._wal(txn, {"type": "ddl", "txn": txn.txn_id,
                        "op": "drop", "table": name})
        return table

    # ------------------------------------------------------------------ DML

    def _wal(self, txn, record):
        if not txn.wal_started:
            self.wal.append({"type": "begin", "txn": txn.txn_id})
            txn.wal_started = True
        self.wal.append(record)

    def lock_for_read(self, txn, table, key):
        if txn.serializable:
            self.lm.lock_table(txn.txn_id, table.name, "IS")
            self.lm.lock_row(txn.txn_id, table.name, key, "S")

    def lock_predicate_read(self, txn, table):
        """Scan/predicate read under SERIALIZABLE -> S table (no phantoms)."""
        if txn.serializable:
            self.lm.lock_table(txn.txn_id, table.name, "S")

    def _prepare_write(self, txn, table, key):
        self.lm.lock_table(txn.txn_id, table.name, "IX")
        self.lm.lock_row(txn.txn_id, table.name, key, "X")

    def current_version(self, txn, table, key):
        """Version this txn should overwrite: its own write, else the latest
        committed version (re-read after the X lock was granted)."""
        return self._latest_writable(table.get_chain(key), txn.txn_id)

    def _check_unique(self, txn, table, key, data):
        """Serialize on each unique value, then reject committed duplicates."""
        for col in table.schema.columns:
            if not col.unique:
                continue
            value = data.get(col.name)
            if value is None:
                continue  # SQL: NULLs are not duplicate of each other
            self.lm.lock_value(
                txn.txn_id, ("U", table.name, col.name, value), "X")
            owner = table.unique_get(col.name, value)
            if owner is not None and owner != key:
                raise ConstraintViolationError(
                    f"unique constraint violated on {table.name}.{col.name}: "
                    f"duplicate value {value!r}")

    def insert_version(self, txn, table, key, data, wal=True):
        self._prepare_write(txn, table, key)
        chain = table.get_chain(key)
        if chain is not None:
            for v in chain:
                # A version blocks a new insert unless it is already gone:
                # committed-deleted, or deleted earlier by this same txn
                # (DELETE then INSERT of the same key is legal).
                gone = v.xmax_ts > 0 or v.xmax == txn.txn_id
                if not gone:
                    raise ConstraintViolationError(
                        f"duplicate primary key {key!r} in {table.name!r}")
        self._check_unique(txn, table, key, data)
        version = Version(dict(data), txn.txn_id)
        # Copy-on-write: publish a fresh chain list so concurrent scanners
        # that already hold the old list keep iterating an immutable object.
        if chain is None:
            if not table.create_chain(key, version):
                raise ConstraintViolationError(
                    f"duplicate primary key {key!r} in {table.name!r}")
        else:
            table.replace_chain(key, [version] + list(chain))
        txn.touched_keys.append((table.name, key))
        if wal:
            self._wal(txn, {"type": "insert", "txn": txn.txn_id,
                            "table": table.name, "key": key,
                            "data": dict(data)})

    def update_version(self, txn, table, key, data, wal=True):
        self._prepare_write(txn, table, key)
        chain = table.get_chain(key)
        current = self._latest_writable(chain, txn.txn_id)
        if current is None:
            raise TransactionError(
                f"cannot update missing row {key!r} in {table.name!r}")
        self._check_unique(txn, table, key, data)
        if current.xmin != txn.txn_id:
            # Uncommitted delete marker: concurrent snapshot readers ignore
            # it (xmax_ts is still 0), so mutating this field is safe.
            current.xmax = txn.txn_id
        new_version = Version(dict(data), txn.txn_id)
        table.replace_chain(key, [new_version] + list(chain))
        txn.touched_keys.append((table.name, key))
        if wal:
            self._wal(txn, {"type": "update", "txn": txn.txn_id,
                            "table": table.name, "key": key,
                            "data": dict(data)})
        return current

    def delete_version(self, txn, table, key, wal=True):
        self._prepare_write(txn, table, key)
        chain = table.get_chain(key)
        current = self._latest_writable(chain, txn.txn_id)
        if current is None:
            return False
        if current.xmin != txn.txn_id:
            current.xmax = txn.txn_id
        txn.touched_keys.append((table.name, key))
        if wal:
            self._wal(txn, {"type": "delete", "txn": txn.txn_id,
                            "table": table.name, "key": key})
        return True

    @staticmethod
    def _latest_writable(chain, txn_id):
        """Latest live version (own write wins, else latest committed)."""
        if chain is None:
            return None
        for v in chain:
            if v.xmin == txn_id:
                return v
        for v in chain:
            if v.xmin_ts > 0 and v.xmax_ts == 0:
                return v
        return None

    # -------------------------------------------------------------- commit

    def commit(self, txn):
        if txn.status != "active":
            raise TransactionError(
                f"transaction {txn.txn_id} already {txn.status}")
        with self.commit_lock:
            with self._mutex:
                self._clock += 1
                commit_ts = self._clock
                # Publish staged DDL before stamping so all touched tables
                # resolve in the committed catalog.
                self.catalog.update(txn.created_tables)
                for name in txn.dropped_tables:
                    self.catalog.pop(name, None)
                txn.created_tables.clear()
                txn.dropped_tables.clear()
                self._stamp(txn, commit_ts)
            if txn.wal_started:
                self.wal.append(
                    {"type": "commit", "txn": txn.txn_id, "ts": commit_ts},
                    fsync=True)
            txn.status = "committed"
            self._finish(txn)
        return commit_ts

    def _stamp(self, txn, ts):
        for tname, key in txn.touched_keys:
            table = self.catalog[tname]
            chain = table.get_chain(key)
            if chain is None:
                continue
            for v in chain:
                if v.xmin == txn.txn_id and v.xmin_ts == 0:
                    v.xmin_ts = ts
                if v.xmax == txn.txn_id and v.xmax_ts == 0:
                    v.xmax_ts = ts
            table.maintain_unique_indexes(chain, txn.txn_id, ts)
        # Opportunistic GC of versions on keys we touched.
        oldest = min((t.snapshot_ts for t in self.active.values()),
                     default=self._clock)
        for tname, key in txn.touched_keys:
            self.catalog[tname].prune(key, oldest)

    def _finish(self, txn):
        self.lm.release_all(txn.txn_id)
        with self._mutex:
            self.active.pop(txn.txn_id, None)

    # --------------------------------------------------------------- abort

    def abort(self, txn):
        if txn.status != "active":
            return
        txn.status = "aborted"
        self._undo(txn)
        if txn.wal_started:
            try:
                self.wal.append({"type": "abort", "txn": txn.txn_id})
            except OSError:
                pass
        self._finish(txn)

    def _undo(self, txn):
        # Reverse order: drop chains we inserted, clear our xmax markers.
        for tname, key in reversed(txn.touched_keys):
            table = self.catalog.get(tname) or txn.created_tables.get(tname)
            if table is None:
                continue
            chain = table.get_chain(key)
            if chain is None:
                continue
            kept = []
            for v in chain:
                if v.xmin == txn.txn_id:
                    continue
                if v.xmax == txn.txn_id:
                    v.xmax = 0
                    v.xmax_ts = 0
                kept.append(v)
            if kept:
                table.replace_chain(key, kept)
            else:
                table.remove_key(key)
        with self._mutex:
            # Unpublish staged tables; dropped tables were never removed.
            for name in txn.created_tables:
                self.catalog.pop(name, None)
            txn.created_tables.clear()
            txn.dropped_tables.clear()

    # ------------------------------------------------------------ recovery

    def recover(self, checkpoint_doc, records):
        """Load the checkpoint then replay WAL txns committed after it.

        The WAL may span multiple checkpoints.  Every frame is collected per
        transaction; a transaction's frames are applied only when a COMMIT
        frame follows *and* its timestamp is above the checkpoint watermark,
        because everything at or below the watermark is already materialised
        in the checkpoint image.  Frames of transactions that were still
        active at checkpoint time are physically kept in the rewritten WAL,
        so when they commit later their commit timestamp lands above the
        watermark and replay applies them correctly.
        """
        checkpoint_ts = BOOTSTRAP_TS
        if checkpoint_doc is not None:
            checkpoint_ts = checkpoint_doc.get("max_ts", BOOTSTRAP_TS)
            self._load_checkpoint(checkpoint_doc)

        frames = {}   # txn -> [records] in WAL order
        committed = []  # (ts, txn)
        for rec in records:
            rtype = rec.get("type")
            if rtype == "checkpoint" or rtype == "begin":
                continue
            txn = rec.get("txn")
            if txn is None:
                continue
            if rtype == "commit":
                committed.append((rec["ts"], txn))
            elif rtype == "abort":
                frames.pop(txn, None)
            else:
                frames.setdefault(txn, []).append(rec)

        max_ts = checkpoint_ts
        for ts, txn in committed:
            if ts <= checkpoint_ts:
                frames.pop(txn, None)
                continue
            for rec in frames.pop(txn, []):
                if rec.get("type") == "ddl":
                    self._apply_recovered_ddl(rec)
                else:
                    self._apply_recovered_dml(rec, ts)
            max_ts = max(max_ts, ts)

        self._clock = max_ts
        for table in self.catalog.values():
            table.rebuild_unique_indexes()
            table.recompute_counters()

    def _load_checkpoint(self, doc):
        for tname, meta in doc.get("tables", {}).items():
            schema = Schema.from_dict(meta["schema"])
            table = Table(schema)
            table.next_rowid = meta.get("next_rowid", 1)
            table.auto_counters = meta.get("auto_counters", {})
            for key, data in meta.get("rows", {}).items():
                table.tree.insert(key, [Version(dict(data), 0, BOOTSTRAP_TS)])
            table.rebuild_unique_indexes()
            self.catalog[tname] = table

    def _apply_recovered_ddl(self, rec):
        if rec["op"] == "create":
            schema = Schema.from_dict(rec["schema"])
            self.catalog[schema.name] = Table(schema)
        elif rec["op"] == "drop":
            self.catalog.pop(rec["table"], None)

    def _apply_recovered_dml(self, rec, ts):
        table = self.catalog.get(rec["table"])
        if table is None:
            return  # table later dropped
        key = rec["key"]
        if rec["type"] == "insert":
            table.tree.insert(key, [Version(dict(rec["data"]), 0, ts)])
            self._bump_counters(table, rec["data"])
        elif rec["type"] == "update":
            chain = table.get_chain(key)
            if chain is None:
                table.tree.insert(key, [Version(dict(rec["data"]), 0, ts)])
            else:
                chain[0].xmax = -1
                chain[0].xmax_ts = ts
                chain.insert(0, Version(dict(rec["data"]), 0, ts))
            self._bump_counters(table, rec["data"])
        elif rec["type"] == "delete":
            chain = table.get_chain(key)
            if chain is not None:
                chain[0].xmax = -1
                chain[0].xmax_ts = ts

    def _bump_counters(self, table, data):
        rid = data.get("_rowid_")
        if isinstance(rid, int):
            table.next_rowid = max(table.next_rowid, rid + 1)
        for col in table.schema.columns:
            if col.autoincrement:
                val = data.get(col.name)
                if isinstance(val, int):
                    table.auto_counters[col.name] = max(
                        table.auto_counters.get(col.name, 1), val + 1)
