"""Write-Ahead Log and on-disk checkpoint format.

WAL file layout
---------------
::

    MAGIC "MDBW" (4 bytes)
    frames...

each frame::

    uint32 payload_length (little endian)
    uint32 CRC32 of payload
    payload (codec-encoded dict)

A torn tail frame (short write, bad CRC) is ignored on recovery, which is
exactly what a crash mid-append looks like.

Durability rule: DML frames are appended (and flushed to the OS) *before* the
change becomes visible in memory; the COMMIT frame is ``fsync``ed.  A single
fsync therefore covers the whole transaction.  Readers never take the WAL
lock, so WAL writes never block reads.

Checkpoint
----------
``checkpoint.db`` holds every schema and the latest committed row image.  It is
written to a temp file and atomically renamed.  Afterwards the WAL is rewritten
to contain the frames of still-active transactions followed by a CHECKPOINT
marker.  Replay is deliberately tolerant (inserts of existing keys are
skipped, updates on missing keys become inserts), so a crash in the tiny window
between checkpoint rename and WAL rewrite is harmless.
"""

import os
import struct
import zlib
import threading

from ..codec import encode, decode

MAGIC = b"MDBW"
HEADER = struct.Struct("<II")


class WAL:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "wb") as f:
                f.write(MAGIC)
                f.flush()
                os.fsync(f.fileno())
        # Persistent append handle: avoids an open()/close() syscall pair for
        # every record. All access is serialised by ``self.lock``; readers use
        # a separate read path and never take this lock.
        self._fh = open(path, "ab")

    def close(self):
        with self.lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    # --------------------------------------------------------------- writing

    def append(self, record, fsync=False):
        """Append one record dict. Thread-safe; never blocks readers.

        Every frame is flushed to the OS so a separate read handle (used by
        checkpoints) can see it, but only the COMMIT frame is ``fsync``ed, so
        a transaction pays one disk barrier while its rows share it.
        """
        payload = encode(record)
        frame = HEADER.pack(len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload
        with self.lock:
            self._fh.write(frame)
            self._fh.flush()
            if fsync:
                os.fsync(self._fh.fileno())

    def rewrite(self, records, checkpoint_ts):
        """Replace the WAL contents with ``records`` plus a checkpoint marker.

        Used by checkpoints. Caller must coordinate so no appends race this
        (the engine holds its checkpoint lock).
        """
        with self.lock:
            self._fh.flush()
            self._fh.close()
            with open(self.path, "wb") as f:
                f.write(MAGIC)
                for rec in records:
                    payload = encode(rec)
                    f.write(HEADER.pack(len(payload),
                                       zlib.crc32(payload) & 0xFFFFFFFF))
                    f.write(payload)
                payload = encode({"type": "checkpoint", "ts": checkpoint_ts})
                f.write(HEADER.pack(len(payload),
                                    zlib.crc32(payload) & 0xFFFFFFFF))
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            self._fh = open(self.path, "ab")

    # --------------------------------------------------------------- reading

    def read_frames(self):
        """Return all decodable frames; a torn tail frame ends the scan."""
        records = []
        if not os.path.exists(self.path):
            return records
        with open(self.path, "rb") as f:
            data = f.read()
        if not data.startswith(MAGIC):
            raise ValueError("WAL file has bad magic header")
        pos = len(MAGIC)
        while pos + HEADER.size <= len(data):
            length, crc = HEADER.unpack_from(data, pos)
            pos += HEADER.size
            if pos + length > len(data):
                break  # torn frame
            payload = data[pos:pos + length]
            pos += length
            if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                break  # torn / corrupted frame
            records.append(decode(bytes(payload)))
        return records


def write_checkpoint(path, document, fsync_dir=False):
    """Atomically write a checkpoint document (dict)."""
    tmp = path + ".tmp"
    payload = encode(document)
    frame = HEADER.pack(len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(frame)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if fsync_dir:
        _fsync_dir(os.path.dirname(os.path.abspath(path)))


def read_checkpoint(path):
    """Return the checkpoint document, or None if no valid checkpoint exists."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data.startswith(MAGIC) or len(data) < len(MAGIC) + HEADER.size:
            return None
        length, crc = HEADER.unpack_from(data, len(MAGIC))
        payload = data[len(MAGIC) + HEADER.size: len(MAGIC) + HEADER.size + length]
        if len(payload) != length or zlib.crc32(payload) & 0xFFFFFFFF != crc:
            return None
        return decode(bytes(payload))
    except (OSError, ValueError):
        return None


def _fsync_dir(directory):
    if not directory:
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
