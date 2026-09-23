"""Self-contained binary codec.

No third-party libraries (not even pickle) are used.  Supported values:
None, bool, int (64-bit), float (double), str (UTF-8), bytes, list, tuple, dict.

Every encoded document is framed by the caller (see :mod:`minidb.wal`).
"""

import struct

TAG_NONE = 0
TAG_TRUE = 1
TAG_FALSE = 2
TAG_INT = 3
TAG_FLOAT = 4
TAG_STR = 5
TAG_BYTES = 6
TAG_LIST = 7
TAG_DICT = 8


class Encoder:
    def __init__(self):
        self.buf = bytearray()

    def encode(self, value):
        t = type(value)
        if value is None:
            self.buf.append(TAG_NONE)
        elif t is bool:
            self.buf.append(TAG_TRUE if value else TAG_FALSE)
        elif t is int:
            if not (-(2 ** 63) <= value < 2 ** 63):
                # fall back to a string for out-of-range integers
                self._write_str(str(value), TAG_STR)
            else:
                self.buf.append(TAG_INT)
                self.buf += struct.pack("<q", value)
        elif t is float:
            self.buf.append(TAG_FLOAT)
            self.buf += struct.pack("<d", value)
        elif t is str:
            self._write_str(value, TAG_STR)
        elif t is bytes:
            data = value.encode("utf-8", "surrogateescape") if False else value
            self.buf.append(TAG_BYTES)
            self.buf += struct.pack("<I", len(value))
            self.buf += value
        elif t in (list, tuple):
            self.buf.append(TAG_LIST)
            self.buf += struct.pack("<I", len(value))
            for item in value:
                self.encode(item)
        elif t is dict:
            self.buf.append(TAG_DICT)
            self.buf += struct.pack("<I", len(value))
            for k, v in value.items():
                self.encode(k)
                self.encode(v)
        else:
            raise TypeError(f"cannot encode value of type {type(value).__name__}")
        return bytes(self.buf)

    def _write_str(self, value, tag):
        data = value.encode("utf-8")
        self.buf.append(tag)
        self.buf += struct.pack("<I", len(data))
        self.buf += data


def encode(value):
    return Encoder().encode(value)


def decode(data):
    value, end = _decode_at(data, 0)
    if end != len(data):
        raise ValueError("trailing bytes in encoded document")
    return value


def _decode_at(data, pos):
    tag = data[pos]
    pos += 1
    if tag == TAG_NONE:
        return None, pos
    if tag == TAG_TRUE:
        return True, pos
    if tag == TAG_FALSE:
        return False, pos
    if tag == TAG_INT:
        (v,) = struct.unpack_from("<q", data, pos)
        return v, pos + 8
    if tag == TAG_FLOAT:
        (v,) = struct.unpack_from("<d", data, pos)
        return v, pos + 8
    if tag in (TAG_STR, TAG_BYTES):
        (length,) = struct.unpack_from("<I", data, pos)
        pos += 4
        raw = data[pos:pos + length]
        pos += length
        if tag == TAG_STR:
            return raw.decode("utf-8"), pos
        return bytes(raw), pos
    if tag == TAG_LIST:
        (count,) = struct.unpack_from("<I", data, pos)
        pos += 4
        out = []
        for _ in range(count):
            item, pos = _decode_at(data, pos)
            out.append(item)
        return out, pos
    if tag == TAG_DICT:
        (count,) = struct.unpack_from("<I", data, pos)
        pos += 4
        out = {}
        for _ in range(count):
            k, pos = _decode_at(data, pos)
            v, pos = _decode_at(data, pos)
            out[k] = v
        return out, pos
    raise ValueError(f"unknown type tag {tag} at {pos - 1}")
