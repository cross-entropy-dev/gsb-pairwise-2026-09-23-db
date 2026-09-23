"""In-memory B+ tree with O(log n) point / range operations.

The B+ tree is used both as:

* the primary index of a table (keyed by primary key / row id), and
* a generic ordered map (e.g. tracking transaction ids).

Only the leaves are linked (leaf chain), which makes full scans and range
scans cheap.  All mutations happen in memory; durability is the WAL layer's
responsibility (see ``minidb.txn.wal``).
"""

from __future__ import annotations

from typing import Any, Generic, Iterator, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "<MISSING>"


_MISSING = _Missing()


class BPlusTree(Generic[K, V]):
    """A classic B+ tree.

    Parameters
    ----------
    order:
        Maximum number of keys per node.  Must be >= 3.  Internal nodes hold
        up to ``order`` keys / ``order + 1`` children; leaves up to ``order``
        entries.
    """

    def __init__(self, order: int = 32) -> None:
        if order < 3:
            raise ValueError("B+ tree order must be >= 3")
        self.order = order
        self.root: _Node[K, V] = _Leaf[K, V]()
        self._size = 0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: K) -> bool:
        return self.get(key) is not _MISSING

    def get(self, key: K, default: Any = _MISSING) -> Any:
        leaf, slot = self._find(key)
        if slot < len(leaf.keys) and leaf.keys[slot] == key:
            return leaf.values[slot]
        if default is _MISSING:
            raise KeyError(key)
        return default

    def __getitem__(self, key: K) -> V:
        return self.get(key)

    def insert(self, key: K, value: V) -> None:
        """Insert ``key`` or replace its value (no duplicate keys)."""
        leaf, slot = self._find(key)
        if slot < len(leaf.keys) and leaf.keys[slot] == key:
            leaf.values[slot] = value  # update in place
            return
        leaf.keys.insert(slot, key)
        leaf.values.insert(slot, value)
        self._size += 1
        if len(leaf.keys) > self.order:
            self._split_leaf(leaf)

    __setitem__ = insert

    def delete(self, key: K) -> bool:
        """Delete ``key``. Returns True if it existed.

        Deletion keeps the tree valid but does *not* eagerly merge underfull
        nodes (the tree shrinks via root collapse only).  This is the common
        pragmatic choice for an in-memory index where rebalancing cost is not
        worth it; O(log n) depth is preserved.
        """
        leaf, slot = self._find(key)
        if slot >= len(leaf.keys) or leaf.keys[slot] != key:
            return False
        leaf.keys.pop(slot)
        leaf.values.pop(slot)
        self._size -= 1
        self._collapse_after_delete()
        return True

    def __delitem__(self, key: K) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def range(self, low: Optional[K] = None, high: Optional[K] = None) -> Iterator[tuple[K, V]]:
        """Yield ``(key, value)`` with ``low <= key < high`` (None = open)."""
        if low is None:
            leaf = self._leftmost_leaf()
            slot = 0
        else:
            leaf, slot = self._find(low)
        while leaf is not None:
            while slot < len(leaf.keys):
                k = leaf.keys[slot]
                if high is not None and k >= high:
                    return
                yield k, leaf.values[slot]
                slot += 1
            leaf = leaf.next
            slot = 0

    def items(self) -> Iterator[tuple[K, V]]:
        yield from self.range()

    def keys_view(self) -> list[K]:
        return [k for k, _ in self.range()]

    def values(self) -> Iterator[V]:
        for _, v in self.range():
            yield v

    def min_key(self) -> Optional[K]:
        leaf = self._leftmost_leaf()
        return leaf.keys[0] if leaf.keys else None

    def depth(self) -> int:
        d = 1
        node = self.root
        while isinstance(node, _Internal):
            d += 1
            node = node.children[0]
        return d

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _find(self, key: K) -> tuple["_Leaf[K, V]", int]:
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[node.traverse(key)]
        # binary search the leaf slot
        keys = node.keys
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return node, lo

    def _leftmost_leaf(self) -> "_Leaf[K, V]":
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[0]
        return node  # type: ignore[return-value]

    def _split_leaf(self, leaf: "_Leaf[K, V]") -> None:
        mid = len(leaf.keys) // 2
        new = _Leaf[K, V]()
        new.keys = leaf.keys[mid:]
        new.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        # leaf chain
        new.next = leaf.next
        leaf.next = new
        new.parent = leaf.parent
        promoted = new.keys[0]
        self._insert_in_parent(leaf, promoted, new)

    def _insert_in_parent(
        self,
        left: "_Node[K, V]",
        key: K,
        right: "_Node[K, V]",
    ) -> None:
        parent = left.parent
        if parent is None:
            new_root = _Internal[K, V]()
            new_root.keys = [key]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self.root = new_root
            return
        idx = parent.traverse_lt(key)  # position of left child
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right)
        right.parent = parent
        if len(parent.keys) > self.order:
            self._split_internal(parent)

    def _split_internal(self, node: "_Internal[K, V]") -> None:
        mid = len(node.keys) // 2
        promoted = node.keys[mid]
        new = _Internal[K, V]()
        new.keys = node.keys[mid + 1:]
        new.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        for child in new.children:
            child.parent = new
        new.parent = node.parent
        self._insert_in_parent(node, promoted, new)

    def _collapse_after_delete(self) -> None:
        """Collapse an internal root with a single child down one level."""
        while isinstance(self.root, _Internal) and len(self.root.keys) == 0:
            self.root = self.root.children[0]
            self.root.parent = None


class _Node(Generic[K, V]):
    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: Optional[_Node[K, V]] = None


class _Leaf(_Node[K, V]):
    __slots__ = ("keys", "values", "next")

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[K] = []
        self.values: list[V] = []
        self.next: Optional[_Leaf[K, V]] = None


class _Internal(_Node[K, V]):
    __slots__ = ("keys", "children")

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[K] = []
        self.children: list[_Node[K, V]] = []

    def traverse(self, key: K) -> int:
        """Index of the child to descend into for ``key``."""
        lo, hi = 0, len(self.keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.keys[mid] <= key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def traverse_lt(self, key: K) -> int:
        """Insertion index of ``key`` (== index of the child strictly left)."""
        lo, hi = 0, len(self.keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if key < self.keys[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo
