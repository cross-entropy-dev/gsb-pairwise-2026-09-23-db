"""An in-memory B+ tree used as the primary index of every table.

Capacity scheme with an even ``order`` b (default 64):

* leaf nodes hold at most ``b`` entries,
* internal nodes have at most ``b`` children (``b - 1`` separator keys),
* leaves are rebalanced below ``b / 2`` entries, internal nodes below
  ``b / 2`` children (``b / 2 - 1`` keys),
* merging two minimum siblings plus the separator fills a node to exactly
  its capacity (``b`` leaf entries / ``b - 1`` internal keys), never over.

Leaf nodes are linked, giving O(log n) point lookups and chain-walk range
scans.  ``to_dict`` / ``from_dict`` persist ordered leaf contents.
"""

from bisect import bisect_left, bisect_right


class _Node:
    __slots__ = ("keys",)

    def __init__(self):
        self.keys = []


class _InternalNode(_Node):
    __slots__ = ("children",)

    def __init__(self):
        super().__init__()
        self.children = []


class _LeafNode(_Node):
    __slots__ = ("values", "next")

    def __init__(self):
        super().__init__()
        self.keys = []
        self.values = []
        self.next = None


class BPlusTree:
    def __init__(self, order=64):
        if order < 4 or order % 2:
            raise ValueError("B+ tree order must be an even integer >= 4")
        self.order = order
        self.root = _LeafNode()
        self._size = 0

    def _min_keys(self, node):
        # Minimum key count for a non-root node.
        half = self.order // 2
        return half if isinstance(node, _LeafNode) else half - 1

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    # ---------------------------------------------------------------- reads

    def get(self, key, default=None):
        leaf, idx = self._find_leaf(key)
        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            return leaf.values[idx]
        return default

    def __contains__(self, key):
        leaf, idx = self._find_leaf(key)
        return idx < len(leaf.keys) and leaf.keys[idx] == key

    def __getitem__(self, key):
        leaf, idx = self._find_leaf(key)
        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            return leaf.values[idx]
        raise KeyError(key)

    def _find_leaf(self, key):
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[bisect_right(node.keys, key)]
        idx = bisect_left(node.keys, key)
        return node, idx

    def range(self, low=None, high=None):
        """Yield ``(key, value)`` with ``low <= key < high`` (half-open)."""
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0 if low is None else bisect_left(node.keys, low)]
        pos = 0 if low is None else bisect_left(node.keys, low)
        while node is not None:
            while pos < len(node.keys):
                k = node.keys[pos]
                if high is not None and k >= high:
                    return
                if low is None or k >= low:
                    yield k, node.values[pos]
                pos += 1
            node = node.next
            pos = 0

    def items(self):
        yield from self.range()

    def keys(self):
        for k, _ in self.range():
            yield k

    def values(self):
        for _, v in self.range():
            yield v

    def min_item(self):
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0]
        return (node.keys[0], node.values[0]) if node.keys else None

    def max_item(self):
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[-1]
        return (node.keys[-1], node.values[-1]) if node.keys else None

    # --------------------------------------------------------------- insert

    def insert(self, key, value):
        """Insert or replace ``key``; returns the previously stored value."""
        if key in self:
            leaf, idx = self._find_leaf(key)
            old = leaf.values[idx]
            leaf.values[idx] = value
            return old

        result = self._insert(self.root, key, value)
        self._size += 1
        if result is not None:
            sep_key, new_node = result
            root = _InternalNode()
            root.keys = [sep_key]
            root.children = [self.root, new_node]
            self.root = root
        return None

    def _insert(self, node, key, value):
        if isinstance(node, _LeafNode):
            idx = bisect_left(node.keys, key)
            node.keys.insert(idx, key)
            node.values.insert(idx, value)
            # A leaf overflows at b + 1 entries (capacity is b).
            if len(node.keys) <= self.order:
                return None
            return self._split_leaf(node)

        idx = bisect_right(node.keys, key)
        result = self._insert(node.children[idx], key, value)
        if result is None:
            return None
        sep_key, new_child = result
        node.keys.insert(idx, sep_key)
        node.children.insert(idx + 1, new_child)
        # An internal node overflows at b keys (= b + 1 children).
        if len(node.keys) < self.order:
            return None
        return self._split_internal(node)

    def _split_leaf(self, leaf):
        # b + 1 entries -> b/2 and b/2 + 1.
        mid = self.order // 2
        new = _LeafNode()
        new.keys = leaf.keys[mid:]
        new.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        new.next = leaf.next
        leaf.next = new
        return new.keys[0], new

    def _split_internal(self, node):
        # b keys (b + 1 children) -> two nodes of b/2 keys each, middle
        # separator promoted.
        mid = self.order // 2
        sep = node.keys[mid]
        new = _InternalNode()
        new.keys = node.keys[mid + 1:]
        new.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return sep, new

    # --------------------------------------------------------------- delete

    def delete(self, key):
        """Delete ``key``; return True if it existed."""
        if key not in self:
            return False
        self._delete(self.root, key)
        self._size -= 1
        if isinstance(self.root, _InternalNode) and len(self.root.children) == 1:
            self.root = self.root.children[0]
        return True

    def _delete(self, node, key):
        if isinstance(node, _LeafNode):
            idx = bisect_left(node.keys, key)
            if idx < len(node.keys) and node.keys[idx] == key:
                node.keys.pop(idx)
                node.values.pop(idx)
            return

        idx = bisect_right(node.keys, key)
        child = node.children[idx]
        self._delete(child, key)
        # The child is a non-root node (root is handled by delete());
        # rebalance it whenever it dropped below its minimum occupancy.
        if child is not self.root and \
                len(child.keys) < self._min_keys(child):
            self._rebalance(node, idx)

    def _rebalance(self, parent, idx):
        child = parent.children[idx]
        left = parent.children[idx - 1] if idx > 0 else None
        right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None

        if left is not None and len(left.keys) > self._min_keys(left):
            self._borrow_left(parent, idx, left, child)
        elif right is not None and len(right.keys) > self._min_keys(right):
            self._borrow_right(parent, idx, child, right)
        elif left is not None:
            self._merge(parent, idx - 1, left, child)
        else:
            self._merge(parent, idx, child, right)

    def _borrow_left(self, parent, idx, left, child):
        if isinstance(child, _LeafNode):
            child.keys.insert(0, left.keys.pop())
            child.values.insert(0, left.values.pop())
            parent.keys[idx - 1] = child.keys[0]
        else:
            child.keys.insert(0, parent.keys[idx - 1])
            parent.keys[idx - 1] = left.keys.pop()
            child.children.insert(0, left.children.pop())

    def _borrow_right(self, parent, idx, child, right):
        if isinstance(child, _LeafNode):
            child.keys.append(right.keys.pop(0))
            child.values.append(right.values.pop(0))
            parent.keys[idx] = right.keys[0]
        else:
            child.keys.append(parent.keys[idx])
            parent.keys[idx] = right.keys.pop(0)
            child.children.append(right.children.pop(0))

    def _merge(self, parent, sep_idx, left, right):
        if isinstance(left, _LeafNode):
            left.keys.extend(right.keys)
            left.values.extend(right.values)
            left.next = right.next
        else:
            left.keys.append(parent.keys[sep_idx])
            left.keys.extend(right.keys)
            left.children.extend(right.children)
        parent.keys.pop(sep_idx)
        parent.children.pop(sep_idx + 1)

    # ---------------------------------------------------------- persistence

    def to_dict(self):
        leaves = []
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0]
        while node is not None:
            leaves.append([list(node.keys), list(node.values)])
            node = node.next
        return {"order": self.order, "size": self._size, "leaves": leaves}

    @classmethod
    def from_dict(cls, data):
        order = data.get("order", 64)
        tree = cls(order=order)
        leaves = []
        for keys, values in data["leaves"]:
            leaf = _LeafNode()
            leaf.keys = list(keys)
            leaf.values = list(values)
            leaves.append(leaf)
        for a, b in zip(leaves, leaves[1:]):
            a.next = b
        tree._size = data.get("size", sum(len(l.keys) for l in leaves))
        if not leaves:
            return tree
        if len(leaves) == 1:
            tree.root = leaves[0]
            return tree
        # Rebuild internal levels bottom-up, at most b children per node.
        level = leaves
        cap = tree.order
        while len(level) > 1:
            parents = []
            for i in range(0, len(level), cap):
                chunk = level[i:i + cap]
                internal = _InternalNode()
                internal.children = chunk
                internal.keys = []
                for child_node in chunk[1:]:
                    n = child_node
                    while isinstance(n, _InternalNode):
                        n = n.children[0]
                    internal.keys.append(n.keys[0])
                parents.append(internal)
            level = parents
        tree.root = level[0]
        return tree
