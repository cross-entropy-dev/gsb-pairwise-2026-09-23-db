"""Tests for the B+ tree: structural invariants and dict-style semantics."""

import os
import sys
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb.storage.btree import BPlusTree, _InternalNode, _LeafNode


def validate(tree):
    """Assert B+ tree invariants; return (height, all leaf keys)."""
    order = tree.order
    min_leaf = order // 2
    min_internal = order // 2 - 1

    def recurse(node, depth, lo, hi):
        if isinstance(node, _LeafNode):
            assert node.keys == sorted(node.keys), "leaf keys not sorted"
            assert len(set(node.keys)) == len(node.keys), "duplicate keys"
            assert len(node.keys) <= order, "leaf overflow"
            if node is not tree.root:
                assert len(node.keys) >= min_leaf, "leaf underflow"
            for k in node.keys:
                assert lo is None or k >= lo
                assert hi is None or k < hi
            return depth, list(node.keys)
        assert node.keys == sorted(node.keys), "internal keys not sorted"
        assert len(node.keys) + 1 == len(node.children), "fan-out mismatch"
        assert len(node.keys) <= order - 1, "internal overflow"
        if node is not tree.root:
            assert len(node.keys) >= min_internal, "internal underflow"
        leaf_depths = []
        bounds = [lo] + list(node.keys) + [hi]
        for child, low, high in zip(node.children, bounds[:-1], bounds[1:]):
            d, _ = recurse(child, depth + 1, low, high)
            leaf_depths.append(d)
        assert len(set(leaf_depths)) == 1, "leaves at different depths"
        return leaf_depths[0], None

    depth, _ = recurse(tree.root, 0, None, None)
    chained = list(tree.keys())
    assert chained == sorted(chained), "leaf chain not globally sorted"
    return depth, chained


class BPlusTreeTests(unittest.TestCase):
    def test_insert_and_get(self):
        tree = BPlusTree(order=4)
        for i in range(100):
            tree.insert(i, i * i)
        for i in range(100):
            self.assertEqual(tree[i], i * i)
        validate(tree)

    def test_replace(self):
        tree = BPlusTree(order=4)
        tree.insert(1, "a")
        tree.insert(1, "b")
        self.assertEqual(tree[1], "b")
        self.assertEqual(len(tree), 1)

    def test_missing_key(self):
        tree = BPlusTree(order=4)
        self.assertIsNone(tree.get(1))
        with self.assertRaises(KeyError):
            _ = tree[1]

    def test_range_scan(self):
        tree = BPlusTree(order=8)
        for i in range(50):
            tree.insert(i, i)
        self.assertEqual([k for k, _ in tree.range(10, 20)],
                         list(range(10, 20)))
        self.assertEqual([k for k, _ in tree.range(high=5)],
                         list(range(0, 5)))
        self.assertEqual([k for k, _ in tree.range(45)],
                         list(range(45, 50)))

    def test_string_keys(self):
        tree = BPlusTree(order=4)
        words = ["banana", "apple", "cherry", "date", "fig"]
        for w in words:
            tree.insert(w, w.upper())
        self.assertEqual([k for k in tree.keys()], sorted(words))

    def test_delete_basic(self):
        tree = BPlusTree(order=4)
        for i in range(100):
            tree.insert(i, i)
        for i in range(0, 100, 2):
            self.assertTrue(tree.delete(i))
        self.assertFalse(tree.delete(10_000))
        remaining = list(tree.keys())
        self.assertEqual(remaining, list(range(1, 100, 2)))
        validate(tree)

    def test_delete_all(self):
        tree = BPlusTree(order=4)
        keys = list(range(200))
        for i in keys:
            tree.insert(i, i)
        random.shuffle(keys)
        for i in keys:
            self.assertTrue(tree.delete(i))
        self.assertEqual(len(tree), 0)
        self.assertEqual(list(tree.keys()), [])
        # Tree still usable after emptying.
        tree.insert(999, "x")
        self.assertEqual(tree[999], "x")

    def test_random_stress(self):
        for order in (4, 8, 16):
            tree = BPlusTree(order=order)
            ref = {}
            rng = random.Random(1234 + order)
            keys = list(range(500))
            rng.shuffle(keys)
            for k in keys:
                tree.insert(k, k)
                ref[k] = k
                if len(ref) % 50 == 0:
                    validate(tree)
            # Random deletions with periodic validation.
            rng.shuffle(keys)
            deleted = set()
            for k in keys[:300]:
                tree.delete(k)
                ref.pop(k)
                deleted.add(k)
                if len(deleted) % 40 == 0:
                    validate(tree)
            validate(tree)
            self.assertEqual(len(tree), len(ref))
            self.assertEqual(sorted(tree.keys()), sorted(ref))
            for k in ref:
                self.assertEqual(tree[k], k)

    def test_leaf_chain_and_height(self):
        tree = BPlusTree(order=16)
        n = 5000
        for i in range(n):
            tree.insert(i, i)
        depth, chained = validate(tree)
        self.assertEqual(len(chained), n)
        # log_16(5000) ~= 3.1, height is small and logarithmic.
        self.assertLessEqual(depth, 3)

    def test_persistence_roundtrip(self):
        tree = BPlusTree(order=8)
        for i in range(200):
            tree.insert(i, f"v{i}")
        for i in range(0, 200, 3):
            tree.delete(i)
        doc = tree.to_dict()
        restored = BPlusTree.from_dict(doc)
        self.assertEqual(sorted(restored.keys()), sorted(tree.keys()))
        for k in tree.keys():
            self.assertEqual(restored[k], tree[k])


if __name__ == "__main__":
    unittest.main()
