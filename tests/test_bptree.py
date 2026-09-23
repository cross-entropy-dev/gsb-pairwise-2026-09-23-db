"""B+ tree correctness tests: CRUD, ordering, splits, ranges, stress."""

import random
import unittest

from minidb.storage.bptree import BPlusTree


class TestBPlusTree(unittest.TestCase):
    def test_insert_and_get(self):
        tree = BPlusTree(order=4)
        for i in range(100):
            tree.insert(i, i * 2)
        for i in range(100):
            self.assertEqual(tree[i], i * 2)
        self.assertEqual(len(tree), 100)

    def test_update_in_place(self):
        tree = BPlusTree(order=4)
        for i in range(20):
            tree.insert(i, i)
        tree.insert(10, 999)
        self.assertEqual(tree[10], 999)
        self.assertEqual(len(tree), 20)

    def test_missing_key(self):
        tree = BPlusTree(order=4)
        with self.assertRaises(KeyError):
            tree.get(42)
        self.assertIsNone(tree.get(42, None))

    def test_delete(self):
        tree = BPlusTree(order=4)
        for i in range(50):
            tree.insert(i, str(i))
        for i in range(0, 50, 2):
            self.assertTrue(tree.delete(i))
        for i in range(50):
            still = tree.get(i, None) is not None
            self.assertEqual(still, i % 2 == 1)
        self.assertEqual(len(tree), 25)
        self.assertFalse(tree.delete(0))

    def test_range_scan_sorted(self):
        tree = BPlusTree(order=4)
        data = random.sample(range(1000), 200)
        for x in data:
            tree.insert(x, x)
        keys = [k for k, _ in tree.range(100, 500)]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(all(100 <= k < 500 for k in keys))
        self.assertEqual(set(keys), {x for x in data if 100 <= x < 500})

    def test_full_scan_order(self):
        tree = BPlusTree(order=3)
        data = random.sample(range(5000), 1000)
        for x in data:
            tree.insert(x, x)
        self.assertEqual(tree.keys_view(), sorted(data))

    def test_string_keys(self):
        tree = BPlusTree(order=5)
        words = ["banana", "apple", "cherry", "date", "elderberry"]
        for w in words:
            tree.insert(w, len(w))
        self.assertEqual(tree.keys_view(), sorted(words))
        self.assertEqual(tree.get("apple"), 5)

    def test_depth_is_logarithmic(self):
        tree = BPlusTree(order=32)
        n = 5000
        for i in range(n):
            tree.insert(i, i)
        # log_32(5000) ~= 2.46 -> depth <= 4
        self.assertLessEqual(tree.depth(), 4)

    def test_randomized_stress(self):
        random.seed(1234)
        for order in (3, 4, 8, 17):
            tree = BPlusTree(order=order)
            ref = {}
            for _ in range(2000):
                op = random.random()
                key = random.randrange(300)
                if op < 0.55:
                    tree.insert(key, key)
                    ref[key] = key
                elif op < 0.8:
                    self.assertEqual(tree.get(key, None), ref.get(key))
                else:
                    self.assertEqual(tree.delete(key), key in ref)
                    ref.pop(key, None)
            for k, v in ref.items():
                self.assertEqual(tree[k], v)
            self.assertEqual(tree.keys_view(), sorted(ref))
            self.assertEqual(len(tree), len(ref))


if __name__ == "__main__":
    unittest.main()
