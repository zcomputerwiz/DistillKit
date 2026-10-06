# Assisted-by: Codex
"""CPU-only tests of bucket indexing and direct-tail reference arithmetic.

Extract the pure preparation function from its source so this test does not import
Torch, Triton or CUDA. This does not compile or execute the experimental GPU kernel.
"""
import ast
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "cce_selected.py"
parsed = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_buckets_cpu")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
prepare = namespace["prepare_buckets_cpu"]


def lse64(x):
    maximum = x.max(-1)
    anchor = np.where(np.isneginf(maximum), 0., maximum)
    total = np.exp(x.astype(np.float64) - anchor[:, None]).sum(-1)
    with np.errstate(divide="ignore"):
        return anchor + np.log(total)


def bucket_reference(logits, ids, buckets):
    rows, vocab = logits.shape
    width = buckets["block_v"]
    picked = np.full(ids.shape, np.nan, dtype=np.float32)
    omitted = logits.copy()
    visits = 0
    for tile in range((vocab + width - 1) // width):
        for row in range(rows):
            count, start = int(buckets["counts"][tile, row]), int(buckets["starts"][tile, row])
            for local in range(count):
                source = start + local
                token = buckets["sorted_ids"][row, source]
                original = buckets["sorted_order"][row, source]
                assert tile * width <= token < min((tile + 1) * width, vocab)
                picked[row, original] = logits[row, token]
                omitted[row, token] = -np.inf
                visits += 1
    return picked, omitted, visits


class BucketsTest(unittest.TestCase):
    def test_boundaries_and_original_order(self):
        ids = np.array([[400, 128, 0, 127, 256, 255], [129, 399, 1, 126, 257, 254]], dtype=np.int64)
        b = prepare(ids, 401)
        self.assertEqual(b["counts"].shape, (4, 2))
        self.assertEqual(b["counts"].dtype, np.uint8)
        np.testing.assert_array_equal(b["counts"].sum(0), [6, 6])
        for row in range(2):
            np.testing.assert_array_equal(b["sorted_ids"][row], ids[row, b["sorted_order"][row]])
        logits = np.arange(802, dtype=np.float32).reshape(2, 401)
        picked, tail, visits = bucket_reference(logits, ids, b)
        np.testing.assert_array_equal(picked, np.take_along_axis(logits, ids, 1))
        self.assertEqual(visits, ids.size)
        self.assertTrue(np.isneginf(np.take_along_axis(tail, ids, 1)).all())

    def test_random_unique_rows_and_partial_final_tile(self):
        rng = np.random.default_rng(4)
        ids = np.stack([rng.choice(501, 64, replace=False) for _ in range(7)])
        b = prepare(ids, 501)
        logits = rng.normal(size=(7, 501)).astype(np.float32)
        picked, tail, visits = bucket_reference(logits, ids, b)
        np.testing.assert_array_equal(picked, np.take_along_axis(logits, ids, 1))
        expected = logits.copy()
        np.put_along_axis(expected, ids, -np.inf, 1)
        np.testing.assert_array_equal(tail, expected)
        self.assertEqual(visits, 7 * 64)
        self.assertTrue((lse64(tail) - lse64(logits) <= 0).all())

    def test_tiny_tail_and_valid_clamp_reference(self):
        rng = np.random.default_rng(7)
        ids = np.stack([rng.choice(385, 64, replace=False) for _ in range(3)])
        logits = np.zeros((3, 385), dtype=np.float32)
        logits[np.arange(3), ids[:, 0]] = np.array([15., 25., 100.], dtype=np.float32)
        b = prepare(ids, 385)
        picked, tail, _ = bucket_reference(logits, ids, b)
        full_lse, tail_lse = lse64(logits), lse64(tail)
        direct_log_probability = tail_lse - full_lse
        self.assertTrue(np.isfinite(direct_log_probability).all())
        self.assertTrue((direct_log_probability < 0).all())
        np.testing.assert_allclose(direct_log_probability, np.log(321.) - full_lse, rtol=0, atol=1e-12)
        # The existing valid-objective clamp is separate from omitted-mass accumulation.
        upper = 1.0 - np.finfo(np.float32).eps
        clamped = np.log(np.clip(np.exp(direct_log_probability), 1 - upper, 1 - 1e-8))
        self.assertTrue(np.isfinite(clamped).all())
        np.testing.assert_array_equal(picked, np.take_along_axis(logits, ids, 1))

    def test_fully_selected_tile(self):
        ids = np.arange(128, dtype=np.int64)[None, ::-1]
        b = prepare(ids, 129)
        logits = np.arange(129, dtype=np.float32)[None, :]
        _, tail, _ = bucket_reference(logits, ids, b)
        self.assertTrue(np.isneginf(tail[0, :128]).all())
        self.assertEqual(lse64(tail)[0], 128.)

    def test_metadata_size(self):
        ids = np.tile(np.arange(64, dtype=np.int64), (4096, 1))
        b = prepare(ids, 248320)
        expected = 2 * 4096 * 1940 + 9 * 4096 * 64
        self.assertEqual(b["metadata_bytes"], expected)
        self.assertEqual(expected, 18251776)
        self.assertTrue(all(x.flags.c_contiguous for x in b.values() if isinstance(x, np.ndarray)))
        self.assertEqual(b["work"]["row_tile_pairs"], 4096)
        self.assertEqual(b["work"]["nonempty_ctas"], 32)
        self.assertEqual(b["work"]["total_ctas"], 32 * 1940)
        self.assertEqual(b["work"]["cta_local_id_iterations"], 32 * 64)

    def test_uint8_prefix_boundary(self):
        ids = np.arange(255, dtype=np.int64)[None, ::-1]
        b = prepare(ids, 257)
        np.testing.assert_array_equal(b["counts"][:, 0], [128, 127, 0])
        np.testing.assert_array_equal(b["starts"][:, 0], [0, 128, 255])
        logits = np.arange(257, dtype=np.float32)[None, :]
        picked, tail, visits = bucket_reference(logits, ids, b)
        np.testing.assert_array_equal(picked, np.take_along_axis(logits, ids, 1))
        self.assertEqual(visits, 255)
        self.assertTrue(np.isneginf(tail[0, :255]).all())

    def test_bad_ids_rejected(self):
        for ids in (np.array([[1, 1]]), np.array([[-1, 2]]), np.array([[0, 401]]),
                    np.array([[0., 1.]]), np.arange(256)[None, :], np.array([], dtype=np.int64)):
            with self.assertRaises(ValueError):
                prepare(ids, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
