import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from haste.layers.rotary_embedding import get_rope
from haste.models.qwen3 import _extract_rope_scaling


class RotaryEmbeddingTest(unittest.TestCase):
    def test_get_rope_accepts_linear_scaling_dict(self):
        rope = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=16,
            base=1000000,
            rope_scaling={"type": "linear", "factor": 2.0},
        )

        positions = torch.arange(6, dtype=torch.long)
        query = torch.randn(6, 16)
        key = torch.randn(6, 16)

        out_q, out_k = rope(positions, query, key)

        self.assertEqual(out_q.shape, query.shape)
        self.assertEqual(out_k.shape, key.shape)

    def test_get_rope_expands_cache_for_dynamic_scaling(self):
        rope = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=4,
            base=1000000,
            rope_scaling={
                "rope_type": "dynamic",
                "factor": 4.0,
                "original_max_position_embeddings": 4,
            },
        )

        positions = torch.tensor([0, 1, 7], dtype=torch.long)
        query = torch.randn(3, 16)
        key = torch.randn(3, 16)

        out_q, out_k = rope(positions, query, key)

        self.assertEqual(out_q.shape, query.shape)
        self.assertEqual(out_k.shape, key.shape)
        self.assertGreaterEqual(rope.max_seq_len_cached, 8)

    def test_extract_rope_scaling_reads_rope_parameters(self):
        class DummyConfig:
            rope_parameters = {"rope_type": "linear", "factor": 2.0}
            rope_scaling = None
            rope_theta = 1000000
            max_position_embeddings = 32768

        rope_scaling = _extract_rope_scaling(DummyConfig())

        self.assertIsNotNone(rope_scaling)
        self.assertEqual(rope_scaling["rope_type"], "linear")
        self.assertEqual(rope_scaling["factor"], 2.0)
        self.assertEqual(rope_scaling["rope_theta"], 1000000)


if __name__ == "__main__":
    unittest.main()
