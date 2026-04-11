import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from haste.layers.attention import Attention, store_kvcache
from haste.utils.context import reset_context, set_context


def _expand_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def _naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    prefix_len: int = 0,
) -> torch.Tensor:
    k = _expand_kv_heads(k, q.size(1))
    v = _expand_kv_heads(v, q.size(1))
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if causal:
        q_positions = torch.arange(q.size(0), device=q.device) + prefix_len
        k_positions = torch.arange(k.size(0), device=q.device)
        mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)


class AttentionCpuTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_store_kvcache_cpu_writes_flat_slots_for_paged_cache(self):
        key = torch.tensor(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]
        )
        value = key + 10
        k_cache = torch.zeros(2, 2, 1, 2)
        v_cache = torch.zeros(2, 2, 1, 2)
        slot_mapping = torch.tensor([0, 3, -1], dtype=torch.int32)

        store_kvcache(key, value, k_cache, v_cache, slot_mapping)

        flat_k = k_cache.view(-1, 1, 2)
        flat_v = v_cache.view(-1, 1, 2)
        self.assertTrue(torch.equal(flat_k[0], key[0]))
        self.assertTrue(torch.equal(flat_k[3], key[1]))
        self.assertTrue(torch.equal(flat_v[0], value[0]))
        self.assertTrue(torch.equal(flat_v[3], value[1]))
        self.assertTrue(torch.equal(flat_k[1], torch.zeros_like(flat_k[1])))

    def test_attention_cpu_prefill_matches_naive_varlen(self):
        scale = 0.7
        attention = Attention(num_heads=4, head_dim=2, scale=scale, num_kv_heads=2)

        q = torch.tensor(
            [
                [0.2, 0.1, 0.0, 0.3, 0.4, 0.1, 0.5, 0.2],
                [0.1, 0.3, 0.2, 0.4, 0.0, 0.2, 0.1, 0.5],
                [0.6, 0.1, 0.3, 0.2, 0.4, 0.5, 0.2, 0.1],
                [0.2, 0.6, 0.1, 0.4, 0.3, 0.0, 0.2, 0.5],
                [0.7, 0.2, 0.5, 0.1, 0.2, 0.4, 0.3, 0.6],
            ],
            dtype=torch.float32,
        )
        k = torch.tensor(
            [
                [0.1, 0.0, 0.3, 0.2],
                [0.2, 0.1, 0.4, 0.3],
                [0.3, 0.2, 0.5, 0.4],
                [0.4, 0.3, 0.6, 0.5],
                [0.5, 0.4, 0.7, 0.6],
            ],
            dtype=torch.float32,
        )
        v = torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.5],
                [0.5, 1.0, 0.0, 1.5],
                [1.5, 0.5, 1.0, 0.0],
                [0.0, 1.5, 0.5, 1.0],
                [1.0, 1.0, 1.5, 0.5],
            ],
            dtype=torch.float32,
        )

        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([], dtype=torch.int32),
        )

        output = attention(q, k, v)

        q_view = q.view(-1, 4, 2)
        k_view = k.view(-1, 2, 2)
        v_view = v.view(-1, 2, 2)
        expected = torch.cat(
            [
                _naive_attention(q_view[0:2], k_view[0:2], v_view[0:2], scale, causal=True),
                _naive_attention(q_view[2:5], k_view[2:5], v_view[2:5], scale, causal=True),
            ],
            dim=0,
        ).view(-1, 8)

        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-5))

    def test_attention_cpu_decode_reads_from_block_tables(self):
        scale = 0.5
        attention = Attention(num_heads=2, head_dim=2, scale=scale, num_kv_heads=1)
        attention.k_cache = torch.zeros(3, 2, 1, 2)
        attention.v_cache = torch.zeros(3, 2, 1, 2)

        q = torch.tensor(
            [
                [0.6, 0.2, 0.4, 0.8],
                [0.3, 0.7, 0.5, 0.1],
            ],
            dtype=torch.float32,
        )
        k = torch.tensor(
            [
                [0.5, 0.1],
                [0.2, 0.6],
            ],
            dtype=torch.float32,
        )
        v = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        flat_k_cache = attention.k_cache.view(-1, 1, 2)
        flat_v_cache = attention.v_cache.view(-1, 1, 2)
        flat_k_cache[0] = torch.tensor([[0.1, 0.0]])
        flat_k_cache[1] = torch.tensor([[0.2, 0.1]])
        flat_k_cache[4] = torch.tensor([[0.4, 0.3]])
        flat_v_cache[0] = torch.tensor([[1.0, 0.0]])
        flat_v_cache[1] = torch.tensor([[0.0, 1.0]])
        flat_v_cache[4] = torch.tensor([[1.0, 1.0]])

        block_tables = torch.tensor(
            [
                [0, 2],
                [1, -1],
            ],
            dtype=torch.int32,
        )
        slot_mapping = torch.tensor([5, 2], dtype=torch.int32)
        context_lens = torch.tensor([4, 1], dtype=torch.int32)

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        output = attention(q, k, v)

        seq0_k = torch.stack(
            [
                flat_k_cache[0],
                flat_k_cache[1],
                flat_k_cache[4],
                k.view(-1, 1, 2)[0],
            ],
            dim=0,
        ).view(-1, 1, 2)
        seq0_v = torch.stack(
            [
                flat_v_cache[0],
                flat_v_cache[1],
                flat_v_cache[4],
                v.view(-1, 1, 2)[0],
            ],
            dim=0,
        ).view(-1, 1, 2)
        seq1_k = k.view(-1, 1, 2)[1:2]
        seq1_v = v.view(-1, 1, 2)[1:2]

        q_view = q.view(-1, 2, 2)
        expected = torch.cat(
            [
                _naive_attention(q_view[0:1], seq0_k, seq0_v, scale, causal=True, prefix_len=3),
                _naive_attention(q_view[1:2], seq1_k, seq1_v, scale, causal=True, prefix_len=0),
            ],
            dim=0,
        ).view(-1, 4)

        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
