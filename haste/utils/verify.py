"""Speculative decoding verification utilities."""

import torch

from haste.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling


def verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor | None,
    speculations: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor,
    cache_hits: torch.Tensor | None = None,
    sampler_x: float | None = None,
    async_fan_out: int | None = None,
    jit_speculate: bool = False,
) -> tuple[list[list[int]], list[int]]:
    """Verify a speculative batch against target logits.
    
    Args:
        logits_p (torch.Tensor): Target model logits
        logits_q (torch.Tensor | None): Draft model logits
        speculations (torch.Tensor): Speculated tokens
        temperatures_target (torch.Tensor): Target model temperatures
        temperatures_draft (torch.Tensor): Draft model temperatures
        cache_hits (torch.Tensor | None, optional): Cache hit indicators. Defaults to None.
        sampler_x (float | None, optional): Sampler X parameter. Defaults to None.
        async_fan_out (int | None, optional): Async fan-out parameter. Defaults to None.
        jit_speculate (bool, optional): Whether to use JIT speculation. Defaults to False.
        
    Returns:
        tuple[list[list[int]], list[int]]: Accepted suffixes and recovery tokens
    """
    del jit_speculate

    device = logits_p.device
    batch_size, kp1, _ = logits_p.shape
    lookahead = kp1 - 1

    draft_tokens = speculations[:, 1:]
    preds_p = logits_p.argmax(dim=-1)

    matches = draft_tokens == preds_p[:, :-1]
    any_mismatch = (~matches).any(dim=1)
    first_mismatch = (~matches).int().argmax(dim=1)
    accept_greedy = torch.where(
        any_mismatch,
        first_mismatch,
        torch.full_like(first_mismatch, lookahead),
    )

    batch_idx = torch.arange(batch_size, device=device)
    recovery_greedy = preds_p[batch_idx, accept_greedy]

    temps_t = temperatures_target
    temps_q = temperatures_draft
    base_ratio_rows = (temps_t > 0) | (temps_q > 0)

    if cache_hits is None:
        ratio_rows = base_ratio_rows
    else:
        ratio_rows = base_ratio_rows & cache_hits.to(torch.bool)

    do_any_ratio = bool(ratio_rows.any().item())
    need_p_probs = bool((temps_t > 0).any().item() or do_any_ratio)
    if do_any_ratio and logits_q is None:
        raise RuntimeError("Draft logits are required for probabilistic verification.")

    probs_p = None
    if need_p_probs:
        probs_p = torch.zeros(batch_size, kp1, logits_p.size(-1), device=device, dtype=torch.float32)
        nz_p = temps_t > 0
        if nz_p.any():
            scaled = logits_p[nz_p].to(torch.float32) / temps_t[nz_p].view(-1, 1, 1).clamp(min=1e-8)
            probs_p[nz_p] = torch.softmax(scaled, dim=-1)
        z_p = ~nz_p
        if z_p.any():
            argmax_p = logits_p[z_p].argmax(dim=-1)
            one_hot_p = torch.zeros_like(logits_p[z_p], dtype=torch.float32)
            one_hot_p.scatter_(2, argmax_p.unsqueeze(-1), 1.0)
            probs_p[z_p] = one_hot_p

    q_all = None
    if do_any_ratio:
        assert logits_q is not None
        q_all = torch.zeros(batch_size, lookahead, logits_q.size(-1), device=device, dtype=torch.float32)
        nz_q = temps_q > 0
        if nz_q.any():
            scaled = logits_q[nz_q].to(torch.float32) / temps_q[nz_q].view(-1, 1, 1).clamp(min=1e-8)
            q_all[nz_q] = torch.softmax(scaled, dim=-1)
        z_q = ~nz_q
        if z_q.any():
            argmax_q = logits_q[z_q].argmax(dim=-1)
            one_hot_q = torch.zeros_like(logits_q[z_q], dtype=torch.float32)
            one_hot_q.scatter_(2, argmax_q.unsqueeze(-1), 1.0)
            q_all[z_q] = one_hot_q

        if sampler_x is not None:
            assert async_fan_out is not None, "async_fan_out must be provided when sampler_x is set"
            q_all = apply_sampler_x_rescaling(q_all, sampler_x, async_fan_out)

        p_all = probs_p[:, :lookahead, :]
        gather_idx = draft_tokens.unsqueeze(-1)
        p_vals = p_all.gather(2, gather_idx).squeeze(-1)
        q_vals = q_all.gather(2, gather_idx).squeeze(-1)

        accept_probs = (p_vals / (q_vals + 1e-10)).clamp(max=1.0)
        accepts = torch.rand_like(accept_probs) <= accept_probs
        reject_any = (~accepts).any(dim=1)
        first_reject = (~accepts).int().argmax(dim=1)
        accept_ratio = torch.where(
            reject_any,
            first_reject,
            torch.full_like(first_reject, lookahead),
        )
        accept_until = torch.where(ratio_rows, accept_ratio, accept_greedy)
    else:
        accept_until = accept_greedy

    if probs_p is None:
        recovery_final = recovery_greedy
    else:
        p_fallback = probs_p[batch_idx, accept_until]
        fallback_dist = p_fallback / p_fallback.sum(dim=1, keepdim=True)

        if do_any_ratio and q_all is not None:
            q_idx_safe = accept_until.clamp(max=lookahead - 1)
            q_slice = q_all[batch_idx, q_idx_safe]
            adjust_mask = (temps_t > 0) & (accept_until < lookahead) & ratio_rows

            adjusted = (p_fallback - q_slice).clamp(min=0.0)
            adjusted_sum = adjusted.sum(dim=1, keepdim=True)
            adjusted_dist = torch.where(adjusted_sum > 0, adjusted / adjusted_sum, fallback_dist)

            recovery_adjusted = torch.multinomial(adjusted_dist, 1).squeeze(1)
            recovery_from_p = torch.multinomial(fallback_dist, 1).squeeze(1)
            recovery_sampled = torch.where(adjust_mask, recovery_adjusted, recovery_from_p)
        else:
            recovery_sampled = torch.multinomial(fallback_dist, 1).squeeze(1)

        recovery_final = torch.where(temps_t > 0, recovery_sampled, recovery_greedy)

    accepted_suffixes: list[list[int]] = []
    starts = speculations[:, 0].tolist()
    counts = accept_until.tolist()
    for batch_idx_int, accepted_count in enumerate(counts):
        suffix = [starts[batch_idx_int]] + draft_tokens[batch_idx_int, :accepted_count].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, recovery_final.tolist()
