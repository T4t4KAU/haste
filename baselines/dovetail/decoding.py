"""Greedy and Dovetail-style speculative decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch


CacheLike = Any


@dataclass
class GenerationResult:
    method: str
    generated_token_ids: list[int]
    elapsed_sec: float
    tokens_per_sec: float
    prefill_sec: float = 0.0
    decode_sec: float = 0.0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_tokens == 0:
            return None
        return self.accepted_tokens / self.proposed_tokens

    @property
    def avg_accepted_tokens(self) -> float | None:
        if self.speculative_steps == 0:
            return None
        return self.accepted_tokens / self.speculative_steps


def _synchronize_if_needed(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _argmax_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def _cache_seq_length(past_key_values: CacheLike) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            key_states = first_layer[0]
            return int(key_states.shape[-2])
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def _crop_past_key_values(past_key_values: CacheLike, max_length: int) -> CacheLike:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return past_key_values

    cropped_layers = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            cropped_layers.append(layer)
            continue
        key_states, value_states, *rest = layer
        if key_states is None or value_states is None:
            cropped_layers.append(tuple(layer))
            continue
        cropped_layers.append(
            (
                key_states[..., :max_length, :].contiguous(),
                value_states[..., :max_length, :].contiguous(),
                *rest,
            )
        )
    return tuple(cropped_layers)


@torch.inference_mode()
def greedy_generate(
    model,
    *,
    input_ids: torch.Tensor,
    device: str,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> GenerationResult:
    _synchronize_if_needed(device)
    prompt_ids = input_ids.to(device)
    prefill_start = perf_counter()
    outputs = model(input_ids=prompt_ids, use_cache=True)
    _synchronize_if_needed(device)
    prefill_sec = perf_counter() - prefill_start
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    generated_token_ids: list[int] = []
    decode_sec = 0.0
    for _ in range(max_new_tokens):
        next_token = _argmax_token(next_logits)
        generated_token_ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        step_input = torch.tensor([[next_token]], device=device)
        step_start = perf_counter()
        outputs = model(
            input_ids=step_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        _synchronize_if_needed(device)
        decode_sec += perf_counter() - step_start
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    _synchronize_if_needed(device)
    elapsed_sec = prefill_sec + decode_sec
    generated_count = len(generated_token_ids)
    return GenerationResult(
        method="cpu_only",
        generated_token_ids=generated_token_ids,
        elapsed_sec=elapsed_sec,
        tokens_per_sec=(generated_count / elapsed_sec) if elapsed_sec > 0 else 0.0,
        prefill_sec=prefill_sec,
        decode_sec=decode_sec,
    )


@torch.inference_mode()
def speculative_greedy_generate(
    target_model,
    draft_model,
    *,
    input_ids: torch.Tensor,
    target_device: str,
    draft_device: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    eos_token_id: int | None,
) -> GenerationResult:
    if num_draft_tokens < 1:
        raise ValueError("num_draft_tokens must be >= 1")

    _synchronize_if_needed(target_device)
    _synchronize_if_needed(draft_device)
    prefill_start = perf_counter()
    target_inputs = input_ids.to(target_device)
    draft_inputs = input_ids.to(draft_device)
    target_outputs = target_model(input_ids=target_inputs, use_cache=True)
    target_past = target_outputs.past_key_values
    target_next_logits = target_outputs.logits[:, -1, :]

    draft_outputs = draft_model(input_ids=draft_inputs, use_cache=True)
    draft_past = draft_outputs.past_key_values
    draft_next_logits = draft_outputs.logits[:, -1, :]
    _synchronize_if_needed(target_device)
    _synchronize_if_needed(draft_device)
    prefill_sec = perf_counter() - prefill_start

    generated_token_ids: list[int] = []
    proposed_tokens = 0
    accepted_tokens = 0
    speculative_steps = 0
    decode_sec = 0.0

    while len(generated_token_ids) < max_new_tokens:
        step_start = perf_counter()
        speculative_steps += 1
        prefix_target_len = _cache_seq_length(target_past)
        prefix_draft_len = _cache_seq_length(draft_past)

        proposals: list[int] = []
        proposal_budget = min(num_draft_tokens, max_new_tokens - len(generated_token_ids))
        for _ in range(proposal_budget):
            draft_token = _argmax_token(draft_next_logits)
            proposals.append(draft_token)
            proposed_tokens += 1
            if eos_token_id is not None and draft_token == eos_token_id:
                break
            draft_step_input = torch.tensor([[draft_token]], device=draft_device)
            draft_outputs = draft_model(
                input_ids=draft_step_input,
                past_key_values=draft_past,
                use_cache=True,
            )
            draft_past = draft_outputs.past_key_values
            draft_next_logits = draft_outputs.logits[:, -1, :]

        verify_input = torch.tensor([proposals], device=target_device)
        target_outputs = target_model(
            input_ids=verify_input,
            past_key_values=target_past,
            use_cache=True,
        )
        target_past = target_outputs.past_key_values
        verify_logits = target_outputs.logits[0]

        accepted_prefix: list[int] = []
        stop_after_accept = False
        replacement_token: int | None = None

        for index, proposal in enumerate(proposals):
            step_logits = target_next_logits if index == 0 else verify_logits[index - 1].unsqueeze(0)
            target_token = _argmax_token(step_logits)
            if target_token == proposal:
                accepted_prefix.append(proposal)
                if eos_token_id is not None and proposal == eos_token_id:
                    stop_after_accept = True
                    break
                continue
            replacement_token = target_token
            break

        if len(accepted_prefix) == len(proposals) and not stop_after_accept:
            replacement_token = _argmax_token(verify_logits[len(proposals) - 1].unsqueeze(0))

        accepted_tokens += len(accepted_prefix)
        target_past = _crop_past_key_values(target_past, prefix_target_len + len(accepted_prefix))
        draft_past = _crop_past_key_values(draft_past, prefix_draft_len + len(accepted_prefix))

        generated_token_ids.extend(accepted_prefix)
        if stop_after_accept:
            break

        if replacement_token is None:
            break

        if len(generated_token_ids) >= max_new_tokens:
            break

        generated_token_ids.append(replacement_token)
        if eos_token_id is not None and replacement_token == eos_token_id:
            break

        target_step_input = torch.tensor([[replacement_token]], device=target_device)
        target_outputs = target_model(
            input_ids=target_step_input,
            past_key_values=target_past,
            use_cache=True,
        )
        target_past = target_outputs.past_key_values
        target_next_logits = target_outputs.logits[:, -1, :]

        draft_step_input = torch.tensor([[replacement_token]], device=draft_device)
        draft_outputs = draft_model(
            input_ids=draft_step_input,
            past_key_values=draft_past,
            use_cache=True,
        )
        draft_past = draft_outputs.past_key_values
        draft_next_logits = draft_outputs.logits[:, -1, :]
        _synchronize_if_needed(target_device)
        _synchronize_if_needed(draft_device)
        decode_sec += perf_counter() - step_start

    _synchronize_if_needed(target_device)
    _synchronize_if_needed(draft_device)
    elapsed_sec = prefill_sec + decode_sec
    generated_count = len(generated_token_ids)
    return GenerationResult(
        method="dovetail",
        generated_token_ids=generated_token_ids,
        elapsed_sec=elapsed_sec,
        tokens_per_sec=(generated_count / elapsed_sec) if elapsed_sec > 0 else 0.0,
        prefill_sec=prefill_sec,
        decode_sec=decode_sec,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        speculative_steps=speculative_steps,
    )
