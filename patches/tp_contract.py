from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

E_TP_INPUT_CONTRACT = "E_TP_INPUT_CONTRACT"


def _tp_contract_error(detail: str) -> RuntimeError:
    return RuntimeError(f"{E_TP_INPUT_CONTRACT}: {detail}")


def validate_tp_input_contract(
    *,
    req_ids: Sequence[str],
    req_id_to_index: object,
    num_computed_tokens_len: int,
    num_prompt_tokens_len: Optional[int],
    token_ids_rows: Optional[int],
    tp_size: int,
) -> Tuple[int, ...]:
    """Validate worker input contract and return req_id-aligned row indices."""
    if not isinstance(req_id_to_index, Mapping):
        raise _tp_contract_error("req_id_to_index missing or invalid")
    if tp_size > 1 and (token_ids_rows is None or token_ids_rows <= 0):
        raise _tp_contract_error("token_ids_cpu missing for tp>1")
    if num_computed_tokens_len <= 0 and req_ids:
        raise _tp_contract_error("num_computed_tokens_cpu missing or empty")

    mapped_indices = []
    for rid in req_ids:
        idx = req_id_to_index.get(rid)
        if idx is None:
            raise _tp_contract_error(f"missing row mapping for req_id={rid}")
        try:
            idx_int = int(idx)
        except Exception as exc:
            raise _tp_contract_error(f"row mapping cast failed for req_id={rid}") from exc
        if idx_int < 0 or idx_int >= num_computed_tokens_len:
            raise _tp_contract_error(
                f"row mapping out of range for req_id={rid}, "
                f"idx={idx_int}, num_computed_len={num_computed_tokens_len}"
            )
        if num_prompt_tokens_len is not None and idx_int >= num_prompt_tokens_len:
            raise _tp_contract_error(
                f"prompt mapping out of range for req_id={rid}, "
                f"idx={idx_int}, num_prompt_len={num_prompt_tokens_len}"
            )
        if tp_size > 1 and token_ids_rows is not None and idx_int >= token_ids_rows:
            raise _tp_contract_error(
                f"token row mapping out of range for req_id={rid}, "
                f"idx={idx_int}, token_ids_rows={token_ids_rows}"
            )
        mapped_indices.append(idx_int)
    return tuple(mapped_indices)


def ensure_tp_slot_by_row(
    *,
    slot_by_row: Sequence[int],
    num_reqs: int,
) -> None:
    """Pin the local invariants behind the cross-rank slot_by_row assumption.

    TP-DET relies on slot_by_row being BITWISE-identical across ranks, but no
    in-band collective may verify that (rank0+broadcast is permanently ruled
    out). This pins the per-rank sufficient conditions instead: row coverage,
    non-negative slots, and step-level injectivity. Any violation means THIS
    rank's slot ledger is corrupt (double-assignment / missing assignment --
    the exact lifecycle-corruption class that surfaces as silent cross-rank
    divergence and wrong-row GPU plans), so fail fast here.
    """
    if len(slot_by_row) < num_reqs:
        raise _tp_contract_error(
            f"slot_by_row shorter than batch: slots={len(slot_by_row)} reqs={num_reqs}"
        )
    slots = tuple(int(slot_by_row[row]) for row in range(num_reqs))
    if any(slot < 0 for slot in slots) or len(set(slots)) != num_reqs:
        raise _tp_contract_error(
            f"slot_by_row assignment corrupt (negative or duplicate slot): {slots}"
        )


def ensure_tp_prompt_lengths(
    *,
    prompt_lengths: Sequence[int],
    num_reqs: int,
) -> None:
    if len(prompt_lengths) != num_reqs:
        raise _tp_contract_error(
            f"prompt_lengths size mismatch: prompt={len(prompt_lengths)} reqs={num_reqs}"
        )
    for idx in range(num_reqs):
        if int(prompt_lengths[idx]) <= 0:
            raise _tp_contract_error(
                f"prompt_lengths must be positive for tp>1: row={idx} value={int(prompt_lengths[idx])}"
            )
