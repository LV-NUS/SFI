"""DRAFT v2 -- production-shaped wrapper for the byte-validated C++ host
derivation of ``ResolvedRowPtrArena.bind_production_row_table``.

This is the would-be ``utils/rrp_bind_host_ext.py``. It embeds the C++ source
proven byte-exact in ``cpp_bind_host_ext_v2.py`` (vs the Python reference
``cpp_ref_bind_host_derive.py`` AND vs the real arena, 300 valid + 12 raise
cases ALL_MATCH) and loads it through the project's standard extension
machinery, mirroring ``utils/fa_sparse_runtime_ext.py``:

  * single module-level ``_MODULE`` / ``_LOAD_ERROR`` cache;
  * ``load_prebuilt_extension(...)`` first, then a
    ``torch.utils.cpp_extension.load_inline(..., with_cuda=False)`` miss path;
  * env gate ``_should_enable()`` -> public ``enabled()``;
  * respects ``TORCH_EXTENSIONS_DIR`` (torch's cpp_extension honours it for the
    build dir -- this module does NOT override it).

v2 vs v1 (the REVERTED regression): the DERIVATION is identical, only the data
crossing the pybind boundary changes. v2 receives the canonical CPU block table
+ reserved_cpu + every batch-length input as CONTIGUOUS int32 BUFFERS
(``array.array('i', ...)`` + ``torch.frombuffer`` / raw ``data_ptr``), reading
``const int32_t*`` directly -- NO per-row Python callback, NO per-element pybind
marshalling. Measured 1.3x@bs2 -> 8.6x@bs256.

Public API
----------
  enabled()                      -> bool   (env VLLM_SPARSE_BIND_HOST_CPP, default ON)
  bind_host_derive_cpp(**inputs) -> dict   (marshals inputs to v2 buffers, calls C++)

The caller (the env-gated fast path in bind_production_row_table) is expected to
catch ANY exception from ``bind_host_derive_cpp`` and fall through to the inline
Python loop in bind_production_row_table, which re-runs the full derivation
byte-identically.

CANONICAL CPU ABI (v2, production)
----------------------------------
The fast path passes ``canonical_cpu=canonical_cpu_source``, where
``canonical_cpu_source`` is EITHER ``canonical_i32`` (when the canonical block
table lives on CPU) OR the ``canonical_block_table_cpu`` kwarg (otherwise), OR
None (CUDA-only, no cpu mirror). ``bind_host_derive_cpp`` marshals this into a
contiguous int32 CPU tensor and passes its ``data_ptr`` + row count + row
stride (== its actual ``stride(0)``). The C++ reads exactly ``canonical_width``
int32 per row at ``base + row*row_stride``, so the leading ``canonical_width``
columns of each actual-width row. This reproduces
``_cpu_block_table_row_source`` byte-for-byte whenever ``canonical_cpu`` is a
rank-2 CPU tensor whose own width is >= ``canonical_width`` -- including the
production case where ``canonical_cpu_source`` is the FULL
``worker_block_table_cpu`` (WIDER than ``canonical_width``); the Python
reference reads ``block_table_cpu[r][c]`` only for c < canonical_width, the
identical element.

To stay fail-safe the wrapper REFUSES (raises -> caller falls through to Python)
any ``canonical_cpu`` that is not a rank-2 CPU tensor coercible to int32, OR one
that is NARROWER than ``canonical_width`` (which would be a genuine
out-of-bound read). A wider canonical, a non-tensor sequence, a wrong
dtype/device, or a non-rank-2 tensor are handled accordingly (wider: accepted &
passed at actual stride; the rest: raise -> safe Python loop). (NOTE: the
v1-ABI ``get_canonical_cpu_row`` callback path is retained ONLY for the local
byte gate's ABI-parity probing; production never supplies it.)

PRODUCTION NOTE: this file deliberately does NOT apply the local-test
``/tmp/codex_sp`` space-fix or any ``sys.path`` insertion. That hack exists only
because the LOCAL workbench path contains spaces; the remote repo path does not,
so production imports torch normally.
"""

from __future__ import annotations

import array
import os
from typing import Optional, Sequence

import torch
from torch.utils.cpp_extension import load_inline

# In the real repo this resolves to utils/torch_extension_cache.py. When the
# prebuilt cache module is unavailable (e.g. an isolated unit test) we fall back
# to a no-op "no prebuilt" stub so the load_inline miss path still runs.
try:  # pragma: no cover - import wiring
    from utils.torch_extension_cache import load_prebuilt_extension
except Exception:  # pragma: no cover - isolated/local
    def load_prebuilt_extension(_name: str):  # type: ignore[misc]
        return None


# [DUAL-GEN-L2a-CPP-SYNC 2026-07-09] 名带 dg2：C++ 校验窗补双代 gen-half 项
# （ABI 追加 compact_gen_count）。prebuilt 缓存按名取件不看源 hash（已知坑），
# 改源必须 bump 名字，否则吃旧 .so。
_EXT_NAME = "rrp_bind_host_cpp_ext_dg2"
_ENABLE_ENV = "VLLM_SPARSE_BIND_HOST_CPP"

_MODULE: Optional[torch.nn.Module] = None
_LOAD_ERROR: Optional[Exception] = None


class RRPBindHostExtUnavailable(RuntimeError):
    """Raised when the C++ host-derivation extension cannot be loaded."""


def enabled() -> bool:
    """True iff the C++ host-derivation fast path is enabled.

    Default ON (byte-exact vs the inline Python loop in bind_production_row_table;
    e2e gate ON==OFF==frozen-baseline
    ALL_MATCH, RRP_BIND_CPP_TRACE confirms it fires). The win is batch-scaling
    (micro-bench 1.31x@bs2 -> 8.55x@bs256); at very small batch the marshalling
    + 2-pass apply makes it ~neutral/slightly slower, so set
    VLLM_SPARSE_BIND_HOST_CPP=0 to force the inline Python loop in
    bind_production_row_table if needed.
    """
    return os.environ.get(_ENABLE_ENV, "1") != "0"


# Internal alias matching the fa_sparse_runtime_ext naming.
def _should_enable() -> bool:
    return enabled()


# ===========================================================================
# Embedded C++ source -- VERBATIM from cpp_bind_host_ext_v2.py (the
# byte-validated v2 port; BUFFER ABI). Do NOT edit by hand: re-sync from
# cpp_bind_host_ext_v2.py if the reference changes, then re-run the byte gate.
# The pybind entrypoint is bind_host_derive_cpp_v2(... raw int64 data_ptr ...).
# ===========================================================================
_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Constants mirrored from the production module (verbatim values).
static const int64_t ROW_CONSUME_MODE_FULL_I32 = 0;
static const int64_t ROW_CONSUME_MODE_SELECTED_I32 = 1;
static const int64_t AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES = -1;

static const char* ROW_SOURCE_KEYS[] = {
    "compact_rows",
    "native_rows",
    "compact_reserved_pages",
    "recent_canonical_pages",
    "middle_native_canonical_pages",
    "native_canonical_pages",
    "compact_rows_with_reserved_pages",
    "compact_rows_with_recent_pages",
    "compact_full_native_fallback_rows",
};
static const int ROW_SOURCE_KEY_COUNT = 9;

static inline int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct AffinePair {
  bool present = false;
  int64_t base = 0;
  int64_t stride = 0;
};

// Verbatim mirror of _affine_base_stride_from_cpu_sequence operating over a raw
// int32 pointer + length. Returns {present=false} for the None case.
static inline AffinePair affine_base_stride_from_ptr(const int32_t* values,
                                                     int64_t len, int64_t start,
                                                     int64_t count) {
  AffinePair out;
  if (start < 0 || count <= 0 || start + count > len) {
    return out;  // None
  }
  int64_t base = static_cast<int64_t>(values[start]);
  if (count == 1) {
    out.present = true;
    out.base = base;
    out.stride = 1;
    return out;
  }
  int64_t stride = static_cast<int64_t>(values[start + 1]) - base;
  for (int64_t offset = 2; offset < count; ++offset) {
    if (static_cast<int64_t>(values[start + offset]) != base + stride * offset) {
      return out;  // None
    }
  }
  out.present = true;
  out.base = base;
  out.stride = stride;
  return out;
}

struct AffineBatchRow {
  int64_t base;
  int64_t stride;
  int64_t segment_pages;
  int64_t second_base;
  int64_t second_stride;
};

struct WriteDescriptor {
  int branch = 0;  // 0 inactive, 1 compact, 2 native
  int64_t row_start = 0;
  int64_t row_stop = 0;
  int64_t effective_k = 0;
  int64_t safe_page_id = 0;
  int64_t compact_pages = 0;
  int64_t slot_start = 0;
  int64_t slot_end = 0;
  int64_t recent_visible_pages = 0;
  int64_t recent_dst = 0;
  int64_t recent_first = 0;
  int64_t canonical_row = 0;
  int64_t visible_pages = 0;
};

// V2 entrypoint. All per-batch host int sequences + reserved_cpu + the canonical
// CPU block table arrive as RAW int32 pointers (data_ptr from contiguous int32
// tensors, validated in the Python wrapper). No Python callbacks, no per-element
// pybind marshalling on the input side.
py::object bind_host_derive_cpp_v2(
    int64_t batch_size,
    int64_t num_kv_heads,
    int64_t max_pages_per_row,
    int64_t page_size_i,
    int64_t safe_page_id_i,
    int64_t compact_capacity_i,
    int64_t canonical_width,
    int64_t canonical_rows,
    int64_t row_effk_ptr,
    int64_t compact_ready_ptr,
    int64_t slot_ptr,
    int64_t cvt_ptr,
    int64_t cot_ptr,
    int64_t rfp_ptr,
    int64_t canon_row_ptr,
    bool reserved_is_tensor,
    int64_t reserved_len,
    bool has_reserved_cpu,
    int64_t reserved_cpu_ptr,
    int64_t reserved_cpu_len,
    bool has_canonical_cpu,
    int64_t canonical_cpu_ptr,
    int64_t canonical_cpu_rows,
    int64_t canonical_cpu_row_stride,
    int64_t compact_gen_count) {
  (void)reserved_is_tensor;  // accepted for ABI parity; not used by derivation

  const int32_t* row_effk = reinterpret_cast<const int32_t*>(row_effk_ptr);
  const int32_t* compact_ready_in = reinterpret_cast<const int32_t*>(compact_ready_ptr);
  const int32_t* slot_v = reinterpret_cast<const int32_t*>(slot_ptr);
  const int32_t* cvt_v = reinterpret_cast<const int32_t*>(cvt_ptr);
  const int32_t* cot_v = reinterpret_cast<const int32_t*>(cot_ptr);
  const int32_t* rfp_v = reinterpret_cast<const int32_t*>(rfp_ptr);
  const int32_t* canon_row_v = reinterpret_cast<const int32_t*>(canon_row_ptr);
  const int32_t* reserved_cpu =
      has_reserved_cpu ? reinterpret_cast<const int32_t*>(reserved_cpu_ptr) : nullptr;
  const int32_t* canonical_cpu =
      has_canonical_cpu ? reinterpret_cast<const int32_t*>(canonical_cpu_ptr) : nullptr;

  auto canonical_row_ptr_or_null = [&](int64_t row) -> const int32_t* {
    if (!has_canonical_cpu) return nullptr;
    if (row < 0 || row >= canonical_cpu_rows) return nullptr;
    return canonical_cpu + row * canonical_cpu_row_stride;
  };

  // --- top-level geometry validation (mirrors the real prologue) ----------
  int64_t max_canonical_row = 0;
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t row = static_cast<int64_t>(canon_row_v[b]);
    if (row >= 0 && row > max_canonical_row) max_canonical_row = row;
  }
  if (canonical_width < 1) {
    throw std::invalid_argument("block_table cols must cover max_pages_per_row");
  }
  if (canonical_rows < max_canonical_row + 1) {
    throw std::invalid_argument("block_table rows must cover batch_size");
  }
  if (canonical_width > max_pages_per_row) {
    throw std::invalid_argument(
        "canonical_block_table width exceeds arena max_pages_per_row");
  }
  if (page_size_i <= 0) {
    throw std::invalid_argument("page_size must be positive");
  }
  if (safe_page_id_i < 0) {
    throw std::invalid_argument("safe_page_id must be non-negative");
  }

  bool batch_has_compact_row = false;
  for (int64_t b = 0; b < batch_size; ++b) {
    if (compact_ready_in[b] != 0) {
      batch_has_compact_row = true;
      break;
    }
  }

  int64_t total_rows = batch_size * num_kv_heads;
  std::vector<int64_t> coverage;
  coverage.reserve(total_rows);
  std::vector<char> compact_ready_out;
  compact_ready_out.reserve(total_rows);
  std::vector<char> row_mode_codes;
  row_mode_codes.reserve(total_rows);

  std::vector<AffineBatchRow> affine_rows_by_batch;
  affine_rows_by_batch.reserve(batch_size);
  std::vector<int64_t> affine_row_modes_by_batch;
  affine_row_modes_by_batch.reserve(batch_size);
  std::vector<int64_t> effective_k_seq;
  effective_k_seq.reserve(batch_size);
  std::vector<WriteDescriptor> write_descriptors;
  write_descriptors.reserve(batch_size);

  int64_t rs_compact_rows = 0;
  int64_t rs_native_rows = 0;
  int64_t rs_compact_reserved_pages = 0;
  int64_t rs_recent_canonical_pages = 0;
  int64_t rs_middle_native_canonical_pages = 0;
  int64_t rs_native_canonical_pages = 0;
  int64_t rs_compact_rows_with_reserved_pages = 0;
  int64_t rs_compact_rows_with_recent_pages = 0;
  int64_t rs_compact_full_native_fallback_rows = 0;
  bool inactive_seen = false;
  int64_t rs_inactive_rows = 0;

  auto append_affine = [&](int64_t row_mode, int64_t base, int64_t stride,
                           int64_t segment_pages, int64_t second_base,
                           int64_t second_stride) {
    affine_row_modes_by_batch.push_back(row_mode);
    AffineBatchRow r;
    r.base = base;
    r.stride = stride;
    r.segment_pages = segment_pages;
    r.second_base = second_base;
    r.second_stride = second_stride;
    affine_rows_by_batch.push_back(r);
  };
  auto append_affine_default = [&](int64_t row_mode) {
    append_affine(row_mode, 0, 1, AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES, 0, 1);
  };

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    int64_t effective_k = static_cast<int64_t>(row_effk[batch]);
    if (effective_k < 0) effective_k = 0;  // max(0, int(...))
    bool is_compact = compact_ready_in[batch] != 0;
    int64_t canonical_row = static_cast<int64_t>(canon_row_v[batch]);
    effective_k_seq.push_back(effective_k);

    int64_t row_start = batch * num_kv_heads;
    int64_t row_stop = row_start + num_kv_heads;

    // ---------------- INACTIVE branch (canonical_row < 0) ---------------
    if (canonical_row < 0) {
      if (effective_k > page_size_i || is_compact) {
        throw std::invalid_argument(
            "inactive canonical rows must be non-compact with at most one safe page");
      }
      inactive_seen = true;
      rs_inactive_rows += num_kv_heads;
      int64_t safe_pages = (effective_k > 0) ? 1 : 0;
      for (int64_t h = 0; h < num_kv_heads; ++h) {
        coverage.push_back(safe_pages);
        compact_ready_out.push_back(0);
        row_mode_codes.push_back(0);  // unset
      }
      append_affine_default(ROW_CONSUME_MODE_FULL_I32);
      WriteDescriptor wd;
      wd.branch = 0;
      wd.row_start = row_start;
      wd.row_stop = row_stop;
      wd.effective_k = effective_k;
      wd.safe_page_id = safe_page_id_i;
      write_descriptors.push_back(wd);
      continue;
    }

    // ---------------- COMPACT branch ------------------------------------
    if (is_compact) {
      int64_t compact_tokens = static_cast<int64_t>(cvt_v[batch]);
      if (compact_tokens < 0) compact_tokens = 0;
      if (compact_tokens % page_size_i != 0) {
        throw std::invalid_argument("compact_valid_tokens must be page-aligned");
      }
      int64_t compact_offset_tokens_i = static_cast<int64_t>(cot_v[batch]);
      if (compact_offset_tokens_i < 0) {
        throw std::invalid_argument("compact_offset_tokens must be non-negative");
      }
      if (compact_offset_tokens_i % page_size_i != 0) {
        throw std::invalid_argument("compact_offset_tokens must be page-aligned");
      }
      int64_t compact_pages = compact_tokens / page_size_i;
      int64_t compact_offset_pages = compact_offset_tokens_i / page_size_i;
      if (compact_pages > compact_capacity_i) {
        throw std::invalid_argument(
            "compact page count exceeds compact_capacity_pages");
      }
      if (effective_k < compact_tokens) {
        throw std::invalid_argument(
            "row_effective_k is smaller than compact tokens");
      }
      int64_t recent_visible_tokens = effective_k - compact_tokens;
      int64_t recent_visible_pages =
          (recent_visible_tokens > 0) ? ceil_div(recent_visible_tokens, page_size_i)
                                      : 0;
      int64_t visible_pages = compact_pages + recent_visible_pages;
      if (visible_pages > max_pages_per_row) {
        throw std::invalid_argument(
            "visible compact/recent pages exceed row width");
      }
      int64_t slot = static_cast<int64_t>(slot_v[batch]);
      // [DUAL-GEN-L2a-CPP-SYNC 2026-07-09] mirror of the Python loop's
      // gen-half window selection (resolved_row_ptr_arena [DUAL-GEN-L2a]):
      // the offset picks the generation half, then the slot window is
      // validated inside that half. gen_count==1 reduces to the historical
      // single-gen expression bit-identically. This term was missing after
      // dual-gen landed (Python loop updated, embedded C++ not re-synced) ->
      // gen-B rows were wrongly rejected as "exceeds slot compact span".
      int64_t gen_stride_pages =
          compact_gen_count > 0 ? reserved_len / compact_gen_count : 0;
      int64_t gen_of_offset =
          (compact_gen_count > 1 && compact_offset_pages >= gen_stride_pages)
              ? 1
              : 0;
      int64_t expected_slot_start =
          gen_of_offset * gen_stride_pages + slot * compact_capacity_i;
      int64_t expected_slot_end = expected_slot_start + compact_capacity_i;
      if (slot < 0 || expected_slot_start < 0 || expected_slot_end > reserved_len) {
        throw std::invalid_argument("slot exceeds reserved_manager_block_ids");
      }
      int64_t slot_start = compact_offset_pages;
      int64_t slot_end = slot_start + compact_pages;
      if (slot_start < expected_slot_start || slot_end > expected_slot_end) {
        throw std::invalid_argument(
            "compact offset plus page count exceeds slot compact span");
      }
      if (slot_start < 0 || slot_end > reserved_len) {
        throw std::invalid_argument(
            "compact offset plus page count exceeds reserved_manager_block_ids");
      }
      int64_t recent_first = static_cast<int64_t>(rfp_v[batch]);
      if (recent_first < 0 ||
          recent_first + recent_visible_pages > canonical_width) {
        throw std::invalid_argument(
            "recent page range exceeds canonical block table");
      }

      rs_compact_rows += num_kv_heads;
      rs_compact_reserved_pages += compact_pages * num_kv_heads;
      rs_recent_canonical_pages += recent_visible_pages * num_kv_heads;
      if (compact_pages > 0) {
        rs_compact_rows_with_reserved_pages += num_kv_heads;
      }
      if (recent_visible_pages > 0) {
        rs_compact_rows_with_recent_pages += num_kv_heads;
      }
      if (compact_pages == 0 && recent_visible_pages > 0) {
        rs_compact_full_native_fallback_rows += num_kv_heads;
      }

      const int32_t* canonical_cpu_row = canonical_row_ptr_or_null(canonical_row);
      bool canonical_cpu_is_none = (canonical_cpu_row == nullptr);

      bool selected_present = false;
      AffineBatchRow selected{};
      if (compact_pages > 0 && has_reserved_cpu) {
        AffinePair compact_affine = affine_base_stride_from_ptr(
            reserved_cpu, reserved_cpu_len, slot_start, compact_pages);
        if (compact_affine.present) {
          AffinePair recent_affine;
          if (recent_visible_pages > 0) {
            if (!canonical_cpu_is_none) {
              recent_affine = affine_base_stride_from_ptr(
                  canonical_cpu_row, canonical_width, recent_first,
                  recent_visible_pages);
            } else {
              recent_affine.present = false;  // None
            }
          } else {
            recent_affine.present = true;
            recent_affine.base =
                compact_affine.base + compact_affine.stride * (compact_pages - 1);
            recent_affine.stride = 1;
          }
          if (recent_affine.present) {
            selected_present = true;
            selected.base = compact_affine.base;
            selected.stride = compact_affine.stride;
            selected.segment_pages = compact_pages;
            selected.second_base = recent_affine.base;
            selected.second_stride = recent_affine.stride;
          }
        }
      }

      for (int64_t h = 0; h < num_kv_heads; ++h) {
        coverage.push_back(visible_pages);
        compact_ready_out.push_back(1);
        row_mode_codes.push_back(1);  // compact
      }
      if (!selected_present) {
        append_affine_default(ROW_CONSUME_MODE_FULL_I32);
      } else {
        append_affine(ROW_CONSUME_MODE_SELECTED_I32, selected.base, selected.stride,
                      selected.segment_pages, selected.second_base,
                      selected.second_stride);
      }
      WriteDescriptor wd;
      wd.branch = 1;
      wd.row_start = row_start;
      wd.row_stop = row_stop;
      wd.effective_k = effective_k;
      wd.compact_pages = compact_pages;
      wd.slot_start = slot_start;
      wd.slot_end = slot_end;
      wd.recent_visible_pages = recent_visible_pages;
      wd.recent_dst = compact_pages;
      wd.recent_first = recent_first;
      wd.canonical_row = canonical_row;
      write_descriptors.push_back(wd);
      continue;
    }

    // ---------------- NATIVE branch -------------------------------------
    int64_t native_ceil = (effective_k > 0) ? ceil_div(effective_k, page_size_i) : 0;
    int64_t visible_pages = native_ceil;
    if (canonical_width < visible_pages) visible_pages = canonical_width;
    if (max_pages_per_row < visible_pages) visible_pages = max_pages_per_row;

    rs_native_rows += num_kv_heads;
    rs_native_canonical_pages += visible_pages * num_kv_heads;
    for (int64_t h = 0; h < num_kv_heads; ++h) {
      coverage.push_back(visible_pages);
      compact_ready_out.push_back(0);
      row_mode_codes.push_back(2);  // native
    }
    {
      WriteDescriptor wd;
      wd.branch = 2;
      wd.row_start = row_start;
      wd.row_stop = row_stop;
      wd.effective_k = effective_k;
      wd.visible_pages = visible_pages;
      wd.canonical_row = canonical_row;
      write_descriptors.push_back(wd);
    }
    if (batch_has_compact_row) {
      append_affine_default(ROW_CONSUME_MODE_FULL_I32);
    } else {
      const int32_t* canonical_cpu_row = canonical_row_ptr_or_null(canonical_row);
      bool canonical_cpu_is_none = (canonical_cpu_row == nullptr);
      bool native_present = false;
      AffinePair native_affine;
      if (!canonical_cpu_is_none && visible_pages > 0) {
        native_affine = affine_base_stride_from_ptr(canonical_cpu_row,
                                                    canonical_width, 0, visible_pages);
        native_present = native_affine.present;
      }
      if (!native_present) {
        append_affine_default(ROW_CONSUME_MODE_FULL_I32);
      } else {
        int64_t native_base = native_affine.base;
        int64_t native_stride = native_affine.stride;
        int64_t last = (visible_pages - 1 > 0) ? (visible_pages - 1) : 0;
        append_affine(ROW_CONSUME_MODE_FULL_I32, native_base, native_stride,
                      visible_pages, native_base + native_stride * last, 1);
      }
    }
  }

  // --- scalar collapse (_prove_scalar_batch_affine) -----------------------
  bool scalar_present = false;
  int64_t scalar_base0 = 0, scalar_stride0 = 0, scalar_batch_stride = 0;
  int64_t n_aff = static_cast<int64_t>(affine_rows_by_batch.size());
  do {
    if (batch_size < 1 || n_aff != batch_size) break;  // fa3_sm90_bs1_affine: was <2
    if (static_cast<int64_t>(affine_row_modes_by_batch.size()) != batch_size) break;
    int64_t base0 = affine_rows_by_batch[0].base;
    int64_t stride0 = affine_rows_by_batch[0].stride;
    std::vector<int64_t> bases;
    bases.reserve(batch_size);
    bool ok = true;
    for (int64_t b_i = 0; b_i < batch_size; ++b_i) {
      const AffineBatchRow& r = affine_rows_by_batch[b_i];
      int64_t b = r.base, st = r.stride, seg = r.segment_pages, sb = r.second_base,
              ss = r.second_stride;
      if (affine_row_modes_by_batch[b_i] != ROW_CONSUME_MODE_SELECTED_I32) {
        ok = false;
        break;
      }
      if (seg == AFFINE_ROW_PTR_FALLBACK_SEGMENT_PAGES || seg <= 0) {
        ok = false;
        break;
      }
      if (st != stride0) {
        ok = false;
        break;
      }
      if (ss != st || sb != b + st * seg) {
        ok = false;
        break;
      }
      bases.push_back(b);
    }
    if (!ok) break;
    if (batch_size == 1) {  // fa3_sm90_bs1_affine: single uniform-affine batch
      scalar_present = true; scalar_base0 = base0; scalar_stride0 = stride0;
      scalar_batch_stride = 0; break;
    }
    int64_t batch_stride = bases[1] - bases[0];
    for (int64_t b_i = 1; b_i < batch_size; ++b_i) {
      if (bases[b_i] - bases[b_i - 1] != batch_stride) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
    scalar_present = true;
    scalar_base0 = base0;
    scalar_stride0 = stride0;
    scalar_batch_stride = batch_stride;
  } while (false);

  // --- build the Python return dict --------------------------------------
  py::list coverage_py;
  for (int64_t v : coverage) coverage_py.append(v);
  py::list compact_ready_py;
  py::object py_true = py::reinterpret_borrow<py::object>(Py_True);
  py::object py_false = py::reinterpret_borrow<py::object>(Py_False);
  for (char v : compact_ready_out) {
    compact_ready_py.append(v ? py_true : py_false);
  }
  py::str s_unset("unset");
  py::str s_compact("compact");
  py::str s_native("native");
  py::list row_modes_py;
  for (char code : row_mode_codes) {
    if (code == 0) {
      row_modes_py.append(s_unset);
    } else if (code == 1) {
      row_modes_py.append(s_compact);
    } else {
      row_modes_py.append(s_native);
    }
  }

  py::dict row_sources_py;
  int64_t schema_vals[ROW_SOURCE_KEY_COUNT] = {
      rs_compact_rows,
      rs_native_rows,
      rs_compact_reserved_pages,
      rs_recent_canonical_pages,
      rs_middle_native_canonical_pages,
      rs_native_canonical_pages,
      rs_compact_rows_with_reserved_pages,
      rs_compact_rows_with_recent_pages,
      rs_compact_full_native_fallback_rows,
  };
  for (int i = 0; i < ROW_SOURCE_KEY_COUNT; ++i) {
    row_sources_py[py::str(ROW_SOURCE_KEYS[i])] = py::int_(schema_vals[i]);
  }
  if (inactive_seen) {
    row_sources_py[py::str("inactive_rows")] = py::int_(rs_inactive_rows);
  }

  py::list affine_rows_py;
  for (const AffineBatchRow& r : affine_rows_by_batch) {
    affine_rows_py.append(py::make_tuple(r.base, r.stride, r.segment_pages,
                                         r.second_base, r.second_stride));
  }
  py::list affine_modes_py;
  for (int64_t m : affine_row_modes_by_batch) affine_modes_py.append(m);

  py::object scalar_py;
  if (scalar_present) {
    scalar_py = py::make_tuple(scalar_base0, scalar_stride0, scalar_batch_stride);
  } else {
    scalar_py = py::none();
  }

  py::list effective_k_py;
  for (int64_t v : effective_k_seq) effective_k_py.append(v);

  py::str k_branch("branch");
  py::str k_row_start("row_start");
  py::str k_row_stop("row_stop");
  py::str k_effective_k("effective_k");
  py::str k_safe_page_id("safe_page_id");
  py::str k_compact_pages("compact_pages");
  py::str k_slot_start("slot_start");
  py::str k_slot_end("slot_end");
  py::str k_recent_visible_pages("recent_visible_pages");
  py::str k_recent_dst("recent_dst");
  py::str k_recent_first("recent_first");
  py::str k_canonical_row("canonical_row");
  py::str k_visible_pages("visible_pages");
  py::str v_inactive("inactive");
  py::str v_compact("compact");
  py::str v_native("native");

  py::list write_desc_py;
  for (const WriteDescriptor& wd : write_descriptors) {
    py::dict d;
    if (wd.branch == 0) {
      d[k_branch] = v_inactive;
      d[k_row_start] = py::int_(wd.row_start);
      d[k_row_stop] = py::int_(wd.row_stop);
      d[k_effective_k] = py::int_(wd.effective_k);
      d[k_safe_page_id] = py::int_(wd.safe_page_id);
    } else if (wd.branch == 1) {
      d[k_branch] = v_compact;
      d[k_row_start] = py::int_(wd.row_start);
      d[k_row_stop] = py::int_(wd.row_stop);
      d[k_effective_k] = py::int_(wd.effective_k);
      d[k_compact_pages] = py::int_(wd.compact_pages);
      d[k_slot_start] = py::int_(wd.slot_start);
      d[k_slot_end] = py::int_(wd.slot_end);
      d[k_recent_visible_pages] = py::int_(wd.recent_visible_pages);
      d[k_recent_dst] = py::int_(wd.recent_dst);
      d[k_recent_first] = py::int_(wd.recent_first);
      d[k_canonical_row] = py::int_(wd.canonical_row);
    } else {
      d[k_branch] = v_native;
      d[k_row_start] = py::int_(wd.row_start);
      d[k_row_stop] = py::int_(wd.row_stop);
      d[k_effective_k] = py::int_(wd.effective_k);
      d[k_visible_pages] = py::int_(wd.visible_pages);
      d[k_canonical_row] = py::int_(wd.canonical_row);
    }
    write_desc_py.append(d);
  }

  py::dict out;
  out["coverage"] = coverage_py;
  out["compact_ready"] = compact_ready_py;
  out["row_modes"] = row_modes_py;
  out["row_sources"] = row_sources_py;
  out["affine_rows_by_batch"] = affine_rows_py;
  out["affine_row_modes_by_batch"] = affine_modes_py;
  out["scalar_collapse"] = scalar_py;
  out["effective_k_seq"] = effective_k_py;
  out["write_descriptors"] = write_desc_py;
  return out;
}
"""


def _load_ext(*, force: bool = False) -> Optional[torch.nn.Module]:
    """Three-stage load mirroring fa_sparse_runtime_ext._load_ext.

    1. return the cached module if already built;
    2. return None if a prior build failed (cached _LOAD_ERROR);
    3. (unless force) honour the env gate;
    4. try the prebuilt cache (load_prebuilt_extension);
    5. fall back to load_inline(with_cuda=False).
    """
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        return None
    if not force and not _should_enable():
        return None
    if not force:
        prebuilt = load_prebuilt_extension(_EXT_NAME)
        # ABI guard: only trust a prebuilt module that actually exposes the v2
        # BUFFER-ABI entrypoint. A stale/name-collided .so (a documented hazard
        # for this box's shared TORCH_EXTENSIONS_DIR) would otherwise be cached
        # and either raise (-> silent Python fallback) or, worse, drift; on
        # mismatch fall through to load_inline so the correct source is built.
        if prebuilt is not None and hasattr(prebuilt, "bind_host_derive_cpp_v2"):
            _MODULE = prebuilt
            return _MODULE
    try:
        _MODULE = load_inline(
            name=_EXT_NAME,
            cpp_sources=[_CPP_SOURCE],
            functions=["bind_host_derive_cpp_v2"],
            with_cuda=False,
            extra_cflags=["-std=c++17", "-O3"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR = exc
        return None
    return _MODULE


def _require_ext(*, force: bool = False) -> torch.nn.Module:
    mod = _load_ext(force=force)
    if mod is None:
        if _LOAD_ERROR is not None:
            raise RRPBindHostExtUnavailable(
                "rrp_bind_host_cpp_ext unavailable"
            ) from _LOAD_ERROR
        raise RRPBindHostExtUnavailable(
            "rrp_bind_host_cpp_ext unavailable; set "
            f"{_ENABLE_ENV}=1 to enable the C++ host-derivation fast path"
        )
    return mod


# ===========================================================================
# Buffer marshalling (v2 BUFFER ABI). Each batch-length sequence + reserved_cpu
# is materialized ONCE into a contiguous int32 CPU tensor via
# array.array('i', ...) + torch.frombuffer (single C-level fill, zero-copy
# view). The canonical CPU block table is marshalled to a contiguous int32 2-D
# CPU tensor. The wrapper keeps every tensor ALIVE while the C++ reads its
# data_ptr (they are appended to a keep-alive list passed only inside the call).
# ===========================================================================
def _as_i32_buffer(seq) -> torch.Tensor:
    """Materialize a Python int sequence as a contiguous 1-D int32 CPU tensor.

    array('i') is the C int (int32 on every platform torch supports here);
    frombuffer yields a contiguous int32 tensor sharing the array's buffer. The
    array object lives inside the tensor's storage (frombuffer keeps the buffer
    referenced) so the data stays valid while C++ holds the data_ptr.
    """
    buf = array.array("i", seq)
    if len(buf) == 0:
        return torch.empty(0, dtype=torch.int32)
    return torch.frombuffer(buf, dtype=torch.int32)


def _marshal_canonical_cpu(canonical_cpu: object, canonical_width: int):
    """Marshal the canonical CPU block table into a contiguous int32 2-D tensor.

    Returns (has_canonical_cpu, tensor_or_None). When the source cannot be
    safely marshalled into a rank-2 CPU int32 tensor whose width is
    AT LEAST canonical_width, RAISES ValueError so the caller falls through to
    the safe Python loop. None input => (False, None) (the CUDA-only / no-cpu
    case -- matches _cpu_block_table_row_source returning None for every row).

    Width check: the C++ reads exactly ``canonical_width`` int32 per row,
    starting at ``base + row * canonical_cpu_row_stride`` where the passed
    row stride == this tensor's ``stride(0)`` (== its actual width once
    contiguous). In production ``canonical_cpu_source`` is the FULL
    ``worker_block_table_cpu``, whose width is >= canonical_width (the Python
    reference ``_cpu_block_table_row_source`` reads ``block_table_cpu[r][c]``
    only for c < canonical_width). So a WIDER table is the normal case and is
    byte-safe: the C++ reads the same leading ``canonical_width`` columns of
    each actual-width row, i.e. the identical element ``block_table_cpu[r][c]``.
    We therefore only REFUSE a NARROWER table (width < canonical_width), which
    would be a genuine out-of-bound read.
    """
    if canonical_cpu is None:
        return False, None
    c = canonical_cpu
    if not isinstance(c, torch.Tensor):
        # _cpu_block_table_row_source accepts non-tensor sequences-of-rows, but
        # production canonical_cpu_source is always a tensor or None. Refuse the
        # exotic sequence path (-> Python fallback) rather than risk a divergent
        # marshal.
        raise ValueError("canonical_cpu must be a torch.Tensor or None")
    if c.device.type != "cpu":
        raise ValueError("canonical_cpu must be a CPU tensor")
    if c.dim() != 2:
        raise ValueError("canonical_cpu must be rank 2")
    if int(c.shape[1]) < int(canonical_width):
        raise ValueError("canonical_cpu width must be at least canonical_width")
    if c.dtype != torch.int32:
        c = c.to(torch.int32)
    # .contiguous() makes stride(0) == actual width; the caller passes that
    # actual stride to the C++ so each row base lands on block_table_cpu[r][0].
    c = c.contiguous()
    return True, c


def _build_buffers(kwargs: dict):
    """Marshal all v2 inputs into buffers. Returns a (positional_args, keepalive)
    tuple ready to splat into the C++ entrypoint. Raises (ValueError) on any
    input that cannot be safely marshalled or that violates the length
    contract; the caller ([BIND-CPP-FAILFAST 2026-07-09]) fails fast on it."""
    bufs: list = []  # keep-alive: tensors backing every data_ptr passed to C++

    # [RRP-BIND-ARRAYLEN 2026-07-11 EXT审计·仅卫生] C++ 侧对下面七个数组一律
    # 按 batch ∈ [0, batch_size) 索引（v2 源: canon_row_v[b]/compact_ready_in
    # [batch]/row_effk[batch]/cvt_v[batch]/cot_v[batch]/slot_v[batch]/
    # rfp_v[batch]），但 ABI 只传 data_ptr 不传长度——短数组 = C++ 越界读。
    # 在封送点做显式长度合同（ValueError 与 C++ 几何合同异常同型）。
    expected_len = int(kwargs["batch_size"])

    def buf(seq, name: str) -> int:
        t = _as_i32_buffer(seq)
        # 合同取 >=（覆盖 C++ 读域即安全）：短数组必拒；容量式载体（若有
        # padding 尾）不误伤。
        if int(t.numel()) < expected_len:
            raise ValueError(
                f"rrp_bind_host_ext: {name} has {int(t.numel())} elements, "
                f"shorter than batch_size={expected_len} (C++ reads "
                f"[0, batch_size) => OOB)"
            )
        bufs.append(t)
        return t.data_ptr()

    row_effk_ptr = buf(kwargs["row_effective_k_by_row"], "row_effective_k_by_row")
    # compact_ready as int32 (0/1); generator avoids an intermediate list.
    compact_ready_ptr = buf(
        (1 if v else 0 for v in kwargs["compact_ready_by_batch"]),
        "compact_ready_by_batch",
    )
    slot_ptr = buf(kwargs["slot_by_row"], "slot_by_row")
    cvt_ptr = buf(kwargs["compact_valid_tokens_by_row"], "compact_valid_tokens_by_row")
    cot_ptr = buf(kwargs["compact_offset_tokens_by_row"], "compact_offset_tokens_by_row")
    rfp_ptr = buf(kwargs["recent_first_page_by_row"], "recent_first_page_by_row")
    canon_row_ptr = buf(kwargs["canonical_row_by_batch"], "canonical_row_by_batch")

    reserved_cpu_values = kwargs["reserved_cpu_values"]
    if reserved_cpu_values is None:
        has_reserved_cpu = False
        reserved_cpu_ptr = 0
        reserved_cpu_len = 0
    else:
        has_reserved_cpu = True
        rt = _as_i32_buffer(reserved_cpu_values)
        bufs.append(rt)
        reserved_cpu_ptr = rt.data_ptr()
        reserved_cpu_len = int(rt.numel())

    canonical_width = int(kwargs["canonical_width"])
    # v2 production: prefer the explicit canonical_cpu tensor (canonical_i32 on
    # CPU, or canonical_block_table_cpu, or None). For ABI parity with the local
    # byte gate, recover a tensor from get_canonical_cpu_row if only that is given.
    canonical_cpu = kwargs.get("canonical_cpu", None)
    if canonical_cpu is None and kwargs.get("get_canonical_cpu_row", None) is not None:
        canonical_cpu = _canonical_tensor_from_callable(
            kwargs["get_canonical_cpu_row"], canonical_width
        )
    has_canonical_cpu, canonical_cpu_t = _marshal_canonical_cpu(
        canonical_cpu, canonical_width
    )
    if not has_canonical_cpu:
        canonical_cpu_ptr = 0
        canonical_cpu_rows = 0
        canonical_cpu_row_stride = 0
    else:
        bufs.append(canonical_cpu_t)
        canonical_cpu_ptr = canonical_cpu_t.data_ptr()
        canonical_cpu_rows = int(canonical_cpu_t.shape[0])
        canonical_cpu_row_stride = int(canonical_cpu_t.stride(0))

    args = (
        int(kwargs["batch_size"]),
        int(kwargs["num_kv_heads"]),
        int(kwargs["max_pages_per_row"]),
        int(kwargs["page_size_i"]),
        int(kwargs["safe_page_id_i"]),
        int(kwargs["compact_capacity_i"]),
        canonical_width,
        int(kwargs["canonical_rows"]),
        row_effk_ptr,
        compact_ready_ptr,
        slot_ptr,
        cvt_ptr,
        cot_ptr,
        rfp_ptr,
        canon_row_ptr,
        bool(kwargs["reserved_is_tensor"]),
        int(kwargs["reserved_len"]),
        has_reserved_cpu,
        reserved_cpu_ptr,
        reserved_cpu_len,
        has_canonical_cpu,
        canonical_cpu_ptr,
        canonical_cpu_rows,
        canonical_cpu_row_stride,
        int(kwargs["compact_gen_count"]),
    )
    return args, bufs


def _canonical_tensor_from_callable(get_row, width: int):
    """Recover a CPU int32 (rows x width) tensor from a get_canonical_cpu_row
    callable (v1-ABI / local-gate compatibility). Returns None if row 0 has no
    cpu source. Production never uses this path (it passes canonical_cpu)."""
    if get_row is None:
        return None
    rows = []
    i = 0
    while True:
        r = get_row(i)
        if r is None:
            break
        if isinstance(r, torch.Tensor):
            rows.append(r.to(torch.int32).cpu().contiguous())
        else:
            rows.append(torch.as_tensor(list(r), dtype=torch.int32))
        i += 1
        if i > 1_000_000:  # safety
            break
    if not rows:
        return None
    return torch.stack(rows, dim=0).contiguous()


def bind_host_derive_cpp(**kwargs) -> dict:
    """Keyword wrapper around the compiled v2 C++ entrypoint (BUFFER ABI).

    Accepts the SAME keyword arguments as the inline Python loop in
    ``bind_production_row_table`` / ``cpp_ref_bind_host_derive.bind_host_derive``,
    PLUS an optional
    ``canonical_cpu`` (the CPU canonical block-table tensor; production passes
    ``canonical_cpu_source`` here). The legacy ``get_canonical_cpu_row`` callback
    is accepted for local-gate ABI parity only.

    Marshals all batch-length sequences + reserved_cpu + the canonical CPU block
    table into contiguous int32 buffers ONCE, then passes raw data_ptrs. Raises
    ``RRPBindHostExtUnavailable`` if the ext cannot be loaded, or ``ValueError``
    if an input cannot be safely marshalled. [BIND-CPP-FAILFAST 2026-07-09]
    调用方（bind_production_row_table 快路径）对任何异常 fail-fast——不再静默
    落穿 Python loop；显式逃生=VLLM_SPARSE_BIND_HOST_CPP=0。
    """
    ext = _require_ext()
    args, _keepalive = _build_buffers(kwargs)
    # _keepalive must outlive the call: the C++ reads the data_ptrs synchronously
    # within this call, then returns a fully materialized Python dict (no tensor
    # aliasing into the output), so it is safe to drop _keepalive on return.
    # [BIND-CPP-VALUEERROR-CONTRACT 2026-07-09] 嵌入 C++ 的全部 16 个 throw 均为
    # std::invalid_argument（输入合同校验，与 Python loop 的 ValueError 同一合同），
    # 但 torch cpp_extension 的异常翻译把它们统一成 RuntimeError——在此译回
    # ValueError，保证两臂对同一坏输入抛同型异常（几何拒绝合同测试跨臂同判）。
    # 翻译是完备的：C++ 源无其它 throw 种类（re-sync 时须保持该不变量）。
    try:
        _rrp_cpp_result = ext.bind_host_derive_cpp_v2(*args)
    except RuntimeError as _cpp_contract_exc:
        _message = str(_cpp_contract_exc)
        # 仅失败分支补足跨来源几何取证。成功热路径不物化 tuple、不重算
        # span，也不增加 extension ABI/编译世代；若 row-mode 与 launch-plan
        # 来自不同快照，报错会直接给出 offending row 的实际/期望窗口。
        if "compact offset plus page count exceeds slot compact span" in _message:
            _message += "; " + _compact_span_error_context(kwargs)
        raise ValueError(_message) from _cpp_contract_exc
    # Optional firing trace (default OFF, byte-neutral): when RRP_BIND_CPP_TRACE
    # is set, append one line per SUCCESSFUL C++ derive so a gate can prove the
    # fast path actually fired in the real CUDA-graph e2e (the ext loads
    # verbose=False, so driver.log carries no load marker).
    if os.environ.get("RRP_BIND_CPP_TRACE"):
        try:
            with open("/tmp/rrp_cpp_fired.cnt", "a") as _trace_f:
                _trace_f.write("1\n")
        except Exception:
            pass
    return _rrp_cpp_result


def _compact_span_error_context(kwargs: dict) -> str:
    """Build exact compact-span diagnostics after the C++ contract rejected.

    This helper is intentionally reachable only from the exception path above;
    keeping it out of ``_build_buffers`` preserves the successful bind cost.
    """

    def _sequence(name: str) -> tuple:
        value = kwargs.get(name, ())
        try:
            return tuple(value)
        except TypeError:
            return tuple()

    batch_size = int(kwargs.get("batch_size", 0))
    page_size = int(kwargs.get("page_size_i", 0))
    capacity_pages = int(kwargs.get("compact_capacity_i", 0))
    reserved_pages = int(kwargs.get("reserved_len", 0))
    gen_count = int(kwargs.get("compact_gen_count", 0))
    gen_stride_pages = reserved_pages // gen_count if gen_count > 0 else 0
    ready = _sequence("compact_ready_by_batch")
    slots = _sequence("slot_by_row")
    valid_tokens = _sequence("compact_valid_tokens_by_row")
    offsets = _sequence("compact_offset_tokens_by_row")
    effective_k = _sequence("row_effective_k_by_row")
    recent_first = _sequence("recent_first_page_by_row")
    rows = []
    for row in range(min(batch_size, len(ready))):
        if not bool(ready[row]):
            continue
        slot = int(slots[row]) if row < len(slots) else None
        valid = int(valid_tokens[row]) if row < len(valid_tokens) else None
        offset = int(offsets[row]) if row < len(offsets) else None
        offset_pages = (
            offset // page_size
            if offset is not None and page_size > 0
            else None
        )
        offset_gen = (
            int(gen_count > 1 and offset_pages >= gen_stride_pages)
            if offset_pages is not None and gen_stride_pages > 0
            else 0
        )
        expected_start = (
            offset_gen * gen_stride_pages + slot * capacity_pages
            if slot is not None
            else None
        )
        rows.append(
            {
                "row": row,
                "slot": slot,
                "valid_tokens": valid,
                "offset_tokens": offset,
                "offset_pages": offset_pages,
                "offset_gen": offset_gen,
                "expected_start_page": expected_start,
                "expected_end_page": (
                    expected_start + capacity_pages
                    if expected_start is not None
                    else None
                ),
                "effective_k": (
                    int(effective_k[row]) if row < len(effective_k) else None
                ),
                "recent_first_page": (
                    int(recent_first[row]) if row < len(recent_first) else None
                ),
            }
        )
    return (
        "rrp_compact_geometry="
        f"{{'page_size': {page_size}, 'capacity_pages': {capacity_pages}, "
        f"'reserved_pages': {reserved_pages}, 'gen_count': {gen_count}, "
        f"'gen_stride_pages': {gen_stride_pages}, 'rows': {rows!r}}}"
    )


def trace_cpp_fallthrough(exc=None):
    """Byte-neutral observability sibling of the RRP_BIND_CPP_TRACE success counter.
    When RRP_BIND_CPP_TRACE is set, append one line per FALLTHROUGH (the C++ fast
    path raised and the arena fell back to the Python loop) so a silent perma-fallback
    is visible alongside /tmp/rrp_cpp_fired.cnt. Records the exception type+message
    (no traceback -> no new import). Never raises; default-OFF = no-op."""
    if not os.environ.get("RRP_BIND_CPP_TRACE"):
        return
    try:
        if exc is None:
            _detail = "fallthrough"
        else:
            _detail = type(exc).__name__ + ": " + str(exc).replace("\n", " ")
        with open("/tmp/rrp_cpp_fallthrough.cnt", "a") as _trace_f:
            _trace_f.write(_detail + "\n")
    except Exception:
        pass
