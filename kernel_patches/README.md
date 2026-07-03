# SFI Kernel Patches

SFI's CUDA fast path is implemented as a patch set on top of
[vllm-project/flash-attention](https://github.com/vllm-project/flash-attention).

| | |
|:--|:--|
| **Upstream base commit** | `f5bc33cfc02c744d24a2e9d50e6db656de40611c` |
| **FA3 patch** | `sfi_fa3_sm80_sm90.patch` — mixed-page / compact-KV forward, dual-source paged KV, resolved-row-ptr routing, score capture; targets **SM80** (Ampere) and **SM90** (Hopper) |
| **FA4 patch** | `sfi_fa4_sm100_cute.patch` — the same compact-KV design ported to the FA4 CuTe-DSL kernels, plus the shared dispatch-layer validation/guards in `vllm_flash_attn/flash_attn_interface.py`; targets **SM100** (Blackwell); applies **on top of** the FA3 patch |

Both patches were generated with `git diff --binary` against the base commit
and verified to apply cleanly (`git apply --check`) in that exact order:

```bash
git clone https://github.com/vllm-project/flash-attention.git
cd flash-attention
git checkout f5bc33cfc02c744d24a2e9d50e6db656de40611c
git submodule update --init csrc/cutlass
git apply /path/to/SFI/kernel_patches/sfi_fa3_sm80_sm90.patch
git apply /path/to/SFI/kernel_patches/sfi_fa4_sm100_cute.patch   # SM100 only
```

`scripts/setup_flash_attention.sh` automates exactly this plus the build.

## What gets built, per architecture

| Architecture | Kernel | Build | Runtime artifact |
|:--|:--|:--|:--|
| SM80 / SM90 | FA3 (C++/CUTLASS) | `python setup.py build_ext --inplace` (~25–40 min) | `vllm_flash_attn/_vllm_fa3_C*.so`, imported at runtime |
| SM100 | FA4 (CuTe DSL, Python) | none ahead of time | JIT-compiled by `cute.compile` on first use, disk-cached |

SFI never modifies your installed vLLM or flash-attention packages on disk:
the patched clone is loaded through `VLLM_SPARSE_FA3_UPSTREAM_ROOT` /
`--fa3-upstream-root` and bridged into vLLM by `patches/fa3_native/install.py`
at process start (runtime patching only).
