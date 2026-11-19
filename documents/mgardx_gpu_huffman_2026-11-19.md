# MGARD-X GPU Huffman as the hpMGARD symbol coder (`xhuff`) — results, 2026-11-19

Written for: hpMGARD developers deciding how the quantised symbol streams should be entropy coded on GPUs.

## What was built

A new first-stage codec `xhuff[:L]` (StageId 4) in `CompressHpMGARDOperator/src/hpmgard/codec/MgardXHuff*`
wraps `mgard_x::ComposedLosslessCompressor<int32_t, uint64_t, SERIAL|HIP>`:

- `xhuff` = MGARD-X Huffman only; `xhuff:L` = MGARD-X `Huffman_Zstd` with zstd level L (zstd runs on the host inside
  MGARD-X); `xhuff|zstd:L` = MGARD-X Huffman followed by our own zstd stage. All are chainable like the other stages.
- The stage runs on the backend device of the compressor (`device=cpu|hip`). The bitstream is device independent:
  GPU-coded streams decode on the CPU and vice versa (verified byte-identical on MI250X, 55969 B test stream and the
  464701 B pressure field).
- Robustness: zigzag codes >= dictionary (8192, `HPMGARD_XHUFF_DICT`), streams < 65536 symbols, an MGARD-X exception,
  a HIP error, or a failed decode-and-compare after the device encode (`HPMGARD_XHUFF_VERIFY=1`, default) all fall
  back to the in-house canonical Huffman; a mode byte records which coder wrote the stream. Dictionary and block size
  (`HPMGARD_XHUFF_BLOCK`, default 20480) are stored in the payload.
- `mgardx::DeviceWorks(device)` preflights MGARD-X itself (compress + decompress of a small array, cached per process);
  the compressor and the tuner skip MGARD-X candidates on a device where it fails instead of aborting.
- Build gotchas (documented in the code): the SERIAL instantiation lives in its own OpenMP-free static target
  (`hpmgard_xhuff_serial`) because MGARD-X declares `serial_dev_id` as `#pragma omp threadprivate` while `libmgard`
  defines a plain int (TLSGD link error otherwise); `Serialize`/`Deserialize` must bracket `Compress`/`Decompress`
  for `lossless_type::Huffman`.

## Results: VKI p4 pressure, step 0, REL_L2 1e-3, fixed p1 cell-average base (file size 264.6 MB)

MI250X compute node (frontier00067, job 5950661, `scripts/xhuff_compute.sbatch`, CSV `documents/gpu/mi250x_xhuff_2026-11-19.csv`):

| codec (base and residual) | device | bytes | CR | compress s | code stage s | decompress s |
|---|---|---|---|---|---|---|
| huffman\|zstd:3 (default) | cpu 16 thr | 129,189 | 2048 | 0.411 | 0.114 | 0.473 |
| huffman\|zstd:3 (default) | hip | 129,189 | 2048 | 0.318 | 0.117 | 0.327 |
| raw\|zstd:3 | hip | 155,492 | 1702 | 0.305 | 0.108 | 0.143 |
| xhuff (MGARD-X Huffman only) | hip | 8,309,988 | 32 | 0.360 | 0.160 | 0.308 |
| xhuff:3 (MGARD-X Huffman+zstd) | hip | 464,701 | 570 | 0.520 | 0.318 | 0.299 |
| xhuff\|zstd:3 | hip | 464,599 | 570 | 0.367 | 0.168 | 0.303 |
| xhuff (SERIAL MGARD-X) | cpu | 8,309,988 | 32 | 0.691 | 0.418 | 0.337 |
| xhuff:3 (SERIAL MGARD-X) | cpu | 464,701 | 570 | 0.712 | 0.428 | 0.363 |
| MGARD-X direct, s=inf (reference) | hip | — | 131 | — | — | — |

Other tolerances on HIP: 1e-2 huffman|zstd:3 37,914 B vs xhuff:3 168,310 B; 1e-4 376,163 B vs 1,842,737 B.
All runs met the pointwise bound exactly (max err / tol = 1.000). The `xhuff` timings include the one-time HIP
runtime and kernel loading of the first call, the host-side zigzag pass and the H2D copy of the 132 MB symbol
array, the MGARD-X workspace allocation, and (for the verify runs) a GPU decode plus host compare; the
`noverify` run was not faster (0.319 s), so the verification is not the cost.

## Conclusions

1. On this data MGARD-X's Huffman does not beat the in-house `huffman|zstd:3`: 3.6x larger with zstd inside
   MGARD-X and 64x larger without it, and no faster even on the GPU. The 20480-symbol block packing of MGARD-X's
   Huffman destroys the repetition of the 32 identical z-layers that zstd exploits after our continuous bitstream
   (`raw|zstd:19` reaches 65 KB on this field for the same reason). The default codec stays `huffman|zstd:3`.
2. MGARD-X's GPU Huffman is correct on MI250X compute nodes and produces exactly the SERIAL bitstream. On the
   login node login03 (MI210, ROCm 7.2.0) the encoder is broken: histogram and sort are right, but the canonical
   codeword generation (GenerateCW) collapses all symbols into one length group (`first[l>1]` = max,
   `entry[l>1]` = symbol count), so the streams are undecodable and the code-length kernel reports 100+-bit
   codewords for wider histograms. The HIP decoder is fine (SERIAL-coded streams decode on the MI210). The fault is
   independent of the merge-path block width (32/64/512), of cooperative groups vs per-operation kernels, and of
   grid vs block barriers, and it reproduces with MGARD builds from 2026-07-23 (af7f4765) and 2026-09-08 (361bc653),
   so it is a property of that device/driver (most likely atomics on the `status` array), not a regression.
   Do not benchmark or validate MGARD-X HIP on login03; the preflight and verify guards make hpMGARD fall back there.
3. The code stage is now the largest part of the GPU compress time (0.11 s of 0.32 s). A GPU coder that helps
   must remove zero runs before entropy coding (MGARD-X `ZeroRLE_Rans` / `SymbolRans`, or an in-house GPU
   zero-run + Huffman) and take the quantised symbols directly from device memory (no host zigzag, no H2D).

Probe programs used for the diagnosis are in `analysis/mgardx_hip_probe/`.
