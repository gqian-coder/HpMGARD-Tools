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

## Addendum, 12:05: device-resident MGARD-X stages (job 5951454, MI250X frontier00119, warm-up excluded)

Residual codec varied, base codec fixed `huffman|zstd:3`, p4 pressure step 0, REL_L2 1e-3, 16 host threads.
`verify` = decode-and-compare after the device encode (now opt-in, `HPMGARD_XHUFF_VERIFY=1`).
CSV: `documents/gpu/mi250x_xdev_2026-11-19.csv`.

| residual codec | device | bytes | CR | compress s | code stage s | decompress s (cpu) |
|---|---|---|---|---|---|---|
| huffman\|zstd:3 (in-house, host) | hip | 129,189 | 2048 | 0.179 | 0.114 | 0.331 |
| raw\|zstd:3 | hip | 157,566 | 1680 | 0.099 | 0.037 | 0.146 |
| xhuff (GPU Huffman), no verify | hip | 8,197,060 | 32 | 0.091 | 0.028 | 0.351 |
| xhuff\|zstd:3 (GPU Huffman + host zstd), no verify | hip | 438,348 | 604 | 0.097 | 0.035 | 0.334 |
| xhuff\|zstd:3, with verify | hip | 438,348 | 604 | 0.208 | 0.143 | 0.334 |
| xhuff:3 (zstd inside MGARD-X), with verify | hip | 438,186 | 604 | 0.211 | 0.146 | 0.323 |
| xrle (ZeroRLE_Rans), no verify | hip | 7,567,201 | 35 | 0.075 | 0.012 | 0.593 |
| xrle\|zstd:3 | hip | 3,252,661 | 81 | 0.120 | 0.056 | 0.597 |
| xrans (SymbolRans), no verify | hip | 6,669,007 | 40 | 0.075 | 0.011 | 0.524 |
| xrans\|zstd:3 | hip | 2,331,402 | 114 | 0.107 | 0.044 | 0.523 |
| xhuff (SERIAL MGARD-X) | cpu | 8,197,060 | 32 | 0.625 | 0.532 | 0.163 |
| xrle / xrans (SERIAL MGARD-X) | cpu | 7.6 MB / 6.7 MB | 35 / 40 | 0.87 / 0.90 | 0.78 / 0.81 | 0.40 / 0.36 |

Other tolerances on hip: 1e-2 `huffman|zstd:3` 37,914 B vs `xhuff|zstd:3` 147,759 B (code 0.109 vs 0.120 s with
verify); 1e-4 376,163 B vs 1,008,849 B (0.124 vs 0.118 s with verify). All runs met the bound exactly.

What changed versus the morning numbers: the GPU Huffman stage now codes the device-resident symbols
(no host zigzag, no 132 MB host-to-device copy, persistent workspace, warm-up excluded), and its real cost
is 28 ms for 33M symbols; with the host zstd behind it the code stage takes 35 ms against 114 ms for the
in-house Huffman + zstd, and the whole GPU compress drops from 0.179 s to 0.097 s. The decode-and-compare
check cost 0.11 s and dominated the earlier measurements; it is therefore off by default and the device
preflight (`mgardx::DeviceWorks`) is the guard.

Size: on this step-0 field the MGARD-X Huffman stream stays 3.4x larger after zstd (438 KB vs 129 KB)
because MGARD-X packs 20480-symbol chunks with per-chunk padding, which breaks the byte-exact repetition
of the 32 identical z-layers that zstd otherwise matches; `raw|zstd:3` (158 KB) shows how much of this
field's redundancy is that repetition rather than low symbol entropy (the Huffman streams carry about 2
bits per symbol). The rANS types without zstd land at CR 35-40 like plain Huffman and their CPU decode is
slower; they remain available as options. Whether the size gap persists on a developed-flow field, which
has no exact layer repetition, must be measured on such data before the GPU default is chosen.

## Addendum, 13:00: GPU path with the auto codec, no host round trips, pinned scratch, sweep/prolong kernels (job 5951809, MI250X frontier00062)

Changes since the previous addendum (operator commits 7691dc2, cdad282, 8508848): `xhuff|zstd:3` is the
default codec on a GPU where MGARD-X works (`auto`); the residual symbols never leave the device on compress
(coded in place by MGARD-X, the host copy is fetched only on a fallback); on decompress MGARD-X decodes on
the device and the reconstruction kernel reads the symbols there; host scratch is pinned and persists in a
process-wide backend instance; restriction uses register-resident tensor sweeps and prolongation a
thread-per-output kernel with on-the-fly Kronecker weights; the MGARD-X dictionary adapts to the stream's
code range (256 to 8192, recorded in the payload). Warm-up excluded, 16 host threads, p4 pressure step 0.
CSV: `documents/gpu/mi250x_gpu_path_2026-11-19.csv`.

| tol | run | bytes | CR | compress s | refactor | base | residual | code | decompress s |
|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | hip, auto (xhuff\|zstd:3), decode on hip | 418,749 | 632 | 0.087 | 0.009 | 0.025 | 0.003 | 0.038 | 0.183 |
| 1e-3 | hip, auto, decode on cpu | 418,749 | 632 | 0.089 | | | | | 0.300 |
| 1e-3 | hip, huffman\|zstd:3 (in-house), decode on cpu | 129,189 | 2048 | 0.175 | 0.009 | 0.020 | 0.008 | 0.115 | 0.358 |
| 1e-3 | cpu, auto (= huffman\|zstd:3) | 129,189 | 2048 | 0.366 | 0.007 | 0.021 | 0.116 | 0.111 | 0.392 |
| 1e-3 | cpu compress, decode on hip | 129,189 | 2048 | 0.270 | | | | | 0.257 |
| 1e-2 | hip, auto, decode on hip | 111,978 | 2363 | 0.079 | 0.009 | 0.023 | 0.003 | 0.030 | 0.159 |
| 1e-2 | hip, in-house | 37,914 | 6980 | 0.168 | | | | 0.108 | 0.344 |
| 1e-4 | hip, auto, decode on hip | 993,734 | 266 | 0.092 | 0.009 | 0.019 | 0.003 | 0.043 | 0.135 |
| 1e-4 | hip, in-house | 376,163 | 704 | 0.187 | | | | 0.127 | 0.390 |
| 1e-3 | hip, tuned (auto list) -> raw\|zstd:3 | 155,492 | 1702 | 0.628 (0.470 tuning) | | | | 0.102 | 0.178 |

All runs met the pointwise bound exactly; the GPU decode of every symbol-stream case equals the CPU decode
bit for bit (test_compressor, 85 checks, including xhuff/xrle/xrans and the dedup base); only MGARD-X's own
double streams (base codec `mgardx`, direct) differ between its SERIAL and HIP inverse transforms and are
checked against the bound instead.

Reading: the whole GPU compress of a 264.6 MB field now takes 0.087 s (the morning's resident-component
version took 0.318 s and the in-house chain on the same GPU 0.175 s); the residual stage dropped from
0.024 s with the per-cell sweep to 0.003 s with the thread-per-output prolongation, and the symbol coder
to 0.038 s of which about 0.03 s is the host zstd behind MGARD-X's Huffman. GPU decompress is 0.18 s against
0.30 s for the CPU decode of the same stream and 0.39 s for the CPU-only path. The remaining size gap to the
in-house chain on this field (419 KB vs 129 KB) is the layer-repetition effect discussed above; on a
developed-flow field it has to be re-measured before the GPU default is final. MGARD-X HIP Huffman streams
remain byte-identical to the SERIAL ones; `xrle` streams differ in bytes between devices but decode on both.
