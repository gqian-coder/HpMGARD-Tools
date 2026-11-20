# MGARD-X GPU Huffman as the hpMGARD symbol coder (`xhuff`) — results, 2026-11-19

Written for: hpMGARD developers deciding how the quantised symbol streams should be entropy coded on GPUs.

## Current state (2026-11-20 10:50)

This memo is a dated log. The sections after this one are kept as written, with bracketed notes where a later
round replaced a statement; where anything below disagrees with this section, this section is current.

Defaults (operator 1eb2d10):
- `auto` codec: `xhuff|zstd:3` on a GPU where MGARD-X passes its preflight, `huffman|zstd:3` otherwise. The GPU
  default stays on MGARD-X's coder because that coder is built and tested across devices; the final choice
  waits for developed-flow data.
- Tuner lists: GPU `xhuff|zstd:3,frbp|zstd:3,huffman|zstd:3,raw|zstd:3`, CPU `huffman|zstd:3,raw|zstd:3,frbp|zstd:3`.
  The ADIOS2 plugin tunes once per variable by default (`tuning=once`, also `auto`; `always` and `fixed` are
  the alternatives).
- MGARD-X stages: dictionary chosen per stream between 256 and 8192 entries, up to 1% out-of-dictionary codes
  stored as MGARD-X outliers, Huffman block 4096, decode-and-compare check opt-in (`HPMGARD_XHUFF_VERIFY=1`).
- zstd: multithreaded from 16 MB inputs (`HPMGARD_ZSTD_MT_MIN_MB`), up to 8 workers.

GPU path: per-component statistics on the device; one fused kernel per cell batch for restriction, base split,
quantisation, decoded base, prolongation and residual; FRBP and MGARD-X stages on device-resident symbols, with
zstd on host threads overlapping the later streams; the caller's interleaved block uploaded once per call
through pinned staging and the result downloaded once into the caller's buffer (`DecompressInto`, used by the
ADIOS2 operator). Every run meets the bound; GPU FRBP streams are byte-identical to the CPU ones; the CPU and
HIP test suites pass.

Latest numbers (MI250X, job 5955222, p4 pressure step 0, REL_L2 1e-3, no tuning, warm-up excluded, decode on
the same device):

| chain | bytes | compress s | GB/s | decompress s | GB/s |
|---|---|---|---|---|---|
| frbp\|zstd:3 on the GPU, median of 4 runs | 304,247 | 0.029 | 9.1 | 0.0205 | 12.9 |
| xhuff\|zstd:3 on the GPU (auto default) | 455,262 | 0.043 | 6.2 | 0.051 | 5.2 |
| huffman\|zstd:3, in-house coder on the host, rest on the GPU | 129,189 | 0.136 | 1.9 | 0.238 | 1.1 |
| huffman\|zstd:3, CPU only | 129,189 | 0.274 | 0.97 | 0.291 | 0.91 |

The 3-component velocity (794 MB) and the 8-block p5 pressure (457 MB) compress at 9.8 and 10.2 GB/s with
`frbp|zstd:3` (03:00 addendum). Where the time goes: the two link copies are about 60 percent of a GPU
compression and 85 percent of a decompression; the device work is about 6 to 7 ms per 265 MB field (about
40 GB/s); the host zstd of the last residual stream adds from under 1 ms to about 10 ms depending on the tolerance.

Ratio against MGARD-X (gate v2, geometric mean over six non-constant fields, 1e-2 / 1e-3 / 1e-4):
`xhuff|zstd:3` is 7.1x / 4.2x / 6.6x smaller than MGARD-X s=inf and the in-house chain 20.8x / 12.1x / 15.9x.
`frbp|zstd:3` has not been through the gate; on the pressure field it is 2.4x larger than the in-house chain
and 1.5x smaller than `xhuff|zstd:3`.

Caveats: all data is step 0, whose exact layer repetition favours zstd after a continuous bitstream; MGARD-X's
HIP Huffman encoder is broken on the login03 MI210 and the preflight falls back there; the CUDA build is
mapped but not compiled (no nvcc on Frontier); most timing rows are single runs.

Open items: developed-flow data for the GPU default; `frbp|zstd:3` across the gate fields; a repeat-run job for
medians; reconstruct-and-download per cell range on decompression; per-component upload so the fused kernel of
one component overlaps the upload of the next; multithreaded zstd for FRBP streams under 16 MB; one-pass
statistics for multi-component variables; folding the statistics into the fused kernel in ABS mode; the CUDA
build on an NVIDIA machine.

## What was built

A new first-stage codec `xhuff[:L]` (StageId 4) in `CompressHpMGARDOperator/src/hpmgard/codec/MgardXHuff*`
wraps `mgard_x::ComposedLosslessCompressor<int32_t, uint64_t, SERIAL|HIP>`:

- `xhuff` = MGARD-X Huffman only; `xhuff:L` = MGARD-X `Huffman_Zstd` with zstd level L (zstd runs on the host inside
  MGARD-X); `xhuff|zstd:L` = MGARD-X Huffman followed by our own zstd stage. All are chainable like the other stages.
- The stage runs on the backend device of the compressor (`device=cpu|hip`). The bitstream is device independent:
  GPU-coded streams decode on the CPU and vice versa (verified byte-identical on MI250X, 55969 B test stream and the
  464701 B pressure field).
- Robustness: zigzag codes >= dictionary (8192, `HPMGARD_XHUFF_DICT`) [now up to 1% kept as MGARD-X outliers, see 01:10], streams < 65536 symbols, an MGARD-X exception,
  a HIP error, or a failed decode-and-compare after the device encode (`HPMGARD_XHUFF_VERIFY=1`, default) [now opt-in, see 12:05] all fall
  back to the in-house canonical Huffman; a mode byte records which coder wrote the stream. Dictionary and block size
  (`HPMGARD_XHUFF_BLOCK`, default 20480) [now 4096, see 01:10] are stored in the payload.
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
   (`raw|zstd:19` reaches 65 KB on this field for the same reason). The default codec stays `huffman|zstd:3`. [On the GPU the default became
   `xhuff|zstd:3` at 13:00; see Current state.]
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
   [The symbols are coded from device memory since 12:05. The zero-run coders stay options, not defaults,
   because the step-0 sparsity is not representative of a developed flow.]

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

## Addendum, 23:10: gate sweep on an MI250X, hpMGARD (GPU, tuned) vs MGARD-X HIP (job 5954493, frontier07079)

Same protocol as the CPU gate of the morning, now entirely on the GPU: hpMGARD tuned with the auto codec list
(`xhuff|zstd:3,huffman|zstd:3,raw|zstd:3`, dedup geometry from the mesh dumps), device-resident path,
decode on the GPU; MGARD-X HIP 1-D on the same absolute pointwise bound (s=inf; s=0 shown for reference).
VKI step 0, p4 all fields and the 8-block p5 pressure. uu, vv, ww, uv, vw and uw are constant at this
step (one Const stream, CR 1.3e6, error 0) and are excluded from the geometric means below. All hpMGARD
runs met the bound (max err / tol = 1.000 or 0.500); MGARD-X s=inf lands at 0.74 to 0.80 of the bound.
CSV: `documents/gpu/mi250x_gate_hip_2026-11-19.csv`; job script `scripts/gate_hip_compute.sbatch`.

1e-2: geomean CR ratio hpMGARD/MGARD-X(s=inf) = 19.5x over 6 non-constant fields; mean compress hpMGARD tuned 0.85 s (incl. tuning) vs MGARD-X HIP 2.04 s; mean hpMGARD GPU decode 0.33 s
1e-3: geomean CR ratio hpMGARD/MGARD-X(s=inf) = 11.7x over 6 non-constant fields; mean compress hpMGARD tuned 0.85 s (incl. tuning) vs MGARD-X HIP 2.08 s; mean hpMGARD GPU decode 0.30 s
1e-4: geomean CR ratio hpMGARD/MGARD-X(s=inf) = 24.9x over 6 non-constant fields; mean compress hpMGARD tuned 1.01 s (incl. tuning) vs MGARD-X HIP 2.24 s; mean hpMGARD GPU decode 0.43 s

| field | tol | hpMGARD CR | chosen codecs | MGARD-X s=inf CR | MGARD-X s=0 CR | ratio (s=inf) | hpMGARD compress s (tuned) | MGARD-X HIP s |
|---|---|---|---|---|---|---|---|---|
| p4_Density_aver | 1e-2 | 8118 | huffman\|zstd:3/huffman\|zstd:3 | 250.6 | 350.0 | 32.4 | 0.66 | 1.65 |
| p4_Pressure_aver | 1e-2 | 6980 | huffman\|zstd:3/huffman\|zstd:3 | 242.4 | 324.6 | 28.8 | 0.66 | 1.65 |
| p4_Rho_ms | 1e-2 | 6469 | huffman\|zstd:3/huffman\|zstd:3 | 209.9 | 288.1 | 30.8 | 0.66 | 1.65 |
| p4_P_ms | 1e-2 | 2550 | raw\|zstd:3/raw\|zstd:3 | 202.3 | 270.4 | 12.6 | 0.59 | 1.65 |
| p4_Velocity_aver | 1e-2 | 1200 | raw\|zstd:3/raw\|zstd:3 | 149.9 | 192.6 | 8.0 | 1.73 | 1.91 |
| p5_Pressure_aver | 1e-2 | 9468 | huffman\|zstd:3/huffman\|zstd:3 | 506.7 | 673.3 | 18.7 | 0.80 | 3.77 |
| p4_Density_aver | 1e-3 | 2195 | huffman\|zstd:3/huffman\|zstd:3 | 136.2 | 167.9 | 16.1 | 0.73 | 1.67 |
| p4_Pressure_aver | 1e-3 | 1702 | raw\|zstd:3/raw\|zstd:3 | 130.6 | 159.4 | 13.0 | 0.60 | 1.64 |
| p4_Rho_ms | 1e-3 | 1531 | raw\|zstd:3/raw\|zstd:3 | 115.4 | 141.3 | 13.3 | 0.60 | 1.68 |
| p4_P_ms | 1e-3 | 1285 | raw\|zstd:3/raw\|zstd:3 | 111.2 | 136.2 | 11.6 | 0.61 | 1.66 |
| p4_Velocity_aver | 1e-3 | 591 | raw\|zstd:3/raw\|zstd:3 | 77.9 | 96.7 | 7.6 | 1.71 | 2.02 |
| p5_Pressure_aver | 1e-3 | 2842 | huffman\|zstd:3/huffman\|zstd:3 | 277.0 | 328.0 | 10.3 | 0.83 | 3.79 |
| p4_Density_aver | 1e-4 | 1045 | raw\|zstd:3/raw\|zstd:3 | 34.0 | 37.4 | 30.8 | 0.83 | 1.81 |
| p4_Pressure_aver | 1e-4 | 1024 | raw\|zstd:3/raw\|zstd:3 | 33.8 | 36.7 | 30.3 | 0.84 | 1.80 |
| p4_Rho_ms | 1e-4 | 946 | raw\|zstd:3/raw\|zstd:3 | 30.3 | 34.9 | 31.2 | 0.82 | 1.82 |
| p4_P_ms | 1e-4 | 745 | raw\|zstd:3/raw\|zstd:3 | 30.4 | 34.8 | 24.5 | 0.84 | 1.83 |
| p4_Velocity_aver | 1e-4 | 412 | raw\|zstd:3/raw\|zstd:3 | 42.0 | 49.1 | 9.8 | 1.66 | 2.17 |
| p5_Pressure_aver | 1e-4 | 1392 | raw\|zstd:3/raw\|zstd:3 | 41.0 | 46.1 | 33.9 | 1.05 | 3.99 |

Compared with the CPU gate (12.5x to 22.6x), the GPU gate is 11.7x to 25x in ratio, the same picture: the
differences come from the tuner's per-field choices (it picked `raw|zstd:3` or `huffman|zstd:3` on these
repetition-heavy fields, never `xhuff|zstd:3`, which is why the auto default does not hurt the ratio). In
time, hpMGARD including its tuning pass (about 0.47 s of each run) is 2x to 5x faster than MGARD-X HIP on the
same GPU; with a fixed configuration (`p4_Pressure_inhouse_*`: 0.16 to 0.18 s) it is about 10x faster.
hpMGARD GPU decode takes 0.1 to 0.7 s depending on the chosen base split. Caveat as before: step-0 data.

## Addendum, 2026-11-20 00:05: gate v2 on an MI250X: fixed `xhuff|zstd:3` vs fixed `huffman|zstd:3` vs tuned, MGARD-X s=inf and s=0 (job 5954583, frontier00187)

Data: `ho_dg_p4_sol` step 0 (Density, Pressure, Rho_ms, P_ms, 3-component Velocity) and `ho_dg_p5_8np_sol`
pressure (8 blocks); the six Reynolds-stress fields are constant at this step and are left out. Every row
ran on the GPU with warm-up, 16 host threads, decode on the GPU, MGARD-X HIP 1-D at the same absolute bound.
Geometric means over the six fields; throughput = raw bytes / compress time.
CSV: `documents/gpu/mi250x_gate2_2026-11-19.csv`; script `scripts/gate_hip2_compute.sbatch`.

| tol | hpMGARD variant | geomean CR | CR vs MGARD-X s=inf | CR vs MGARD-X s=0 | compress s | GB/s | GPU decode s | MGARD-X HIP compress s |
|---|---|---|---|---|---|---|---|---|
| 1e-2 | xhuff\|zstd:3 | 1706 | 7.1x | 5.3x | 0.126 | 3.22 | 0.270 | 1.99 |
| 1e-2 | huffman\|zstd:3 | 5023 | 20.8x | 15.5x | 0.246 | 1.59 | 0.426 | 2.06 |
| 1e-2 | tuned | 5070 | 21.0x | 15.7x | 0.846 | 0.45 | 0.355 | 2.06 |
| 1e-3 | xhuff\|zstd:3 | 550 | 4.2x | 3.5x | 0.169 | 2.90 | 0.342 | 2.02 |
| 1e-3 | huffman\|zstd:3 | 1577 | 12.1x | 9.9x | 0.255 | 1.54 | 0.461 | 2.09 |
| 1e-3 | tuned | 1466 | 11.2x | 9.2x | 0.825 | 0.46 | 0.307 | 2.07 |
| 1e-4 | xhuff\|zstd:3 | 231 | 6.6x | 5.9x | 0.171 | 2.93 | 0.303 | 2.17 |
| 1e-4 | huffman\|zstd:3 | 554 | 15.9x | 14.0x | 0.273 | 1.45 | 0.510 | 2.25 |
| 1e-4 | tuned | 825 | 23.6x | 20.9x | 0.982 | 0.37 | 0.461 | 2.24 |

Reading:
- Throughput: the GPU `xhuff|zstd:3` path compresses the p4 scalars at 3.0 to 3.2 GB/s (0.083 to 0.089 s per
  265 MB field) and the 8-block p5 pressure at 3.8 to 4.0 GB/s; the in-house chain on the same GPU reaches
  1.5 to 1.6 GB/s, MGARD-X HIP 0.13 to 0.17 GB/s. The tuner adds about 0.47 s (0.35 to 0.56 GB/s overall).
- Ratio: on this field set `xhuff|zstd:3` is 2.8 to 3.5x larger than the in-house chain (the layer-repetition
  effect) but still 4.2x to 7.1x smaller than MGARD-X s=inf. The velocity rows at 1e-3 and 1e-4 fell back to
  the in-house Huffman for one component because its alphabet exceeded the 8192-entry dictionary cap
  (compress 0.54 s instead of about 0.25 s); the cap is now 65536 so the GPU path stays on the device.
  [At 01:10 the cap went back to 8192 with outliers, which keeps velocity on the device without the slow
  65536-entry codebook.] On the CPU build that component codes to 1.20 MB with MGARD-X Huffman versus 0.43 MB
  with the in-house coder.
- L2 comparison: MGARD-X s=0 at the same nominal tolerance reaches an RMSE close to hpMGARD's (pressure 1e-3:
  24.6 vs 24.5; velocity 0.0299 vs 0.0322; density 0.000178 vs 0.000308) while exceeding the pointwise bound
  by 1.45 to 1.72x; its CR is only 1.1 to 1.4x above s=inf. At that roughly equal RMS error hpMGARD's CR is
  3.5x to 5.9x higher with `xhuff|zstd:3` and 9.9x to 15.5x higher with the in-house chain, with the
  pointwise bound still met. A strictly equal-RMSE comparison would need hpMGARD's tolerance scaled per
  field; the RMSE columns in the CSV allow that.
- Dedup geometry cache: pressure with the dedup base, cold 1.010 s (geometry 0.527 s) vs warm 0.103 s.
- zstd workers: no effect on the Huffman chains (their zstd input is 8 MB and takes about 10 ms) and a 1.4%
  size cost from independent jobs, so multithreading now starts at 16 MB inputs, where it matters: the
  tuner's `raw|zstd:3` configuration codes its 132 MB stream in 0.034 s with 8 workers against 0.10 s before.
- All hpMGARD runs met the bound (max err / tol = 1.000; 0.500 where the tuner chose an unsplit base).

## Addendum, 2026-11-20 01:10: device-resident base path, block 4096, outliers (job 5955003, MI250X frontier00040)

Changes: restriction, base split (cell average, dedup, plain), base quantisation and the decoded base now run
on the device and the base streams are coded from device memory (operator f7560f6); the MGARD-X dictionary
cap is back to 8192 with up to 1% out-of-dictionary codes stored as MGARD-X outliers instead of a host
fallback, and the Huffman block defaults to 4096 (204febc). No tuning in any row; warm-up excluded; 16 host
threads; decode on the same device. CSV: `documents/gpu/mi250x_devbase_2026-11-20.csv`.

| case (fixed chain) | bytes | CR | compress s | GB/s | decode s | decode GB/s | stats | base | residual | code |
|---|---|---|---|---|---|---|---|---|---|---|
| p4 pressure 1e-2, hip xhuff\|zstd:3 | 136,535 | 1938 | 0.057 | 4.6 | 0.066 | 4.0 | 0.006 | 0.014 | 0.002 | 0.015 |
| p4 pressure 1e-3, hip xhuff\|zstd:3 | 455,262 | 581 | 0.065 | 4.1 | 0.075 | 3.5 | 0.005 | 0.016 | 0.002 | 0.023 |
| p4 pressure 1e-4, hip xhuff\|zstd:3 | 1,149,209 | 230 | 0.076 | 3.5 | 0.075 | 3.5 | 0.005 | 0.017 | 0.002 | 0.034 |
| p4 pressure 1e-3, hip huffman\|zstd:3 | 129,189 | 2048 | 0.158 | 1.7 | 0.255 | 1.0 | 0.005 | 0.018 | 0.007 | 0.112 |
| p4 pressure 1e-3, cpu huffman\|zstd:3 | 129,189 | 2048 | 0.372 | 0.7 | 0.398 | 0.7 | 0.006 | 0.024 | 0.107 | 0.118 |
| p4 pressure 1e-3, hip dedup base, xhuff\|zstd:3 | 559,933 | 473 | 0.080 | 3.3 | 0.085 | 3.1 | 0.007 | 0.018 | 0.002 | 0.026 |
| p4 velocity (3 comp, 794 MB) 1e-3, hip xhuff\|zstd:3 | 4,142,576 | 192 | 0.294 | 2.7 | 0.531 | 1.5 | 0.047 | 0.060 | 0.006 | 0.139 |
| p4 velocity 1e-3, hip huffman\|zstd:3 | 1,778,670 | 446 | 0.552 | 1.4 | 0.899 | 0.9 | 0.056 | 0.058 | 0.021 | 0.372 |
| p5 pressure (8 blocks, 457 MB) 1e-3, hip xhuff\|zstd:3 | 608,231 | 752 | 0.105 | 4.4 | 0.101 | 4.5 | 0.011 | 0.021 | 0.003 | 0.033 |

The restriction column is zero because the device path launches it asynchronously and its time lands in the
base stage; the difference between the stage sum and the total (about 0.02 s for a scalar field) is the
component upload (265 MB pinned, 6 to 8 ms), the symbol-stream copies to the host and launch/sync latencies.

Block size (`HPMGARD_XHUFF_BLOCK`), p4 pressure 1e-3, hip compress + hip decode:

| block | bytes | compress s | decode s |
|---|---|---|---|
| 1024 | 504,547 | 0.064 | 0.055 |
| 4096 (new default) | 455,262 | 0.064 | 0.075 |
| 20480 (MGARD-X's previous setting) | 418,749 | 0.076 | 0.184 |

Velocity: 4.20 / 4.14 / 3.94 MB and decode 0.47 / 0.52 / 0.85 s for 1024 / 4096 / 20480. MGARD-X's decoder
runs one thread per block, so 33M symbols give only 1.6k threads at 20480 and 8k at 4096; the encoder also
gains a little from the smaller blocks.

What the device base path changed (p4 pressure 1e-3, xhuff): compress 0.087 -> 0.065 s, decode 0.183 ->
0.075 s (block 4096 included), base stage 0.025 -> 0.016 s, residual 0.003 -> 0.002 s. Velocity went from
0.772 s (with the 65536 dictionary that made MGARD-X's codebook construction slow) to 0.294 s with the 8192
dictionary plus outliers. The dedup split on the GPU gives the same streams as the CPU on these fields.

## Addendum, 2026-11-20 02:00: fused GPU path and GPU FRBP (job 5955128, MI250X frontier00093)

Changes (operator 81ceb47, 662e7bf): statistics on the device after the single upload; one fused kernel per cell
batch for restriction, base split (plain, cell average, dedup), quantisation, decoded base, prolongation and
residual quantisation, with one overflow-flag read per component; FRBP stage on the GPU (byte-identical to the
CPU format, verified: the `frbp|zstd:3` stream of the pressure field is 304,247 bytes from both backends and
the dedup variant 453,843 bytes from both); zstd of each stream on a worker thread while the device codes the
next one; fused reconstruction on decompression from device-decoded streams; memoised operators. No tuning,
warm-up excluded, 16 host threads, decode on the GPU unless noted. CSV: `documents/gpu/mi250x_fused_2026-11-20.csv`.

| case | bytes | CR | compress s | GB/s | decompress s | GB/s |
|---|---|---|---|---|---|---|
| p4 pressure 1e-3, frbp\|zstd:3 | 304,247 | 870 | 0.035 | 7.6 | 0.034 | 7.8 |
| p4 pressure 1e-3, frbp (no zstd) | 6,572,219 | 40 | 0.032 | 8.3 | 0.032 | 8.3 |
| p4 pressure 1e-3, xhuff\|zstd:3 | 455,262 | 581 | 0.047 | 5.6 | 0.065 | 4.1 |
| p4 pressure 1e-3, huffman\|zstd:3 (host coder) | 129,189 | 2048 | 0.131 | 2.0 | 0.224 | 1.2 |
| p4 pressure 1e-3, CPU, huffman\|zstd:3 | 129,189 | 2048 | 0.355 | 0.75 | 0.394 | 0.67 |
| p4 pressure 1e-3, frbp\|zstd:3, decode on CPU | 304,247 | 870 | 0.035 | 7.6 | 0.183 | 1.5 |
| p4 pressure 1e-2 / 1e-4, frbp\|zstd:3 | 67,698 / 884,329 | 3909 / 299 | 0.031 / 0.042 | 8.5 / 6.3 | 0.032 / 0.035 | 8.3 / 7.6 |
| p4 pressure 1e-3, dedup base, frbp\|zstd:3 | 453,843 | 583 | 0.048 | 5.5 | 0.045 | 5.9 |
| p4 velocity (794 MB) 1e-3, frbp\|zstd:3 | 3,072,866 | 258 | 0.131 | 6.1 | 0.390 | 2.0 |
| p5 pressure (457 MB, 8 blocks) 1e-3, frbp\|zstd:3 | 193,412 | 2364 | 0.056 | 8.2 | 0.056 | 8.2 |
| p4 pressure 1e-3, tuned (list incl. frbp) | 218,949 (raw\|zstd:3) | 1209 | 0.494 (0.45 tuning) | | 0.216 | |

Breakdown of the 0.035 s (pressure 1e-3, frbp|zstd:3): statistics 0.2 ms, fused kernel 3 ms, FRBP + stream copy
3 ms, and about 28 ms of host-side data movement: the 265 MB memcpy that extracts the component into pinned
memory (about 10 ms) [removed at 03:00], the upload (6 to 8 ms), the zstd of the residual stream (about 8 ms, overlapped only
with later streams) and the header. Compared with the morning's 0.318 s this is 9x; the paper's 100 GB/s is
device-to-device with fixed-rate coding and no host stages, so the remaining factor is almost entirely the
host boundary (the GPU work is now about 6 ms per field).

Codec picture on this field: GPU FRBP + zstd is smaller than MGARD-X Huffman + zstd here (304 vs 455 KB) and
faster (both directions 8 GB/s class); the in-house Huffman + zstd stays the smallest (129 KB) at a quarter of
the throughput. The GPU tuner list now includes `frbp|zstd:3`; the auto default stays `xhuff|zstd:3`.

## Addendum, 2026-11-20 03:00: block-resident path, no host extraction copy (job 5955222, MI250X frontier00054)

Corrected 2026-11-20 10:50: the first version of this addendum quoted the fastest of four pressure runs
(0.027 s, 9.8 GB/s) as the result and a 1.3x compression gain. The table now gives the median and the range
where the job repeated a case and marks every other row as a single run; the previous round's figures, quoted
for comparison, were single runs as well.

Changes (operator c07a347, tools c10aa3b): the caller's interleaved block is uploaded once per `Compress` call
through two 32 MB pinned staging buffers on two streams, an OpenMP memcpy into one buffer overlapping the
link transfer of the other; the statistics and fused kernels read component c with stride ncomp, so the
265 MB host extraction memcpy of the previous round is gone and a 3-component variable is de-interleaved by
the kernels. Decompression mirrors it: every component is reconstructed interleaved into one device block and
downloaded once, chunked and pipelined, straight into the caller's buffer. New `DecompressInto(double* out)`:
the ADIOS2 operator now reconstructs directly into its destination buffer (before: a temporary vector, a zero
fill and a second 265 MB copy). The bench times `DecompressInto` into a buffer allocated and touched
beforehand, as ADIOS2's destination would be, and records the decode breakdown. No tuning, warm-up excluded,
decode on the GPU unless noted. The job ran two cases four times, at 16, 8, 32 and 56 host threads (the
thread sweep); every other row ran once at 16 threads. CSV: `documents/gpu/mi250x_resident_2026-11-20.csv`.

| case | runs | bytes | compress s | GB/s | decompress s | GB/s | previous round, compress / decompress s |
|---|---|---|---|---|---|---|---|
| p4 pressure 1e-3, frbp\|zstd:3 | 4 | 304,247 | 0.029 median (0.027 to 0.039) | 9.1 | 0.0205 median (0.020 to 0.022) | 12.9 | 0.035 / 0.034 |
| p4 pressure 1e-3, frbp (no zstd) | 1 | 6,572,219 | 0.025 | 10.6 | 0.019 | 13.9 | 0.032 / 0.032 |
| p4 pressure 1e-3, xhuff\|zstd:3 | 1 | 455,262 | 0.043 | 6.2 | 0.051 | 5.2 | 0.047 / 0.065 |
| p4 pressure 1e-3, huffman\|zstd:3 (host coder) | 1 | 129,189 | 0.136 | 1.9 | 0.238 | 1.1 | 0.131 / 0.224 |
| p4 pressure 1e-3, CPU, huffman\|zstd:3 | 1 | 129,189 | 0.274 | 0.97 | 0.291 | 0.91 | 0.355 / 0.394 |
| p4 pressure 1e-3, frbp\|zstd:3, decode on CPU | 1 | 304,247 | 0.032 | 8.3 | 0.057 | 4.6 | 0.035 / 0.183 |
| p4 pressure 1e-2, frbp\|zstd:3 | 1 | 67,698 | 0.025 | 10.6 | 0.019 | 13.9 | 0.031 / 0.032 |
| p4 pressure 1e-4, frbp\|zstd:3 | 1 | 884,329 | 0.039 | 6.8 | 0.022 | 12.0 | 0.042 / 0.035 |
| p4 pressure 1e-3, dedup base, frbp\|zstd:3 | 1 | 453,843 | 0.040 | 6.6 | 0.031 | 8.5 | 0.048 / 0.045 |
| p4 velocity (794 MB, 3 comp) 1e-3, frbp\|zstd:3 | 4 | 3,072,866 | 0.081 median (0.079 to 0.104) | 9.8 | 0.0595 median (0.059 to 0.061) | 13.3 | 0.131 / 0.390 |
| p5 pressure (457 MB, 8 blocks) 1e-3, frbp\|zstd:3 | 1 | 193,412 | 0.045 | 10.2 | 0.031 | 14.8 | 0.056 / 0.056 |
| p4 pressure 1e-3, tuned (list incl. frbp) | 1 | 218,949 (raw\|zstd:3) | 0.459 (0.45 tuning) | | 0.099 | | 0.494 / 0.216 |

Where the time goes (pressure 1e-3, frbp|zstd:3, median 0.029 s compress and 0.0205 s decompress):
- Upload of the 265 MB block: 16 to 17 ms in three of the four runs and 26 ms in the slowest, which is the
  whole spread of the compress time. Across the job the single-component uploads took 15 to 17 ms, except that
  run and the three runs that decode on the CPU (19 to 22 ms): about 16 GB/s through the staging pipeline.
- Device work: statistics 0.3 ms, fused kernel 3.3 ms, FRBP 3 ms.
- Host zstd of the residual stream. For a scalar it is the last stream, so nothing overlaps it. Estimated as
  the compress time minus the timed stages, less the same remainder of the frbp-only run: under 1 ms at 1e-2,
  3 to 4.5 ms at 1e-3, and about 10 ms at 1e-4, where the FRBP stream is 9.9 MB and so below the 16 MB
  multithreading threshold.
- Decompression: 2.6 ms of stream decoding and kernels plus 17 to 19 ms of download (about 15 GB/s).

The staging memcpy thread count does not change the upload: the 8- and 32-thread runs and the other 16-thread
runs of the job all upload in 15 to 17 ms. At 56 threads the compression is slower overall (velocity code
stage 30 ms instead of 10 ms), likely because the 56 OpenMP threads then compete with the zstd worker threads
for the 56 cores. The two link copies are about 60 percent of the compression time and 85 percent of the
decompression time at 1e-3; the device-side work is about 6 to 7 ms per 265 MB field (about 40 GB/s).

Gains against the previous round (medians where repeated, single runs otherwise):

| case | compress | decompress |
|---|---|---|
| p4 pressure, frbp\|zstd:3 | 1.2x (0.035 to 0.029 s) | 1.6x (0.034 to 0.0205 s; 1.5 to 1.7x over the four runs) |
| p4 velocity, frbp\|zstd:3 | 1.6x (0.131 to 0.081 s) | 6.6x (0.390 to 0.0595 s) |
| p5 pressure, frbp\|zstd:3 | 1.2x (0.056 to 0.045 s) | 1.8x (0.056 to 0.031 s) |
| p4 pressure, CPU only | 1.3x (0.355 to 0.274 s) | 1.35x (0.394 to 0.291 s) |
| p4 pressure, GPU compress and CPU decode | 1.1x (0.035 to 0.032 s) | 3.2x (0.183 to 0.057 s) |

The velocity decompression gains most because the previous version reconstructed the components one by one
into a host vector, zero-filled it and copied it again. The CPU-only gains come from the parallel scalar
copies and the in-place output.

What remains is the link, plus the residual zstd at tight tolerances. If ADIOS2 hands the operator device
buffers, both copies vanish and the field runs at the device rate (about 40 GB/s, the lower end of the mp-MDR
paper's range). With host buffers the floor is one upload plus one download; the further overlaps available
are to reconstruct and download cell ranges in a pipeline on decompression (hides the 3 ms of kernels under
the download) and, for multi-component variables, to run the fused kernel of component c under the upload of
component c+1, which needs the upload to be split per component rather than per block. Both are second-order
(10 to 15 percent). Multithreaded zstd for the 10 MB FRBP stream at 1e-4 is untested; on the 8 MB Huffman
streams it gave no gain and cost 1.4% in size.

## Addendum, 2026-11-20 10:50: who computes the statistics, and the CUDA shim

Statistics inside the operator. For every component of every block the compressor computes the minimum, the
maximum and the sum of squares in one pass: on the GPU with a reduction kernel over the resident block,
reading component c with stride ncomp; on the CPU with an OpenMP reduction. They serve two purposes:
- the scale that turns a relative tolerance into an absolute bound: the RMS for REL_L2, which is the
  operator's default mode, and the range for REL_RANGE;
- constant detection in every mode: a component whose minimum equals its maximum is stored as one 8-byte value
  and skips the pipeline. The six Reynolds-stress fields of the p4 file take this path at step 0.

NaN and Inf do not depend on the statistics: the quantiser's overflow flag catches both in every mode, and the
component is then stored raw.

The converter's pre-pass. `hpmgard_adios_compress` first reads every block of a variable at a step, computes one
scale per component over all blocks, and passes absolute per-component tolerances with mode ABS, so every block
of the variable gets the same absolute bound. The operator cannot do this itself: ADIOS2 calls it once per
block with that block's data and no communicator, so inside the operator a relative tolerance can only be
relative to the block. An application that wants one bound per variable has to reduce the scale over its ranks
before Put and pass ABS tolerances, as the converter does. In the converter's case the operator's sum of
squares goes unused; its minimum and maximum still drive the constant check. The pre-pass costs a second full
read of every compressed variable per step from the file (not measured); `--block-norm` and `--mode ABS` skip it.

Cost of the operator's statistics (job 5955222, 1e-3):

| field | chain | statistics | whole compress |
|---|---|---|---|
| p4 pressure, GPU | frbp\|zstd:3 | 0.3 ms | 0.029 s (median) |
| p5 pressure, GPU | frbp\|zstd:3 | 0.4 ms | 0.045 s |
| p4 velocity, GPU | frbp\|zstd:3 | 2.0 ms | 0.081 s (median) |
| p4 pressure, CPU | huffman\|zstd:3 | 2.7 ms | 0.274 s |

The velocity pays more because each of its three component passes reads the whole interleaved block; one pass
computing all components would cut that to about a third. In ABS mode the separate pass could be folded into
the fused kernel, which already reads every value and returns a flag, saving the kernel and one synchronous
readback per component: under 1 percent of a host-to-host call now, a few percent of the device-only time if
ADIOS2 hands over device buffers. Neither change is made.

CUDA shim. Operator 1eb2d10 maps the stream API used by the pipelined copies (`hipStream_t`, `hipStreamCreate`,
`hipStreamSynchronize`, `hipMemcpyAsync`) to CUDA. A name-by-name check finds every HIP runtime name used by
the device sources mapped. The CUDA build is still not compiled, since Frontier has no nvcc.
