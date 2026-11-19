# M1 gate: hpMGARD (CPU reference) versus MGARD-X on the VKI HODG data

Date: 2026-11-19. Machine: Frontier login node (AMD EPYC, 16 OpenMP threads for hpMGARD; MGARD-X
1.7.0 SERIAL build, single thread). Data: `vki_hodg/ho_dg_p4_sol/sol_1_aver.bp` step 0, all 13 scalar
fields (Velocity as one 3-component variable) and the p5 pressure field (8 blocks treated as one block,
no mesh). Tolerance: one pointwise bound per field, `tol = eb × RMS(field)` (REL_L2), eb = 1e-2, 1e-3,
1e-4; both codecs are given the same absolute bound (MGARD-X 1-D flattened, `s = inf`; `s = 0` listed
for reference, it is an L2 bound and exceeds the pointwise bound by 1.5x). hpMGARD runs with the
auto-tuner (3% sample) and its default codec list; the times include tuning. Raw data and scripts:
`HODG/results/gate/` (`run_gate.sh`, `gate_p4.csv`), summary by `analysis/summarize_gate.py`.

The six Reynolds-stress fields are identically zero at step 0 of these files (nothing to compress; both
coders store a few bytes, MGARD-X reports CR 7.9 with a zero tolerance) and are excluded from the means.

## Result

| eb | hpMGARD CR (geomean, 7 fields) | MGARD-X s=inf | MGARD-X s=0 | ratio vs s=inf | hpMGARD s (mean) | MGARD-X s (mean, serial) |
|---|---|---|---|---|---|---|
| 1e-2 | 5435 | 241 | 324 | 22.6x | 0.80 | 5.4 |
| 1e-3 | 1637 | 131 | 160 | 12.5x | 0.82 | 5.6 |
| 1e-4 | 549 | 37 | 41 | 14.8x | 1.08 | 5.6 |

Every hpMGARD run met its bound exactly (max error / tol = 1.000: the residual quantiser's worst case).
Single-threaded, fixed configuration, p4 pressure at 1e-3: compress 0.61 s, decompress 0.35 s, against
MGARD-X's 4.1 s compress (serial). With 16 threads: 0.31 s / 0.30 s. The thread scaling is poor because
the Huffman and zstd stages are still single-threaded; the numeric loops are parallel.

The tuner's choices varied with the field and the tolerance: p1 cell-average base for pressure, density
and the mean squares at 1e-2 and 1e-3; cell mean (p0) at 1e-4 and for the velocity vector; p2 base for
P_ms at 1e-2; "no split" (p4) for velocity at 1e-4. Dedup never won on this data (its coarse stream is
one value per vertex, and here the cell means are already extremely compressible).

Full per-field table: `HODG/results/gate/gate_summary.md`.

## Codec ablation (p4 pressure, eb 1e-3, fixed p1 cell-average base, same chain for base and residual)

| chain | bytes | CR | compress s (16 thr) | decompress s |
|---|---|---|---|---|
| `huffman|zstd:3` | 129,189 | 2048 | 0.32 | 0.32 |
| `huffman|zstd:19` | 115,942 | 2283 | 0.55 | 0.31 |
| `huffman` | 8,182,108 | 32 | 0.31 | 0.27 |
| `frbp|zstd:3` | 304,247 | 870 | 0.23 | 0.13 |
| `frbp|zstd:19` | 163,421 | 1619 | 0.50 | 0.12 |
| `frbp` | 6,572,219 | 40 | 0.24 | 0.10 |
| `raw|zstd:3` | 155,492 | 1702 | 0.28 | 0.13 |
| `raw|zstd:19` | 65,311 | 4052 | 0.48 | 0.12 |

Observations:
- zstd-3 versus zstd-19 (the requested comparison): level 19 buys 10% (after Huffman) to 58% (raw ints)
  smaller streams for 1.7x to 2x the compression time; decompression time is unchanged. Level 3 stays the
  default; level 19 is a tuner candidate for write-once data.
- An entropy coder alone (Huffman or FRBP) cannot code below about one bit per value, so both stall near
  CR 30 to 40 on these sparse residuals; the run-length/LZ stage after them is what delivers the ratio.
- `raw|zstd` beat `huffman|zstd` here: this field is uniform along z (32 identical layers), and zstd finds
  those long-range matches in the aligned int32 stream that Huffman's variable-length codes hide. On
  developed turbulence the ordering is expected to flip; `raw|zstd:3` has been added to the tuner's default
  candidate list so the choice is made per block from the data.

## End-to-end through ADIOS2 (driver + plugin)

| file | steps | input | output | wall (16 thr) | bound check |
|---|---|---|---|---|---|
| p1 `sol_1_aver.bp` | 5 | 1.65 GB | 554 MB (fields ≈ 5 MB, passthrough mesh 549 MB) | 11.0 s | all 13 fields, all steps: max err = tol |
| p4 `sol_1_aver.bp` | 1 | 3.5 GB (step) | 112 MB (fields ≈ 2 MB, mesh 110 MB) | 9.3 s | all fields: max err = tol |
| p5 `sol_1_aver.bp` (8 blocks) | 1 | 6.1 GB (step) | 113 MB (fields ≈ 2.5 MB, mesh 111 MB) | 21.6 s | per-block bounds met; see note |

Note on p5: the first run normalised the tolerance per block (eb × RMS of each block); against the
variable-wide RMS the max error then lands between 1.1 and 1.8 × eb. The driver now computes one scale per
variable and component over all blocks of a step and passes absolute bounds to every block
(`--block-norm` restores the old behaviour). Rerun of p5 step 0 with variable-wide bounds: 16.9 s, 113 MB, max error = tol on every component of every field.

The compressed HODG files are dominated by the mesh, which the writer repeats on every step
(coordinates 53 MB + connectivity 57 MB for this mesh); compressing or de-duplicating the mesh across
steps is the next lever on file size and is independent of the field compressor.

## Verdict for the gate

Ratio: met, by 12x to 23x in geometric mean at equal pointwise bound (far above the 1.2x to 5x the
paper measured on developed rotor data; this dataset is an initial transient and inflates all ratios,
but the ordering is consistent). Speed: met on CPU, 6.7x faster single-threaded and 13x with 16 threads
against MGARD-X serial at equal bound. The GPU backend (M2) still has to be measured against MGARD-X
HIP.

## Addendum: first HIP backend measurement (login03, MI210, 2026-11-19 10:10)

`HODG/results/gate/gpu_login03.csv`, p4 pressure, eb 1e-3, fixed p1 cell-average configuration:

| backend | compress s | decompress s | CR | stages (stats / refactor / base / residual / code) |
|---|---|---|---|---|
| CPU, 16 threads | 0.28 | 0.34 | 2048.5 | 0.008 / 0.016 / 0.018 / 0.071 / 0.119 |
| HIP (naive, per-call staging) | 0.46 | 0.28 | 2048.5 | 0.008 / 0.057 / 0.020 / 0.213 / 0.119 |
| HIP compress, CPU decode | 0.32 | 0.26 | 2048.5 | cross-backend: identical bytes, bound met |

The streams are byte-identical between CPU and HIP. The naive HIP backend loses time moving 264 MB
arrays over PCIe for every stage, and the host-side Huffman+zstd coding is the largest single stage on
both backends; device-resident data and GPU codecs are therefore the next two steps. MGARD-X's own HIP
compressor returns status 1 on this MI210 (a known MGARD issue; it works on the MI250X compute nodes),
so the GPU comparison against MGARD-X has to be made on a compute node.

## Addendum 2: device-resident HIP backend (login03, MI210, 10:15)

Each component is now uploaded once and stays on the device; the residual is formed and quantised on
the GPU against the prolonged base in one fused pass, and decoding reconstructs in one fused pass
(`HODG/results/gate/gpu_login03_resident.csv`, p4 pressure, eb 1e-3, fixed configuration):

| backend | compress s | decompress s | stages (stats / refactor / base / residual / code) |
|---|---|---|---|
| CPU, 16 threads | 0.27 | 0.34 | 0.008 / 0.012 / 0.018 / 0.069 / 0.114 |
| HIP, resident | 0.25 | 0.27 | 0.008 / 0.019 / 0.019 / 0.020 / 0.107 |
| HIP compress, CPU decode | 0.26 | 0.28 | identical bytes |
| CPU compress, HIP decode | 0.32 | 0.45 (one outlier of 3.3 s, not reproducible) | identical bytes |

The residual stage dropped from 0.21 s to 0.02 s and the HIP compressor is now at parity with 16 CPU
threads on this small field; the host-side Huffman+zstd coding (0.11 s) is the remaining bottleneck on
both backends, so the GPU codecs come next.