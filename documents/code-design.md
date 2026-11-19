# CompressHpMGARDOperator + HpMGARD-Tools: code design

Status: revision 3, 2026-11-19. Scope narrowed to **compression only** (section 0); auto-tuning of the
hierarchy added (section 3). Companion to [plan.md](plan.md) and
[01_hodg_format_review.md](01_hodg_format_review.md).

Repositories (both under `gongq/HODG/`):

| repo | remote | content |
|---|---|---|
| `CompressHpMGARDOperator/` | `git@github.com:gqian-coder/CompressHpMGARDOperator.git` (cloned, empty) | the ADIOS2 operator library only: `src/`, `CMakeLists.txt`, `cmake/`, `runconf_*`, `README.md`, `LICENSE` |
| `HpMGARD-Tools/` | to be created on GitHub by you (`gh` is not installed on Frontier); name is a proposal | drivers/wrappers, tests, fixtures, scripts, analysis, documents, in the style of `Unstructured-ReMesh` |

`mp-MGARD/` is not modified; files are copied from it with provenance.

## 0. Goal and scope

An ADIOS2 plugin operator for high-order DG/FR solution arrays in the Fides `cell_grid` layout
(`vtkDGHex/<var>` = `[cell][solution point][component]`), using the p-multilevel split (base + residual)
of mp-MGARD as the decorrelating transform and error-bounded quantisation + lossless coding, in the
software shape of `CompressMGARDCentroidOperator`:

1. one shared library `libCompressHpMGARDOperator.so`, loaded through `ADIOS2_PLUGIN_PATH`;
2. CPU (OpenMP), HIP and CUDA backends from one source tree; compression and decompression on every
   backend; any stream decodes on any backend;
3. a self-describing, vendor-neutral bitstream; the decoder needs no parameters or environment;
4. all four cell shapes (HEX, TET, PRISM, PYRAMID); HEX verified on the VKI data, the others
   implemented but unverified until a sample exists.

**Compression only (v1).** The operator produces one error-bounded compressed stream per block and
decodes it back in full. What is dropped from mp-MGARD for v1: the stored multi-level ladder for
retrieval, the bitplane (BABP) codec and its chunk metadata, `PlanRetrieval` / `Retrieve` /
`RetrieveAtLevel` / `RetrievalSession`, the predicted-error tables. What stays: the p-refactoring as the
transform (steps 1 to 5 below, a two-level decomposition unless the auto-tuner chooses an intermediate
level), the error model's budget allocation, FRBP, and the dedup/SFC base. The header keeps a version
number, flags and a stream directory so that refactoring/progressive retrieval can be added later
without a format break.

**Hard goal: better than MGARD-X in both compression ratio and speed** at the same error bound, on
the same hardware. Section 4 gives the evidence so far and the measurement gate that decides go/no-go
before GPU work starts.

## 1. Data model

| item | VKI files | general rule |
|---|---|---|
| block | one ADIOS2 block of `vtkDGHex/<var>`, dims `{ncells, nsp*ncomp}`, double or float | `{ncells, nsp*ncomp}` per cell type; local arrays, one block per writer rank (p5: 8 blocks) |
| cell type | hexahedron | `vtkDGHex`→HEX, `vtkDGTet`→TET, `vtkDGWdg`→PRISM, `vtkDGPyr`→PYRAMID |
| order p | `<var>/vtkDGHex/order` (1..5) | nsp = (p+1)^3 HEX, (p+1)(p+2)(p+3)/6 TET, (p+1)^2(p+2)/2 PRISM, (p+1)(p+2)(2p+3)/6 PYRAMID |
| DOF order | lexicographic, r fastest (hpMusic `xi_sp[k]` innermost) | hpMusic solution-point order per shape |
| node set | Gauss-Legendre (basis "G", verified) | Gauss-Legendre default, Gauss-Lobatto selectable |
| components | 1, or 3 for Velocity, interleaved per point | = `CellBatch::nv` |
| mesh | `coordinates/points {N,3}`, `vtkDGHex/conn {ncells,27}`, VTK HexC2 node order, block-local ids in multi-block files | only for the DEDUP base and the Hilbert ordering |

The block maps onto `mp_mgard::CellBatch` without a copy. One operator instance = one (cell type, p)
batch; mixed meshes arrive as separate variables.

## 2. Algorithm (approved 2026-11-19)

Per block and component, with base order `p_min` (default 1) and the data order p:

1. **Restriction to the base order (Gauss L2 projection).** `u^(b) = R u`, R built on the data node set
   (= hpMusic's `restrict_pN` tables = mp-MGARD's `RPOperatorTable`), Kronecker in r-fastest order.
2. **Closed (corner) representation of the base.** The base polynomial is stored by its values at the
   closed node set of order `p_min` (p_min = 1: the 8 cell corners): `c = (E⊗E⊗E) u^(b)`. Same
   polynomial, invertible; the base→data prolongation `P_c` is pure interpolation (∞-norm 1.0 in 3-D
   instead of 3.87 for Gauss-point values; spectral norm 3.95 instead of 6.09). p_min = 0 is the
   cell-mean base (one value per cell, `P_c` = broadcast).
3. **Residual.** `r = u − P_c c` at the data nodes (one level; an intermediate level is used only when
   the auto-tuner finds it pays).
4. **Base split.** CELLAVG: weighted mean `avg` + remainder `c − avg`. DEDUP_SFC: per-vertex mean over
   the corner values that share a mesh vertex (`conn[:,0:8]` via the corner permutation
   `[0,1,3,2,4,5,7,6]`), remainder in Hilbert order of the corner coordinates; exact in location because
   the base values sit on the vertices.
5. **Quantise and code (closed loop).** The residual is taken against the *decoded* base,
   `r = u − P_c(avg' + rem')`, and quantised with quantum `2·tol`, so the pointwise error of the output is
   bounded by `tol` regardless of how coarsely the base was coded. The base quantum (`base_quantum_ratio ×
   2·tol`) is therefore a pure rate-allocation knob (coarser base = fewer base bytes, more residual bytes)
   that the tuner chooses; `A = ‖P_c‖∞` only governs how much base quantisation noise the residual has to
   absorb, which is why the closed representation (A = 1.0) still pays. Each stream is coded with a
   chainable lossless pipeline (section 2.2). Reconstruction `u' = P_c (avg' + rem') + r'`.
   (Implemented as designed here; this supersedes the open-loop budget `tol_base·A + tol_res ≤ tol` of
   revision 2, which the emulation used.)

Engine changes versus mp-MGARD: tensor node set selectable (`RPOperatorTable` Gauss / `GLLRPOperatorTable`
Lobatto behind one `TensorRPTable` interface); E composed into the base restriction and `P_c` into the
prolongation; per-interval prolongation norms in the error model; weighted CELLAVG mean for p_min > 1.

### 2.1 What `error_model` is

The amplification-aware budget allocator from mp-MGARD (host code): from the target tolerance, the
per-component scale (RMS or range), the DOF counts and the prolongation norms it returns the absolute
quantisation tolerance of every stream (`avg_tol`, `res_tol` for the base split, `level_tol` for the
residual) so that the reconstruction meets the target after prolongation. In v1 it is a few hundred
lines in `src/core/`; its output is written into the header and never recomputed by the decoder.

### 2.2 Stream codecs

| codec | where it runs | role |
|---|---|---|
| `raw` (int32 symbols as bytes) | CPU reference | first stage; useful with `zstd` for testing |
| `frbp` (fixed-rate bit packing, mp-MGARD FIXED-mode format: 32-value blocks, 1 rate byte + sign word + rate×4 bytes) | CPU reference (new); GPU kernel ported in M2 | first stage; no entropy coding |
| `huffman[:dict_bits]` (canonical Huffman over zigzag symbols, escaped outliers, codes ≤ 32 bits) | CPU reference (new); GPU in M2 | first stage; the entropy coder |
| `zstd[:level]` (libzstd frame of the previous stage's bytes) | CPU (libzstd) | any later stage; removes runs/LZ redundancy the entropy coder cannot |
| `mgardx` (MGARD-X 1-D on a double stream) | libmgard SERIAL/HIP/CUDA | optional base-stream codec and the direct safety-net candidate |
| `xhuff[:L]` (MGARD-X Huffman; `:L` = Huffman + zstd level L inside MGARD-X) | libmgard SERIAL or HIP, chosen by the backend device | first stage; GPU entropy coder, bitstream identical on every device |
| `xrle` (MGARD-X ZeroRLE_Rans: zero run lengths + rANS) | libmgard SERIAL or HIP | first stage; plain option, no sparsity assumption |
| `xrans` (MGARD-X SymbolRans) | libmgard SERIAL or HIP | first stage; plain option |

The MGARD-X stages (added 2026-11-19) code the residual symbols straight from device memory when the HIP
backend produced them (`Backend::DeviceSymbols()`): zigzag and range check run on the device, the buffer
is wrapped without a copy, one MGARD-X compressor object per (type, dictionary, block, length) persists
for the process, and the first call per type is a warm-up. Every MGARD-X stage falls back to the in-house
Huffman (a mode byte records it) when a code is outside the dictionary (8192, `HPMGARD_XHUFF_DICT`), the
stream is shorter than 65536 symbols, MGARD-X fails or does not pass its device preflight (Frontier
login-node MI210s), or the decode-and-compare check after encoding fails (`HPMGARD_XHUFF_VERIFY=0`
skips it). The rANS types fall back on one-symbol alphabets (MGARD-X segfaults there). Results are in
`documents/mgardx_gpu_huffman_2026-11-19.md`.

Stages are chained by a spec string (`huffman|zstd:3`, `frbp|zstd:19`, `raw|zstd:3`, `huffman`, ...) so
coders can be switched and combined for testing from the ADIOS2 parameters or the bench tool; the chain is
written into the stream container and decoded without parameters. A standing test item: `zstd:3` versus
`zstd:19` on the same quantised streams (ratio versus time). First measurement on p4 pressure (section 4.1):
`huffman|zstd:3` 52 KB, `frbp|zstd:3` 54 KB, `frbp` alone 240 KB on the 3% sample; Huffman alone cannot code
below one bit per symbol, so the LZ/run-length stage after it is what captures sparse residuals.

Codec streams are specified in `documents/bitstream.md`; the CPU implementations are the reference and
the GPU kernels (M2) must produce identical bytes on identical quantised input (byte test). MGARD-X
streams are vendor-neutral by construction.

## 3. Auto-tuning of the hierarchy and budget (your suggestion)

Compression-only means the stored decomposition does not have to serve retrieval, so it can be chosen
per block (or once per variable) purely to minimise bytes. At the start of `Operate`:

1. Take a spatially spread sample of cells (every k-th cell, default 3% of the block, at least 2,000
   cells).
2. For every candidate configuration, run the real pipeline on the sample and measure the coded bytes:
   - base order `p_min ∈ {0, 1, 2}` (closed representation), optionally one intermediate level
     `{p_min, m, p}` for p ≥ 4;
   - base budget fraction `f ∈ {0.1, 0.2, 0.3, 0.5}` of the tolerance;
   - residual codec `{FRBP, HUFF_ZSTD}`; base codec `{FRBP, HUFF_ZSTD, MGARDX}`;
   - base split `{CELLAVG, DEDUP_SFC}` when a mesh is available;
   - **direct MGARD-X on the whole block** as a safety-net candidate.
3. Pick the smallest; write the choice into the header; compress the block with it.

The error bound holds for every candidate because the budget allocation is exact for each hierarchy,
so the tuner only trades bytes and time. Cost: with ~20 candidates on 3% of the cells the tuning pass is
about 60% of one full compression in the worst case; an order-0 entropy estimate of the quantised sample
prunes the field to the two or three best before the exact sample encodes, which brings it to ~10%. The
driver can tune on the first block of a variable and pass the fixed choice (`tuning=fixed` + explicit
parameters) to the remaining blocks and later time steps. The direct-MGARD-X candidate makes the
compression-ratio goal hold by construction (never worse than MGARD-X by more than the header); the
speed goal holds whenever the refactoring path wins, which the evidence below says is the common case.

## 4. Can it beat MGARD-X? Evidence and the gate

Recorded on the p5 rotor (developed turbulence, mp-MDR with FRBP residuals and the full progressive
ladder `{1,3,5}`, `stats/eb_search_p5`, `stats/p5_singleEB_ratios.csv`, variable rho, group 000):

| target rel-L2 | mp-MDR CR (achieved rel-L2) | MGARD-X CR (achieved rel-L2) |
|---|---|---|
| 1e-2 | 158 (1.3e-3) | 132 (4.1e-3) |
| 1e-3 | 110 (6.1e-4) | 20.6 (5.0e-4) |
| 1e-4 | 13.3 (9.2e-5) | 8.4 (5.2e-5) |

Even with no entropy coding and with the progressive overhead, the p-split beat MGARD-X by 1.2x to
5x in ratio. Emulated on the VKI p4 pressure field (initial transient, L∞ bound eb 5e-3 of range): direct
MGARD-X 1-D 171 (s=inf) / 217 (s=0); p1 corner base + zstd residual 614; cell-mean base 841. The
compression-only benchmark over all 13 p4 fields and one p5 block at REL_L2 1e-3 and 1e-2
(`HpMGARD-Tools/analysis/bench_compression_only.py`, results in
`HODG/results/bench_compression_only_p4.csv`) is summarised in section 4.1.

Speed: MGARD-X HIP on one MI250X GCD ran at about 1.8 GB/s with s=inf and 4 levels and 0.5 GB/s with
s=0 (October benchmarks); mp-MGARD measured its R/P sweeps at 76 GB/s and FRBP at ≥ 100 GB/s on the
same hardware. The compression-only pipeline is four streaming passes over the block (statistics,
restriction + E, fused residual + quantise, coder) plus a coder on 6.4% of the data for the base, so a
3x to 10x speed advantage over MGARD-X s=inf is expected on GPU, more against s=0, provided the ratio
mode's Huffman+zstd stage (the same stage MGARD-X pays) does not dominate. On CPU the same pipeline is
OpenMP streaming loops against MGARD-X SERIAL at ~45 MB/s (measured here: 6 s for 264 MB).

**Gate (end of M1):** the CPU reference implementation must beat MGARD-X 1-D (s=inf and s=0) on the 13
VKI p4 fields and the p5 blocks at REL_L2 1e-2, 1e-3 and 1e-4, in bytes at equal pointwise error and in
wall time on the same core count. If a variable loses on bytes, the auto-tuner's MGARD-X candidate
covers it and the result is recorded; if the pipeline loses on time, the design is revisited before any
GPU work.

### 4.1 Benchmark summary (2026-11-19, CPU emulation, `HODG/results/bench_compression_only_p4.csv`)

Setup: step 0 of the VKI files, tolerance = eb × RMS of the field (REL_L2), all codecs at the same
absolute pointwise bound; MGARD-X 1-D flattened (serial, default levels); scheme = p1 Gauss projection,
corner-stored base coded by MGARD-X 1-D at 20% of the budget, residual quantised at 80% and coded by
zstd-19 (proxy for Huffman+zstd) or by the FRBP size formula; cell-mean variant likewise. The six
Reynolds-stress fields are identically zero at step 0 and are excluded (nothing to compress). Geometric
means over the seven non-trivial p4 fields (Density, Pressure, Rho_ms, P_ms, U, V, W):

| REL_L2 target | MGARD-X s=inf | MGARD-X s=0 | p1 base + zstd residual | p1 base + FRBP residual | cell-mean base + zstd | cell-mean + FRBP |
|---|---|---|---|---|---|---|
| 1e-3 | 102 | 125 | **479** | 38 | **1304** | 34 |
| 1e-2 | 190 | 252 | **1259** | 64 | **2547** | 55 |

The p5 block behaves the same (140 / 163 vs 399 / 1263 at 1e-3). Max error of the scheme is 0.80 tol
on every field (pointwise by construction); MGARD-X s=inf lands at 1.0 tol and s=0 at 1.5 to 1.7 tol.
Per field the p1 scheme beats MGARD-X s=inf by 8x to 21x and the cell-mean scheme by 11x to 40x on
this (near-uniform, initial-transient) data; on the developed p5 rotor the paper measured 1.2x to 5x with
FRBP residuals, so the real-data expectation is a factor of a few, not an order of magnitude.

Two conclusions that shape the design: (1) **FRBP alone loses to MGARD-X** here (38 vs 102): fixed-rate
coding wastes bits on sparse residuals, so the ratio mode must entropy-code the quantised residual
(Huffman+zstd, MGARD-X's own stage); (2) the cell-mean base wins on this transient data while the paper's
rotor data favoured the per-vertex p1 base, which is exactly why the base order is left to the
auto-tuner. CPU times in this emulation (numpy einsum + zstd-19, 8 to 10 s per 264 MB field) are not
representative of the C++/OpenMP implementation; MGARD-X serial took 4 to 8 s per field. The speed goal
is settled by the M1 gate, not by this table.

## 5. Backends

One `Backend` interface, three implementations; the compressed format is backend-independent.

| operation | CPU (new, OpenMP) | HIP / CUDA (ported mp-MGARD kernels) |
|---|---|---|
| statistics (min/max/sum-sq per component) | loops | `field_stats.hip` |
| restriction / E / prolongation (Kronecker for HEX/QUAD, dense for TET/PRISM/PYRAMID) | `ApplyRestrict/ApplyProlong` exist; Kronecker sweeps added | `gpu_kernels.hip`, `nek_gpu_rp_pipeline.h` (+ E kernel) |
| cellavg split, dedup gather/scatter | loops | `nek5000_cellavg_split.hip`, `dedup_gpu.hip` |
| Hilbert order, dedup groups | host only, integer-quantised coordinates (both backends) | same host code |
| quantise + FRBP | reference implementation | `frbp_encoding.hip` |
| HUFF_ZSTD / MGARDX | libmgard SERIAL | libmgard HIP / CUDA |
| auto-tuner | host driver over the backend's sample encodes | same |

Rules: residuals are floating point, so different backends may emit different (both valid) streams for
the same input; every backend must decode every valid stream within the bound. Permutations are
computed on the host from integer-quantised coordinates and protected by CRCs. `device=auto` picks a GPU
when present, else CPU; `device=cpu` forces the CPU path (used by the cross-backend tests).

## 6. Operator (`CompressHpMGARDOperator` repo, `src/`)

```
src/operator/CompressHpMGARDOperator.{h,cpp}  PluginOperatorInterface: Operate, InverseOperate, IsDataTypeValid, GetEstimatedSize, AddExtraParameters; OperatorCreate/Destroy
src/operator/Params.{h,cpp}                   parameter parsing, HPMGARD_* env overrides, validation, defaults
src/operator/Header.{h,cpp}                   header + stream directory (de)serialisation, CRC32, versions (section 7)
src/operator/MeshAdapter.{h,cpp}              cell_grid conn/points via ADIOS2, block selection, cache, corner permutations per shape, host Hilbert/dedup geometry
src/operator/Tuner.{h,cpp}                    section 3
src/operator/Backend.{h,cpp}                  backend selection and dispatch (cpu|hip|cuda|auto)
src/core/      ported mp-MGARD core (mesh types, config, engine, operator tables, kernels, dedup, field stats, error model)
src/codec/     frbp_{cpu,gpu}.*, lossless wrappers (MGARD-X Huffman+zstd, libzstd)
compat/cuda/   mp-MGARD's hip→cuda shim
CMakeLists.txt options ENABLE_HIP / ENABLE_CUDA (exclusive, both off = CPU only), ADIOS2 ≥ 2.10, MGARD-X optional, zstd, OpenMP
runconf_cpu  runconf_hip  runconf_cuda
```

**Operate:** parse/validate params → derive nsp, ncomp → below `threshold`: raw → `CellBatch` view → mesh
for DEDUP (cache per block) → statistics → tuner (unless `tuning=fixed`) → refactor, split, quantise, code
on the selected backend → header + directory + streams into `bufferOut`; if compressed ≥ raw, rewrite as raw.

**InverseOperate:** parse preamble (type byte, versions, size, CRC) → raw flag → memcpy; else rebuild the
configuration and stream views → mesh if DEDUP (`ResolveMeshSource`: explicit
`meshfile`+`connectivity_variable`+`coordinates_variable` from params/env, or the file being read via
`EngineName` with the names embedded in the header; verify geometry CRCs) → backend `auto` → decode
streams, dequantise, reconstruct → `dataOut`.

### 6.1 Parameters (ADIOS2 `Params`; `HPMGARD_<NAME>` env overrides the runtime knobs)

| name | required | default | meaning |
|---|---|---|---|
| `cell_type` | yes | | `hexahedron|tetrahedron|wedge|pyramid` (vtkDG* spellings accepted) |
| `order` | yes | | p of the data |
| `components` | no | derived | 1 or 3, cross-checked against dims |
| `node_set` | no | `gauss` | `gauss` or `lobatto`; recorded |
| `tolerance`, `mode` | yes | mode `REL_L2` | `ABS|REL_L2|REL_RANGE`, per component with per-component scale; `normalization_scale` optional |
| `tuning` | no | `auto` | `auto` (section 3) or `fixed` (use the explicit settings below) |
| `tuning_sample` | no | 0.03 | fraction of cells sampled by the tuner |
| `base_order` | fixed | 1 | p_min ∈ {0,1,2}; `levels` optional intermediate orders |
| `base_fraction` | fixed | 0.2 | share of the tolerance given to the base |
| `base_method` | fixed | `dedup_sfc` if mesh given, else `cellavg` | `cellavg|dedup_sfc|none` |
| `base_codec`, `residual_codec` | fixed | `huff_zstd` if MGARD-X present, else `frbp` | `frbp|huff_zstd|zstd|mgardx` (base only) |
| `meshfile`, `connectivity_variable`, `coordinates_variable` | DEDUP only | self-contained on decode | one `{N,3}` coordinates array |
| `blockid` | yes | | block index per Put, selects the mesh block |
| `threshold` | no | 100000 | bytes below which a block is stored raw |
| `device`, `device_id` | no | `auto` | `auto|cpu|hip|cuda` |
| `verbose`, `timing` | no | off | stderr logging |

## 7. Metadata header (MGARD-X style)

MGARD-X's `Metadata` is a hand-serialised, fixed-field record (software and file versions, size, CRC32,
endianness, then data/decomposition/encoding fields, versions readable first; protobuf is used by classic
MGARD, not MGARD-X). The implemented header follows that shape and is specified field by field in
`HpMGARD-Tools/documents/bitstream.md`: a 22-byte preamble (`"HPMG"`, operator type 54, software and
file versions, endianness, metadata size, CRC32, flags), then block identity, discretisation (shape, p,
nsp, ncomp, DOF layout, node set), configuration (base order/representation/method, norm, tolerance, per
component scale, pointwise bound, base and residual quanta, `amp_inf`, codec chain strings, direct flag),
optional mesh identity (connectivity and coordinate variable names, vertex-group count) and geometry
CRCs, provenance (writer string, MGARD-X version, backend), and the stream directory (component, kind,
codec, offset, bytes, count per stream). Readers accept file versions 1.x, verify the CRC and fail with
explicit version numbers on anything newer. About 300 bytes per block.

## 8. HpMGARD-Tools repo (drivers, tests, scripts), Unstructured-ReMesh style

```
HpMGARD-Tools/
  CMakeLists.txt  build_scripts_{cpu,hip,cuda}.sh   builds against installed ADIOS2 + CompressHpMGARDOperator (+ MGARD-X)
  README.md  documents/                             format review, this design, bitstream.md, benchmark memos
  hpmgard_adios_compress.cpp                        MPI driver: cell_grid BP in, compressed BP out, passthrough of everything else, per-variable params from the file's attributes, blockid per Put, tune-once-per-variable, byte/time logs
  hpmgard_adios_decompress.cpp                      compressed BP → decompressed BP
  hpmgard_verify.cpp                                original vs decompressed: max / RMSE / relative per variable and block
  hpmgard_meta_dump.cpp                             prints a block's header and directory
  tests/                                            operators (R·P = I, zero residual for degree-(p−1) polynomials with the matching node set only, E/P_c exactness, ‖P_c‖∞ = 1), header round trip, FRBP CPU-vs-GPU bytes, round trips per shape/p/node set/base order, bound compliance ABS/REL_L2/REL_RANGE, plugin write/read through ADIOS2, multi-block, raw fallback, tuner determinism
  tests/fixtures/                                   streams written by CPU, HIP and CUDA builds; every backend decodes all within bound
  scripts/                                          Frontier sbatch (compress p1..p5, verify, timing), CUDA-box scripts
  analysis/                                         dumpbp, mgard_test2, analyze_*.py, extract_hpmusic_hex_tables.py, bench_compression_only.py (from HODG/tools)
```

Cross-backend matrix: streams from {CPU, HIP, CUDA} × decoders {CPU, HIP, CUDA}; decoded values equal
across decoders within 1e-12 of the range; error ≤ bound everywhere.

## 8.1 Status (2026-11-19, end of day one)

Update 2026-11-19 afternoon: HIP backend with device-resident components and fused residual/reconstruct
kernels; MGARD-X lossless stages `xhuff`/`xrle`/`xrans` with the device-resident encode path and the
`mgardx::DeviceWorks` preflight; compute-node validation (MI250X) of the GPU path and of MGARD-X direct;
the login-node MI210 is known to break MGARD-X's HIP Huffman encoder (falls back automatically).

Both repositories exist and are pushed. Built and passing on Frontier (CPU, CCE 21):
- `CompressHpMGARDOperator`: library `hpmgard` (quadrature, tensor and ported simplex operators, closed
  base, cellavg and dedup/Hilbert base splits, codecs raw/frbp/huffman/zstd with chain container,
  MGARD-X wrapper, header, closed-loop compressor, tuner) and the ADIOS2 plugin (`Operate`,
  `InverseOperate`, `MeshAdapter`, self-contained decode via `EngineName`, raw fallback, float support).
  bpls decodes compressed files once `ADIOS2_PLUGIN_PATH` points at the install (the Cray wrapper's
  auto-linked libsci must be left out of the build: `runconf_cpu` unloads `cray-libsci` and
  `darshan-runtime`).
- `HpMGARD-Tools`: `hpmgard_bench`, `hpmgard_adios_compress` (passthrough driver), `hpmgard_adios_decompress`,
  `hpmgard_verify`, `hpmgard_adios_roundtrip`, unit tests (5/5 pass), analysis scripts, documents.
- End to end on the real p1 file (1.6 GB, 13 fields × 5 steps): 11 s wall on 16 threads, every field within
  its pointwise bound on every step; the 13 fields shrink to about 5 MB, the output (554 MB) is now
  dominated by the uncompressed mesh that the HODG file rewrites every step.
- M2 started (2026-11-19 10:00): `Backend` interface for the numeric stages with the CPU implementation and a
  first HIP backend (dense per-cell operators, elementwise kernels, per-call staging); `runconf_hip` and the
  `HPMGARD_ENABLE_HIP` option; the HIP library and plugin compile and install (`install-hip`); validated on
  login03's MI210: `HPMGARD_TEST_DEVICE=hip ./test_compressor` passes 50/50 checks compressing on the GPU and
  decoding on the CPU with byte-identical stream sizes; the full HIP test suite (incl. the ADIOS2 plugin
  round trip on `device=auto`) passes on the MI210. MGARD-X streams follow the backend device at run time.
  First GPU timing (p4 pressure, 1e-3, fixed config): CPU 16 threads 0.28 s, naive HIP 0.46 s with
  byte-identical output; the HIP residual stage spends 0.21 s staging 264 MB arrays per call and the
  host-side Huffman+zstd coding (0.12 s) is the largest stage on both backends. Device-resident data (one
  upload and one fused residual/reconstruct pass per component) then brought the HIP residual stage to
  0.02 s and the HIP compressor to 0.25 s, parity with 16 CPU threads, with byte-identical streams in both
  cross-backend directions. Resume at:
  device-resident data across stages, tensor-sweep kernels, GPU FRBP/Huffman with the byte-identity test,
  MGARD-X HIP device type, CUDA via the compat shim.
- Not done: CUDA/cross-backend matrix (M5), MPI in the drivers, simplex verification, parallel Huffman/zstd.

## 9. Milestones and functionality checklist

M0 repositories (operator repo cloned, tools repo initialised locally)
1. Operator repo: README, LICENSE, CMake skeleton, `runconf_*`, `.gitignore`; first push.
2. Tools repo: README, CMake skeleton, `documents/` (this design, format review), `analysis/` moved from
   `HODG/tools`; GitHub remote created by you; first push.

M1 CPU reference implementation and the gate
3. Copy the needed mp-MGARD host sources (operator tables, quadrature, error model, sfc) with
   `documents/porting-notes.md`; keep namespaces until the one-shot rename.
4. CPU pipeline: statistics, Kronecker/dense R/P sweeps, E and `P_c`, cellavg/dedup, quantise, FRBP
   reference, HUFF_ZSTD via MGARD-X SERIAL, libzstd fallback, MGARDX base coder.
5. Error-budget allocation with per-interval norms and the closed base; weighted mean for p_min > 1.
6. Tuner (section 3) with entropy pruning and the direct-MGARD-X candidate.
7. `test_operators`, round-trip and bound tests on synthetic fields; **gate benchmark** (section 4) on
   VKI p1..p5 vs MGARD-X s=inf and s=0; memo.

M2 GPU backend
8. Copy the GPU kernels and FRBP from mp-MGARD (HIP build), E kernel, fused residual+quantise kernel,
   HUFF_ZSTD via MGARD-X HIP; `Backend` dispatch; FRBP CPU-vs-GPU byte test; GPU timing vs MGARD-X HIP.

M3 ADIOS2 operator and header
9. Plugin class and registration; Params; Header (section 7) with CRC32, version gating, raw fallback;
   `GetEstimatedSize` = raw + header worst case.
10. Full compress/decompress paths; `hpmgard_meta_dump`; header round-trip and plugin write/read tests.

M4 mesh paths, drivers, HODG runs
11. `MeshAdapter` (block selection, cache, corner permutation per shape, host dedup/Hilbert, CRCs);
    `ResolveMeshSource` (all-or-nothing rule).
12. `hpmgard_adios_compress` (passthrough, tune once per variable), `hpmgard_adios_decompress`,
    `hpmgard_verify`; p1..p5 runs on Frontier (GPU and CPU) with bound verification; memo.

M5 CUDA and cross-backend
13. CUDA build via `compat/cuda`; fixtures from the CUDA machine; the full 3×3 decode matrix.

M6 hardening
14. Namespace rename, dead-code removal, CPU performance pass (OpenMP scaling), docs.

Later (not v1): stored ladder for progressive retrieval, bitplane codec, retrieval API, level-split
write mode. The header's version, flags and directory already leave room for them.

## 10. Decisions recorded (2026-11-19)

- v1 is compression only (no stored ladder, no bitplane codec, no retrieval API); refactoring and
  progressive retrieval come later on the same header.
- Hard goal: beat MGARD-X in ratio and speed at the same bound; gate at the end of M1; direct MGARD-X
  kept as a tuner candidate so the ratio can never be worse.
- Auto-tuning of base order, levels, budget split, codecs and base split per block or once per variable.
- Operator repo = library sources only; tools/tests/scripts/wrappers/fixtures/documents in `HpMGARD-Tools`.
- CPU compressor and decompressor required; any backend decodes any stream (3×3 matrix).
- Defaults: base order 1, `dedup_sfc` with mesh else `cellavg`, MGARD-X optional (Huffman+zstd and the
  MGARDX base coder need it; FRBP and libzstd do not), tolerance semantics `REL_L2`.
- Header: MGARD-X style, no protobuf. Non-hex shapes shipped implemented but unverified.
- mp-MGARD untouched; sources copied with provenance.

Still open: the tools repo name (`HpMGARD-Tools`) and its GitHub remote.

## 11. Risks

- The ratio goal depends on the residual's entropy coder: FRBP alone wastes bits on blocks with a single
  large value; HUFF_ZSTD (MGARD-X's own stage) is the ratio mode and makes MGARD-X a practical, if
  formally optional, dependency.
- The speed goal on GPU depends on keeping the pipeline to streaming passes; the host-side Hilbert/dedup
  geometry is computed once per block and cached, the tuner is bounded by its sample size.
- Permutations: host computation + CRCs turn a cross-platform mismatch into a loud failure.
- `GetEstimatedSize` must stay generous (FRBP can exceed the input on noisy data before the raw fallback).
- mp-MGARD's HIP tests have not run since its build directories were purged; M2 starts with that rebuild
  (in mp-MGARD's own tree, unchanged).
