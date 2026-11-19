# VKI HODG dataset: format review and compression feasibility

Date: 2026-11-18. Scope: `/lustre/orion/proj-shared/cfd164/vki_hodg/` (focus `ho_dg_p4_sol/sol_1_aver.bp`), compared with the
old `3d_case/p4/run94_p4/sol/sol_aver.bp` and the checkpoint `gongq/3d_case/sol_ckpt_p5_grouped_5/group_006.h5`.
Tools and raw outputs: `HODG/tools/`, `HODG/results/`. Everything below was measured on login node with
`Software/ADIOS2/install-adios-serial/bin/bpls`, `HODG/tools/dumpbp` and MGARD-serial 1.7.0.

## 1. What the "HODG data format" is

* The ScienceDirect paper (PII S001046552300005X, CPC 2023, DOI 10.1016/j.cpc.2023.108660) is "HODG: high-order
  discontinuous Galerkin methods for solving compressible Euler and Navier-Stokes equations, an open-source
  component-based development framework" (He, Wang, Liu, Feng, Zhang, Yuan, Wang). It describes a DG solver
  (C++11, DG-P0..P2, hybrid and second-order meshes). ScienceDirect refused the fetch (HTTP 403), so I could not
  read the paper itself; only the Mendeley dataset page was readable. It does not mention hpMusic.
* The files on disk are **not** that paper's format. They are ADIOS2 BP5 files written by hpMusic (the `globals/*`
  attributes are the hpMusic run parameters) following the **Fides / VTK "cell_grid" data model**
  (`Fides_Data_Model = "cell_grid"`, `cell_types = {"vtkDGHex"}`, a `fides/schema` JSON attribute). This is the
  VTK `vtkCellGrid` model for discontinuous-Galerkin cells: a shared geometry plus per-cell DOF arrays, with each
  attribute described by (function_space, basis, order).

## 2. Structure of `ho_dg_p4_sol/sol_1_aver.bp` (and the other p-orders)

| item | content |
|---|---|
| steps | 5, `time` = 1.25e-7 .. 6.25e-7 s (the first 5 outputs of a run) |
| mesh nodes | `coordinates/points` double {2,194,465 x 3}, xyz interleaved, shared between cells |
| connectivity | `vtkDGHex/conn` int64 {264,640 x 27}, 0-based, global point ids (block-local in the 8-block p5 file) |
| shape attrs | `shape/vtkDGHex/{function_space=HGRAD, basis="C", order=2, connectivity="vtkDGHex/conn", values="coordinates/points", dof_sharing="coordinates"}` = quadratic 27-node (HEX27) curved hexahedra with shared nodes |
| solution | `vtkDGHex/<var>` double {264,640 x 125} for scalars, {264,640 x 375} for `Velocity_aver`; attrs `<var>/vtkDGHex/{function_space=HGRAD, basis="G", order=4, values}`; no connectivity/dof_sharing, i.e. per-cell (discontinuous) DOFs |
| variables (aver) | Density_aver, Velocity_aver(3), Pressure_aver, uu, vv, ww, uv, vw, uw, Rho_ms, P_ms = 13 scalar fields |
| variables (sol_1.bp) | Density, Velocity(3), Pressure, Q_CRITERIA, vort(3), SCHLIEREN |
| sizes | per scalar field per step: p1 16.9 MB, p2 57.2 MB, p3 135.5 MB, p4 264.6 MB, p5 457.3 MB; p4 aver file 17.75 GB = 13 x 264.6 MB x 5 steps + 5 x 110 MB mesh (the static mesh is rewritten every step) |

Verified facts (scripts `tools/analyze_hodg.py`, `tools/analyze_nodes2.py`):

* **Same mesh for p1..p5**: 264,640 cells, 2,194,465 points in every file. The p5 file is written by 8 ranks: 8 blocks
  of 33,078-33,081 cells with block-local point numbering and 32,470 duplicated interface points (2,226,935 total).
* **27-node ordering = VTK `vtkDGHex` HGRAD C2 (HexC2Basis) order**: corners 0-7, edges 8-11 (bottom), 12-15
  (vertical), 16-19 (top), node 20 = cell center, 21 = bottom face (t=-1), 22 = top face (t=+1), 23 = r- face,
  24 = r+ face, 25 = s- face, 26 = s+ face. Checked geometrically on all 264,640 cells (node 20 is the cell centroid,
  21/22 the face centroids, 8/12/16 the edge midpoints, deviations only from curvature). Note this differs from
  VTK's legacy `vtkTriQuadraticHexahedron` order (side faces 20-23, bottom 24, top 25, center 26).
* **Mesh topology**: a 2-D quad layer of 8,270 cells / 8,608 vertices extruded in z over 32 layers (dz = 0.007396 for
  every cell, 65 z-levels). 284,064 distinct corner vertices, each corner shared by up to 12 cells (7.45 on average).
  All points are referenced.
* **DOF layout inside a cell**: (p+1)^3 values in lexicographic order with the first reference axis (r, i) fastest,
  then s (j), then t (k). Vectors are stored `[cell][point][component]` (x,y,z interleaved per point): component
  ranges of Velocity_aver are U in [-138, 346], V in [-138, 138], W in [-3.3, 3.3].
* **DOF locations = Gauss-Legendre points** (the hpMusic "gsp" convention), not Gauss-Lobatto. Test: for 518,304
  pairs of cells sharing an in-plane face, both cells' degree-4 interpolants were extrapolated to the shared face
  under a node hypothesis; restricted to smooth cells with real variation, the median jump relative to the
  within-cell variation is 0.000 for Gauss-Legendre nodes and 0.032-0.049 for Gauss-Lobatto nodes. The basis token
  "G" is not a documented VTK token (VTK documents C, I, F, A), so a Fides/VTK reader needs this convention to be
  registered; worth confirming the intended meaning of "G" with Norbert/Ken, but the data say Gauss-Legendre.
* **Physical position of a DOF** (i,j,k) of cell c is therefore
  `x = sum_n N_n(xi_i, xi_j, xi_k) * points[conn[c,n]]`, with N_n the 27 HexC2 shape functions and xi the
  (p+1) Gauss-Legendre nodes on [-1,1]. It is not stored anywhere in the file.
* **The data are an initial transient, not converged statistics**: step 0 of `sol_1_aver.bp` is identical to step 0
  of `sol_1.bp`, `uu` is exactly 0 at step 0 and ~776 by step 4. 40% of all pressure DOFs equal 90400 (the initial
  value); only 24% of the cells have a within-cell pressure variation above 1e-3 of the range; the field is uniform
  along z (RMS neighbour difference 6e-6 of range along k vs 1.4e-2 along i and 7.7e-2 along j). Any compression
  ratio measured on these files is inflated.

### Answer to question 1 (does the file contain correct mesh information, where is it)

Yes. The mesh is `coordinates/points` + `vtkDGHex/conn` with the `shape/vtkDGHex/*` attributes (quadratic HEX27,
shared nodes, 0-based global indices, VTK HexC2 node order). The p-order of every solution variable is in
`<var>/vtkDGHex/order`, the number of DOFs per cell is the second array dimension ((p+1)^3 or 3(p+1)^3). What the
file does not store is the location of the solution points; that follows from basis "G" + order + the HEX27 mapping
as above.

## 3. Comparison with the old `sol_aver.bp` and the checkpoint `group_006.h5`

| | new HODG `sol_1_aver.bp` (VKI, HEX) | old `run94_p4/sol/sol_aver.bp` (3-D rotor) | checkpoint `group_006.h5` (3-D rotor, p5) |
|---|---|---|---|
| geometry | true quadratic HEX27 mesh, shared nodes, 2.19 M points | **linear HEXA_8 sub-cells** (3.68e9 hexes, 29.5e9 int64 = 236 GB connectivity) with one coordinate per solution point, points replicated per high-order cell; triangle surface zones written as QUAD_4 | none (mesh lives in separate hpMesh `.pxda` / `dofxyz_proc*.dat` files) |
| coordinates | `coordinates/points` {N,3} interleaved, shared | `/<zone>/GridCoordinates/CoordinateX/Y/Z`, one value per field value (2.99e9 per array) | none |
| solution layout | one array per variable, `[cell][sp]` ((p+1)^3 per cell), vectors `[cell][sp][comp]`, no replication | one 1-D array per variable, `[point]` in the replicated-point order, p-order lost | `/sol/q` one array, AoS `[cell][sp][var]` with nv = 5 conserved variables, `/sol/{p_cell, pt_cell, type_cell}` per cell |
| p-order | attribute per variable (uniform p) | not recoverable | `p_cell` per cell (all 5 here), `pt_cell` = DOFs per cell (56 TET, 126 PRISM, 91 PYRAMID) |
| shapes | HEX only | mixed, hidden by the sub-cell decomposition | TET10 5,548,366 / PRISM18 2,050,454 / PYRAMID14 1,535 (hpMusic type codes 9/15/25); no HEX |
| blocks | 1 (p1-p4) or 8 (p5), one block per writer rank | 296 blocks (37 subfiles x 8 ranks) | 76 of the 576 original rank blocks regrouped into one file (`block_groups.json`) |
| size per scalar | 264.6 MB (p4) | 23.9 GB (volume zone, p4) | 22.8 GB for 5 variables |

So the statement that the HODG arrays "share the logic of the checkpoint" is correct in the sense that matters for
mp-MGARD: both are cell-contiguous blocks of solution-point values at hpMusic's Gauss points, i fastest, with no
replicated nodes, and nothing of the old `sol_aver.bp` point-cloud layout. The differences: HODG is SoA (one array per
variable, components interleaved only for vectors) where the checkpoint is AoS over the 5 conserved variables; HODG
has a uniform p per file where the checkpoint has p per cell; HODG carries the real mesh, the checkpoint none; HODG is
HEX-only and 2.5-D while the rotor is mixed TET/PRISM/PYR.

**Purge damage found while comparing (not caused by this work):** `3d_case/p4/run94_p4/sol/sol_aver.bp` has only
`data.0` of its 37 subfiles left (metadata intact, blocks 0-7 of 296 readable, bpls throws "couldn't open data.1";
directory mtime Nov 14 06:37). `p2/run26_p2` and `p3/run62_p3` `sol_aver.bp` have no data files at all. Intact:
`p5/run222_p5/sol/sol_aver.bp` (37 subfiles, 998 GB) and `p1/sol-unencrypted-uncompressed/sol_2896000_aver.bp`.
A value-level p4 old-vs-new comparison is impossible beyond blocks 0-7, and the two cases are different geometries
anyway (rotor vs VKI cascade).

## 4. Can `mgard` and `CompressMGARDCentroidOperator` compress the HODG arrays?

### 4.1 Plain MGARD (ADIOS2 `mgard` operator): yes, mechanically

Measured with MGARD-X SERIAL (`tools/run_mgard_sweep.sh`, `results/mgard_sweep_p4.csv`) on p4 `Pressure_aver`
step 0 (33.08 M doubles, range 45,985), tolerance = eb x range, ABS mode, default 3 global levels:

| layout handed to MGARD | s | eb 5e-3: CR / max err / RMSE/range | eb 1e-3: CR / max err / RMSE/range | compress / decompress time |
|---|---|---|---|---|
| 1-D {33,080,000} | inf | **171** / 179 (bound 230) / 7.4e-4 | **108** / 34.5 (bound 46) / 1.8e-4 | 6 s / 2.3 s |
| 1-D | 0 | 217 / 365 (1.6x bound) / 1.1e-3 | 138 / 70.9 (1.5x) / 2.3e-4 | 8 s / 5 s |
| 2-D {264,640 x 125} (as stored) | inf | 120 / 179 / 7.6e-4 | 84 / 35.0 / 1.8e-4 | 3 s / 3 s |
| 2-D | 0 | 107 / 349 / 9.3e-4 | 62 / 74.0 / 2.0e-4 | 5 s / 5 s |
| 4-D {264,640 x 5 x 5 x 5} (cell tensor) | inf | 46 / 2.9 (1% of bound) / 1.2e-5 | 32 / 0.67 / 2.7e-6 | 137 s / 40 s |
| 4-D | 0 | 136 / 270 / 6.8e-4 | 92 / 51.7 / 1.5e-4 | 139 s / 42 s |

Density_aver behaves the same (1-D s=inf CR 177 / 115). Conclusions: the operator works on these arrays; the
flattened 1-D view is the best plain-MGARD layout, because the stored 2-D shape mixes three tensor axes in its second
dimension and the 4-D view has only 5 points per axis (few levels, a very conservative L-inf estimate, 30x slower).
The ADIOS2 operator uses the block shape it is given, i.e. 2-D, so a writer or wrapper should present each block as
1-D (or MGARD should be given the natural {ncells, p+1, p+1, p+1} shape only under s=0). The high absolute CRs come
from the near-uniform initial field and will drop on a developed solution.

### 4.2 CompressMGARDCentroidOperator: no, not as it stands

The plugin's data model is the old `sol_aver.bp` one: a nodal field on a linear mesh where every field value has its
own coordinate (`nnodes == length of CoordinateX/Y/Z`, `nodes_per_cell` field values per connectivity row) and the
same physical location is replicated in many cells, which `group_centroid` de-duplicates. In HODG (a) there are
33 M field values per variable but only 2.19 M mesh points, and the field values are not at mesh points at all
(interior Gauss points); (b) coordinates are one interleaved {N,3} array, not `CoordinateX/Y/Z`; (c) connectivity has
27 nodes per cell against 125 DOFs per cell; (d) DG DOFs are never replicated, so `group_centroid` would degenerate
to the full field (ngroups == nnodes) and only `reorder_mgard` (space-filling-curve order + 1-D MGARD) would remain,
after generating one coordinate per DOF from the HEX27 mapping. The plugin therefore cannot read these files
(`CompressMGARDCentroidOperator.cpp:977` rejects `coordinate length != nnodes`). The DG analogue of its coarse/fine
split is exactly mp-MGARD's base + residual refactoring (section 5), so adapting the centroid plugin is not the
right path.

## 5. Can mp-MGARD's p-refactoring work on HODG? Yes, this is the natural fit

* The HODG arrays are already in the engine's `CellBatch` form (`data[cell*n_sp*nv + sp*nv + var]`, HEX,
  `n_sp = (p+1)^3`, nv = 1 for scalars and 3 for Velocity, i fastest), one batch per file because p is uniform.
  `MeshInfo.p_full` comes from the `order` attribute, `cell_corners` for `DEDUP_SFC` from `conn[:, 0:8]` (VTK corner
  order maps to tensor-corner order by the permutation 0,1,3,2,4,5,7,6) and `corner_xyz` from `coordinates/points`.
  The p5 file's 8 blocks are independent batches with their own local numbering.
* The data favour the method: 98.1% of the pressure variance is between cell means and 1.9% inside cells; the p1
  L2 restriction with Gauss-Legendre operators leaves 1.16e-2 of the energy (about the mean) in the residual
  (cell-mean base: 1.89e-2), while holding 6.4% of the DOFs.
* **One algorithmic mismatch must be fixed**: `RefactorEngine::Initialize` builds only the Gauss-Lobatto table and
  "HEX/QUAD use the GLL table" (`mp-MGARD/src/core/refactor_engine.hip`, `nek_gpu_rp_pipeline.h`, inherited from the
  Nek5000 path). HODG/hpMusic hex DOFs are at Gauss-Legendre points, for which the Gauss table `RPOperatorTable`
  (`rp_operators.h`, built from hpMusic's own node values) already exists in the tree but is wired to nothing.
  Reconstruction stays exact with either table (u = P base + residual), but with the GLL operators the p1 residual
  carries 1.73e-2 instead of 1.16e-2 of the energy (49% more) and the error model's amplification factors are those of
  the wrong basis. The rotor experiments never exposed this because that mesh has no HEX cells.
* **Emulated gain** (`tools/emulate_prefactor.py`, `results/emulate_*.out`; CPU, L-inf budget split
  tol_base x ||P||inf + tol_res = tol with ||P_1->4||inf = 3.87; base coded with MGARD 1-D s=inf, residual quantized
  with quantum 2 tol_res and entropy coded; the max error always equals tol_res, so the bound holds):

| scheme, eb = 5e-3 (tol 230) | base bytes | residual bytes (zlib / lzma) | CR (zlib / lzma) | max err |
|---|---|---|---|---|
| direct MGARD 1-D s=inf (reference) | | | 171 | 179 |
| p1 Gauss base (20% of budget) + residual | 503 KB | 230 KB / 33 KB | **361 / 494** | 184 |
| cell-mean base (20%) + residual | 47 KB | 267 KB / 47 KB | **841 / 2797** | 184 |
| eb = 1e-3 (tol 46): direct MGARD 1-D | | | 108 | 34.5 |
| p1 Gauss base + residual | 611 KB | 270 KB / 48 KB | 300 / 402 | 36.8 |
| cell-mean base + residual | 115 KB | 303 KB / 67 KB | 633 / 1455 | 36.8 |

  On this field the residual quantizes to zero in most cells, which is why the base-plus-residual split beats direct
  MGARD by 2-5x (and lzma, which sees the 32 identical z-layers, by much more). mp-MGARD's FRBP/bitplane coders sit
  between zlib and lzma here; the point is the structure, not the coder. Expect far smaller ratios on a converged
  average, but the same ordering of methods.

## 6. Feasibility summary and what to do next

1. **Format**: fully understood and self-consistent (mesh, connectivity order, DOF order, node set, component
   interleave all verified). Only the meaning of basis "G" is undocumented in VTK; confirm "Gauss-Legendre points"
   with the writer.
2. **Data**: the current files are the first five output steps of a fresh run and are mostly at the initial state.
   Ask for a `sol_*_aver.bp` from a developed/converged window before quoting compression ratios.
3. **Plain MGARD**: works now (1-D per block recommended; CR 171 at eb 5e-3 L-inf on this transient field).
4. **Centroid plugin**: not applicable to DG data; do not port it.
5. **mp-MGARD**: the algorithm applies directly. Work items (API, not algorithm): a `MeshReader` for the cell_grid
   layout (scalars nv=1, Velocity nv=3, `cell_corners`/`corner_xyz` from conn/points, per-block for p5); a Gauss
   operator table for HEX in `RefactorEngine`/`NekGPURPPipeline` (the `RPOperatorTable` in `rp_operators.h`, same
   API as the GLL table); rebuild mp-MGARD (its `build_hip`/`build_mgard` binaries were lost in the purge); then run
   p1..p5. The CPU emulation above is the expected shape of the result: a cell-mean or p1 base plus a small residual,
   with the residual nearly free wherever the solution is smooth within a cell.

Raw dumps used for the measurements were written to the session scratchpad (tmpfs) and can be regenerated with
`tools/dumpbp <file.bp> <variable> <step> <out.bin>`.
