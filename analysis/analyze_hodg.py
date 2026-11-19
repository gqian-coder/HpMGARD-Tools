#!/usr/bin/env python3
"""Structural checks on the raw dumps of ho_dg_p4_sol/sol_1_aver.bp step 0.
usage: analyze_hodg.py <dumpdir>
"""
import sys, numpy as np
d = sys.argv[1].rstrip('/')
NC, NSP, NP, P = 264640, 125, 2194465, 4
conn = np.fromfile(f'{d}/p4_conn_s0.bin', dtype=np.int64).reshape(NC, 27)
pts  = np.fromfile(f'{d}/p4_points_s0.bin', dtype=np.float64).reshape(NP, 3)
pr   = np.fromfile(f'{d}/p4_Pressure_aver_s0.bin', dtype=np.float64).reshape(NC, NSP)
vel  = np.fromfile(f'{d}/p4_Velocity_aver_s0.bin', dtype=np.float64).reshape(NC, NSP, 3)

print("== connectivity / mesh ==")
print("conn min/max:", conn.min(), conn.max(), " npoints:", NP)
ref = np.bincount(conn.ravel(), minlength=NP)
print("points never referenced:", int((ref == 0).sum()))
print("references per point histogram (refs:count):", {int(k): int(v) for k, v in zip(*np.unique(ref, return_counts=True))})
corners = conn[:, :8]
nvert = len(np.unique(corners))
print(f"distinct corner vertices: {nvert}  (8*ncells/nvert = {8*NC/nvert:.2f} corner DOFs per vertex)")
print(f"distinct edge nodes: {len(np.unique(conn[:, 8:20]))}, face nodes: {len(np.unique(conn[:, 21:27]))}, center nodes: {len(np.unique(conn[:, 20]))} (= ncells? {len(np.unique(conn[:,20]))==NC})")
# node-order check on every cell: node 20 is the mean of the 8 corners (center), 21/22 are the means of the bottom/top corners
X = pts[conn]  # (NC,27,3)
cen = X[:, :8].mean(1); err_c = np.abs(X[:, 20] - cen).max(1)
bot = X[:, [0,1,2,3]].mean(1); top = X[:, [4,5,6,7]].mean(1)
err_b = np.abs(X[:, 21] - bot).max(1); err_t = np.abs(X[:, 22] - top).max(1)
e8 = np.abs(X[:, 8] - 0.5*(X[:, 0]+X[:, 1])).max(1); e12 = np.abs(X[:, 12] - 0.5*(X[:, 0]+X[:, 4])).max(1); e16 = np.abs(X[:, 16] - 0.5*(X[:, 4]+X[:, 5])).max(1)
hsz = np.linalg.norm(X[:, 6] - X[:, 0], axis=1)  # cell diagonal
for name, e in [("node20=center", err_c), ("node21=bottom face", err_b), ("node22=top face", err_t), ("node8=mid(0,1)", e8), ("node12=mid(0,4)", e12), ("node16=mid(4,5)", e16)]:
    r = e / hsz
    print(f"  {name:20s}: max |dev|/diag = {r.max():.3e}, 99.9% = {np.quantile(r, 0.999):.3e}")
# extrusion check: is the mesh extruded along z?
dz = X[:, 4, 2] - X[:, 0, 2]
print(f"corner0->corner4 dz: min {dz.min():.6f} max {dz.max():.6f}; dx,dy of that edge max |.|: {np.abs(X[:,4,:2]-X[:,0,:2]).max():.2e}")
zl = np.unique(np.round(pts[:, 2], 7)); print("distinct z-levels of points:", len(zl), "z range", pts[:,2].min(), pts[:,2].max())

print("\n== field layout ==")
print("Velocity component ranges (interpreting layout [cell][sp][comp]):", [(float(vel[:,:,c].min()), float(vel[:,:,c].max())) for c in range(3)])
u = pr.reshape(NC, P+1, P+1, P+1)  # [cell][k][j][i], i fastest
rng = pr.max() - pr.min()
def rmsdiff(a, b): return np.sqrt(np.mean((a-b)**2))
print(f"Pressure range {rng:.1f}; RMS of neighbour differences within cells, as fraction of range:")
print(f"  along i (stride 1):  {rmsdiff(u[:,:,:,1:], u[:,:,:,:-1])/rng:.3e}")
print(f"  along j (stride 5):  {rmsdiff(u[:,:,1:,:], u[:,:,:-1,:])/rng:.3e}")
print(f"  along k (stride 25): {rmsdiff(u[:,1:,:,:], u[:,:-1,:,:])/rng:.3e}")
rs = np.random.RandomState(0); idx = rs.permutation(NC)
print(f"  between consecutive cells (DOF 0): {rmsdiff(pr[1:,0], pr[:-1,0])/rng:.3e};  between random cells: {rmsdiff(pr[idx,0], pr[:,0])/rng:.3e}")
cm = pr.mean(1); wv = pr.var(1).mean(); bv = cm.var()
print(f"  variance split: between-cell-means {bv/(bv+wv)*100:.1f}%  within-cell {wv/(bv+wv)*100:.1f}%")

print("\n== Gauss vs GLL node hypothesis (face-continuity test on z-stacked cell pairs) ==")
gauss = {1: np.array([-0.5773502691896258, 0.5773502691896258]),
         4: np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640])}
gll   = {1: np.array([-1.0, 1.0]), 4: np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0])}
def lagrange_row(nodes, x):
    n = len(nodes); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w
key = {tuple(r): c for c, r in enumerate(corners[:, 4:8].tolist())}   # top corners of A
pairs = [(key[tuple(r)], c) for c, r in enumerate(corners[:, 0:4].tolist()) if tuple(r) in key]
pairs = np.array(pairs); A, B = pairs[:, 0], pairs[:, 1]
print(f"z-stacked pairs found (B.bottom corners == A.top corners, same order): {len(pairs)}")
for name, nd in [("Gauss-Legendre", gauss[4]), ("GLL", gll[4])]:
    wtop = lagrange_row(nd, +1.0); wbot = lagrange_row(nd, -1.0)
    fa = np.einsum('k,ckji->cji', wtop, u[A]); fb = np.einsum('k,ckji->cji', wbot, u[B])
    jump = np.abs(fa - fb)
    # scale: variation of the field across one cell in the k direction
    var_k = np.abs(u[A][:, -1] - u[A][:, 0])
    print(f"  {name:15s}: mean|jump| = {jump.mean():.4g}  median = {np.median(jump):.4g}  99% = {np.quantile(jump,0.99):.4g}   (mean within-cell k-variation of A: {var_k.mean():.4g})")
# control: nodal values at the outermost nodes without extrapolation
jump_nodal = np.abs(u[A][:, -1] - u[B][:, 0]); print(f"  {'nodal (no extrap.)':15s}: mean|jump| = {jump_nodal.mean():.4g}  median = {np.median(jump_nodal):.4g}")

print("\n== p4 -> p1 restriction residual energy (Kronecker L2 projection), Gauss vs GLL ==")
def weights(nodes):
    # quadrature weights for Lagrange basis on given nodes via exact integration of basis polynomials
    n = len(nodes); Wt = np.zeros(n)
    xs, ws = np.polynomial.legendre.leggauss(n + 2)
    for j in range(n):
        Wt[j] = sum(wq * lagrange_row(nodes, xq)[j] for xq, wq in zip(xs, ws))
    return Wt
def RP(nodes_lo, nodes_hi):
    Pm = np.array([lagrange_row(nodes_lo, x) for x in nodes_hi])   # (n_hi, n_lo)
    wf, wc = weights(nodes_hi), weights(nodes_lo)
    Rm = (Pm.T * wf) / wc[:, None]                                  # (n_lo, n_hi)
    return Rm, Pm
tot = np.sum((pr - pr.mean())**2)
for name, nds in [("Gauss-Legendre", gauss), ("GLL", gll)]:
    R, Pm = RP(nds[1], nds[4])
    base = np.einsum('xk,yj,zi,ckji->cxyz', R, R, R, u)
    rec  = np.einsum('kx,jy,iz,cxyz->ckji', Pm, Pm, Pm, base)
    res  = u - rec
    print(f"  {name:15s}: residual energy / total (about mean) = {np.sum(res**2)/tot:.4e};  base holds {8/125*100:.1f}% of DOFs; RMS(res)/range = {np.sqrt(np.mean(res**2))/rng:.3e}")
# p0 (cell mean) for reference
res0 = pr - pr.mean(1, keepdims=True)
print(f"  {'p0 cell mean':15s}: residual energy / total = {np.sum(res0**2)/tot:.4e}; RMS(res)/range = {np.sqrt(np.mean(res0**2))/rng:.3e}")
