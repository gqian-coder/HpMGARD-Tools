#!/usr/bin/env python3
"""Second look at the node-set question and at how uniform the step-0 field is."""
import sys, numpy as np
d = sys.argv[1].rstrip('/')
NC, NP = 264640, 2194465
conn = np.fromfile(f'{d}/p4_conn_s0.bin', dtype=np.int64).reshape(NC, 27)
pts  = np.fromfile(f'{d}/p4_points_s0.bin', dtype=np.float64).reshape(NP, 3)
pr   = np.fromfile(f'{d}/p4_Pressure_aver_s0.bin', dtype=np.float64).reshape(NC, 5, 5, 5)
vel  = np.fromfile(f'{d}/p4_Velocity_aver_s0.bin', dtype=np.float64).reshape(NC, 5, 5, 5, 3)
rng = pr.max() - pr.min()
u = pr[:, 0]                               # [c][j][i]
vm = np.linalg.norm(vel[:, 0], axis=-1)    # |V| on the k=0 slice, [c][j][i]

print("== how uniform is the step-0 field? ==")
flat = pr.ravel(); vals, cnts = np.unique(np.round(flat, 6), return_counts=True); m = cnts.argmax()
print(f"most frequent pressure value {vals[m]:.6f} occurs in {cnts[m]/flat.size*100:.2f}% of all DOFs")
cv = u.reshape(NC, 25).max(1) - u.reshape(NC, 25).min(1)
for thr in [1e-9, 1e-6, 1e-3, 1e-2]:
    print(f"  cells with within-cell pressure variation > {thr:g} x range: {(cv > thr*rng).sum()} ({(cv > thr*rng).mean()*100:.2f}%)")
cmean = u.reshape(NC, 25).mean(1)
print(f"  cells whose mean deviates from the modal value by > 1e-3 x range: {(np.abs(cmean - vals[m]) > 1e-3*rng).sum()}")

print("\n== wall-adjacent cells: |V| on the first node line vs the second (no-slip wall) ==")
corners = conn[:, :4]
edges = [((1, 2), 'r+'), ((0, 3), 'r-'), ((0, 1), 's-'), ((3, 2), 's+')]
first = np.concatenate([corners[:, e[0][0]] for e in edges]); second = np.concatenate([corners[:, e[0][1]] for e in edges])
key = np.minimum(first, second) * NP + np.maximum(first, second)
uk, cnt = np.unique(key, return_counts=True)
bkeys = set(uk[cnt == 1].tolist())
isb = np.array([k in bkeys for k in key.tolist()])
cell = np.tile(np.arange(NC), 4); etype = np.repeat(np.arange(4), NC)
bc, be = cell[isb], etype[isb]
# node lines parallel to the edge: line 0 nearest the edge, line 1 next; for r+ : i=4,3 ; r- : i=0,1 ; s- : j=0,1 ; s+ : j=4,3
def lines(arr, c, e):
    out0 = np.empty((len(c), 5)); out1 = np.empty((len(c), 5)); out2 = np.empty((len(c), 5))
    for t, (i0, i1, i2, ax) in enumerate([(4, 3, 2, 'i'), (0, 1, 2, 'i'), (0, 1, 2, 'j'), (4, 3, 2, 'j')]):
        sel = e == t
        if ax == 'i': out0[sel], out1[sel], out2[sel] = arr[c[sel], :, i0], arr[c[sel], :, i1], arr[c[sel], :, i2]
        else:         out0[sel], out1[sel], out2[sel] = arr[c[sel], i0, :], arr[c[sel], i1, :], arr[c[sel], i2, :]
    return out0, out1, out2
v0, v1, v2 = lines(vm, bc, be)
mid = (pts[first[isb]] + pts[second[isb]]) / 2
# classify boundary edges by location: the blade is the closed curve with |V| small; inlet/outlet/periodic have |V| ~ freestream
m0, m1, m2 = v0.mean(1), v1.mean(1), v2.mean(1)
print(f"boundary edges: {isb.sum()} ( {isb.sum()//32} per layer )")
for name, sel in [("all boundary edges", np.ones(len(bc), bool)), ("edges with |V| line0 < 0.5 x line1 (candidate wall)", m0 < 0.5*m1), ("edges with |V| line0 > 0.9 x line1 (far field)", m0 > 0.9*m1)]:
    if sel.sum() == 0: continue
    r01 = m0[sel] / np.maximum(m1[sel], 1e-30); r12 = m1[sel] / np.maximum(m2[sel], 1e-30)
    print(f"  {name:52s}: n={sel.sum():6d}  median |V|0={np.median(m0[sel]):.4g} |V|1={np.median(m1[sel]):.4g} |V|2={np.median(m2[sel]):.4g}  median ratio line0/line1={np.median(r01):.3f}  line1/line2={np.median(r12):.3f}")
print("  reference: for a linear profile from a no-slip wall, Gauss nodes give line0/line1 = (1-0.9062)/(1-0.5385) = 0.203 and line1/line2 = 0.462;")
print("             GLL nodes give line0/line1 = 0 (DOF on the wall) and line1/line2 = 0.345; equispaced 0 and 0.5")
print("  x-range of candidate wall edge midpoints:", mid[m0 < 0.5*m1][:, 0].min() if (m0 < 0.5*m1).any() else None, mid[m0 < 0.5*m1][:, 0].max() if (m0 < 0.5*m1).any() else None)

print("\n== Gauss vs GLL jump test restricted to pairs with moderate, resolved variation ==")
order = np.argsort(key, kind='stable'); ks = key[order]; same = ks[1:] == ks[:-1]
a, b = order[:-1][same], order[1:][same]; flip = first[a] != first[b]
def lagrange_row(nodes, x):
    n = len(nodes); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w
def edge_values(nodes):
    wp, wm = lagrange_row(nodes, +1.0), lagrange_row(nodes, -1.0)
    F = np.empty((4 * NC, 5))
    F[0*NC:1*NC] = np.einsum('i,cji->cj', wp, u); F[1*NC:2*NC] = np.einsum('i,cji->cj', wm, u)
    F[2*NC:3*NC] = np.einsum('j,cji->ci', wm, u); F[3*NC:4*NC] = np.einsum('j,cji->ci', wp, u)
    return F
hyp = {'Gauss-Legendre': np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]),
       'GLL (Lobatto)': np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0])}
# smoothness proxy per cell: ratio of second differences to first differences along i and j (small => resolved polynomial)
d1 = np.abs(np.diff(u, axis=2)).mean((1, 2)) + np.abs(np.diff(u, axis=1)).mean((1, 2))
d2 = np.abs(np.diff(u, n=2, axis=2)).mean((1, 2)) + np.abs(np.diff(u, n=2, axis=1)).mean((1, 2))
smooth = d2 / np.maximum(d1, 1e-30)
ca, cb = cell[a], cell[b]
va = cv[ca]; vb = cv[cb]
for lo, hi in [(1e-5, 1e-3), (1e-3, 1e-2), (1e-2, 1.0)]:
    sel = (va > lo*rng) & (va < hi*rng) & (vb > lo*rng) & (vb < hi*rng) & (smooth[ca] < 0.5) & (smooth[cb] < 0.5)
    if sel.sum() < 10: print(f"  variation in ({lo:g},{hi:g}) x range, smooth cells: only {sel.sum()} pairs"); continue
    line = f"  variation in ({lo:g},{hi:g}) x range, smooth cells, {sel.sum():6d} pairs: "
    for name, nodes in hyp.items():
        F = edge_values(nodes); fa, fb = F[a[sel]], F[b[sel]].copy(); fb[flip[sel]] = fb[flip[sel]][:, ::-1]
        nj = np.abs(fa - fb).mean(1) / (0.5 * (va[sel] + vb[sel]))
        line += f"{name}: median jump/variation = {np.median(nj):.3f}   "
    print(line)
