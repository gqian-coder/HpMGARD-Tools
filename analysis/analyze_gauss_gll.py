#!/usr/bin/env python3
"""Which 1-D node set carries the per-cell DOFs of vtkDGHex/Pressure_aver?
Test: for every pair of cells sharing an in-plane face (edge of the quad layer), extrapolate both
cells' degree-4 interpolants to the shared face under a node hypothesis and measure the jump.
The field is z-uniform, so the k=0 slice of every cell is used (2-D problem on the quad layer).
usage: analyze_gauss_gll.py <dumpdir>
"""
import sys, numpy as np
d = sys.argv[1].rstrip('/')
NC, NP = 264640, 2194465
conn = np.fromfile(f'{d}/p4_conn_s0.bin', dtype=np.int64).reshape(NC, 27)
pr = np.fromfile(f'{d}/p4_Pressure_aver_s0.bin', dtype=np.float64).reshape(NC, 5, 5, 5)   # [c][k][j][i]
u = pr[:, 0, :, :]                     # k = 0 slice: [c][j][i], i <-> r, j <-> s
rng = pr.max() - pr.min()
corners = conn[:, :4]                  # quad corners: 0=(-1,-1) 1=(+1,-1) 2=(+1,+1) 3=(-1,+1)

def lagrange_row(nodes, x):
    n = len(nodes); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w

# oriented edges (first corner -> second corner = increasing tangential coordinate)
edges = [((1, 2), 'r+'), ((0, 3), 'r-'), ((0, 1), 's-'), ((3, 2), 's+')]
first = np.concatenate([corners[:, e[0][0]] for e in edges]); second = np.concatenate([corners[:, e[0][1]] for e in edges])
key = np.minimum(first, second) * NP + np.maximum(first, second)
order = np.argsort(key, kind='stable'); ks = key[order]
same = ks[1:] == ks[:-1]
a, b = order[:-1][same], order[1:][same]
cnt = np.unique(ks, return_counts=True)[1]
print(f"edges: {len(key)} total, keys shared by exactly 2 cells: {(cnt==2).sum()}, by 1 (boundary): {(cnt==1).sum()}, by >2: {(cnt>2).sum()}")
flip = first[a] != first[b]           # opposite traversal direction in the two cells
print(f"pairs used: {len(a)}  (reversed orientation in {flip.sum()} of them)")

def edge_values(nodes):
    wp, wm = lagrange_row(nodes, +1.0), lagrange_row(nodes, -1.0)
    F = np.empty((4 * NC, 5))
    F[0*NC:1*NC] = np.einsum('i,cji->cj', wp, u)    # r+ face, values along s
    F[1*NC:2*NC] = np.einsum('i,cji->cj', wm, u)    # r- face
    F[2*NC:3*NC] = np.einsum('j,cji->ci', wm, u)    # s- face, values along r
    F[3*NC:4*NC] = np.einsum('j,cji->ci', wp, u)    # s+ face
    return F

hyp = {
  'Gauss-Legendre': np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]),
  'GLL (Lobatto)':  np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0]),
  'equispaced':     np.linspace(-1, 1, 5),
  'Chebyshev-Gauss': -np.cos((2*np.arange(5)+1)*np.pi/10),
}
# scale of the in-plane variation across one cell (mean over cells of max-min of the 5x5 slice)
cell_var = (u.reshape(NC, 25).max(1) - u.reshape(NC, 25).min(1))
print(f"pressure range {rng:.1f}; mean within-cell in-plane variation {cell_var.mean():.4g}; median {np.median(cell_var):.4g}")
for name, nodes in hyp.items():
    F = edge_values(nodes)
    fa, fb = F[a], F[b].copy()
    fb[flip] = fb[flip][:, ::-1]
    jump = np.abs(fa - fb)
    print(f"  {name:16s}: mean|jump| = {jump.mean():10.4g}  median = {np.median(jump):10.4g}  99% = {np.quantile(jump, 0.99):10.4g}  mean|jump|/range = {jump.mean()/rng:.3e}")
# control: nodal values at the outermost nodes without extrapolation (== GLL hypothesis by construction)
