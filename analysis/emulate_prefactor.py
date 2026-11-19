#!/usr/bin/env python3
"""CPU emulation of an mp-MGARD-style split for vtkDGHex/Pressure_aver (p4, step 0):
   base = L2 restriction to p1 (Gauss-Legendre Kronecker operators) or p0 (cell mean),
   base compressed with MGARD-X 1-D (s=inf, abs tol_base), residual = u - P(decoded base)
   quantized with quantum 2*tol_res and entropy-coded (zlib-9 / lzma; order-0 entropy bound).
   L-inf guarantee: tol_base * ||P||_inf + tol_res = tol.
usage: emulate_prefactor.py <dumpdir> <eb_rel_range> [base_fraction_list]
"""
import sys, os, subprocess, zlib, lzma, numpy as np
d = sys.argv[1].rstrip('/'); eb = float(sys.argv[2])
fracs = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.2', '0.5'])]
nodeset = sys.argv[4] if len(sys.argv) > 4 else 'gauss'   # gauss | gll : node set assumed by the R/P operators
T = '/lustre/orion/cfd164/proj-shared/gongq/HODG/tools/mgard_test2'
NC = 264640
pr = np.fromfile(f'{d}/p4_Pressure_aver_s0.bin', dtype=np.float64).reshape(NC, 5, 5, 5)
rng = pr.max() - pr.min(); tol = eb * rng; nbytes = pr.size * 8
print(f"Pressure_aver p4 step 0: n={pr.size}, range={rng:.1f}, eb={eb:g} -> abs tol={tol:.4g}; raw {nbytes/1e6:.1f} MB")

gauss = {1: np.array([-0.5773502691896258, 0.5773502691896258]),
         4: np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640])}
def lagrange_row(nodes, x):
    n = len(nodes); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w
def weights(nodes):
    n = len(nodes); xs, ws = np.polynomial.legendre.leggauss(n + 2)
    return np.array([sum(wq * lagrange_row(nodes, xq)[j] for xq, wq in zip(xs, ws)) for j in range(n)])
def RP(lo, hi):
    Pm = np.array([lagrange_row(lo, x) for x in hi]); wf, wc = weights(hi), weights(lo)
    return (Pm.T * wf) / wc[:, None], Pm

def mgard_1d(arr, abs_tol, tag):
    fn = f'{d}/emul_{tag}.bin'; out = f'{d}/emul_{tag}_dec.bin'
    arr.astype(np.float64).tofile(fn)
    r = subprocess.run([T, fn, str(arr.size), "0", "inf", "default", tag, out, f"{abs_tol:.17g}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    f = r.stdout.strip().split(','); cbytes = int(f[8]); maxerr = float(f[10])
    dec = np.fromfile(out, dtype=np.float64).reshape(arr.shape)
    return dec, cbytes, maxerr

def code_sizes(k):
    b = k.astype(np.int32).tobytes()
    z = len(zlib.compress(b, 9)); l = len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))
    vals, cnt = np.unique(k, return_counts=True); p = cnt / cnt.sum(); H = -(p * np.log2(p)).sum()
    return z, l, H * k.size / 8, len(vals)

gll = {1: np.array([-1.0, 1.0]), 4: np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0])}
nodes = gauss if nodeset == 'gauss' else gll
print(f'operators built on {nodeset} nodes')
R, Pm = RP(nodes[1], nodes[4])
Ainf = np.abs(Pm).sum(1).max() ** 3     # ||P1->4 (x) P (x) P||_inf
# corner-represented p1 base: E maps p1 Gauss samples -> values at the cell corners (+-1), Pc interpolates corners -> p4 nodes
E  = np.array([lagrange_row(nodes[1], -1.0), lagrange_row(nodes[1], +1.0)])
Pc = np.array([lagrange_row(np.array([-1.0, 1.0]), x) for x in nodes[4]])
def norms(M, tag):
    print(f"  {tag:34s}: 1-D inf-norm {np.abs(M).sum(1).max():.3f}  2-norm {np.linalg.norm(M, 2):.3f}   3-D inf-norm {np.abs(M).sum(1).max()**3:.3f}  2-norm {np.linalg.norm(M, 2)**3:.3f}")
print("prolongation amplification:")
norms(Pm, f"P {nodeset} p1 nodes -> {nodeset} p4 nodes")
norms(Pc, f"P corners (+-1) -> {nodeset} p4 nodes")
print(f"||P_1->4||_inf (3-D) = {Ainf:.3f}")
print("scheme,base_frac,tol_base,tol_res,base_bytes,res_zlib,res_lzma,res_entropy_bound,total_zlib,CR_zlib,CR_lzma,CR_entropy,max_err,rmse_over_range,n_symbols")
for scheme in ['p1_gauss', 'p1_corner', 'p0_mean']:
    for f in fracs:
        if scheme == 'p1_gauss':
            base = np.einsum('xk,yj,zi,ckji->cxyz', R, R, R, pr); A = Ainf
            tol_b = f * tol / A; tol_r = tol - tol_b * A
            dec, bbytes, bmax = mgard_1d(base.ravel(), tol_b, 'p1base')
            rec = np.einsum('kx,jy,iz,cxyz->ckji', Pm, Pm, Pm, dec.reshape(base.shape))
        elif scheme == 'p1_corner':
            bg = np.einsum('xk,yj,zi,ckji->cxyz', R, R, R, pr)                 # true L2 projection onto p1 (Gauss)
            base = np.einsum('ax,by,cz,ixyz->iabc', E, E, E, bg)              # re-expressed at the 8 cell corners
            A = np.abs(Pc).sum(1).max() ** 3
            tol_b = f * tol / A; tol_r = tol - tol_b * A
            dec, bbytes, bmax = mgard_1d(base.ravel(), tol_b, 'p1corner')
            rec = np.einsum('kx,jy,iz,cxyz->ckji', Pc, Pc, Pc, dec.reshape(base.shape))
        else:
            base = pr.reshape(NC, -1).mean(1); A = 1.0
            tol_b = f * tol; tol_r = tol - tol_b
            dec, bbytes, bmax = mgard_1d(base, tol_b, 'p0base')
            rec = np.broadcast_to(dec[:, None, None, None], pr.shape)
        res = pr - rec; q = 2 * tol_r
        k = np.rint(res / q); recon = rec + k * q
        err = np.abs(recon - pr); mx = err.max(); rmse = np.sqrt((err**2).mean())
        z, l, hb, nsym = code_sizes(k)
        print(f"{scheme},{f},{tol_b:.4g},{tol_r:.4g},{bbytes},{z},{l},{int(hb)},{bbytes+z},{nbytes/(bbytes+z):.1f},{nbytes/(bbytes+l):.1f},{nbytes/(bbytes+hb):.1f},{mx:.4g},{rmse/rng:.3e},{nsym}")
    print(f"  ({scheme}: base = {base.size} values = {base.size/pr.size*100:.1f}% of DOFs)")
