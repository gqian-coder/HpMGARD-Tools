#!/usr/bin/env python3
"""Extract hpMusic's hard-coded hexahedron tables and identify the node set / projection rule.
Sources (restored clone):  src/core/scheme_coef/scheme_coef_cpr_gsp_gfp_hex.C  (sp_1d_pN nodes, mean_1d_pN weights)
                           src/core/scheme_coef/dof_adapt_coef_hex.C          (p_inc_pN prolongation, p_dec_pN restriction)
usage: extract_hpmusic_hex_tables.py <hoMusic clone root> <out_dir>
"""
import re, sys, json, numpy as np
root, out = sys.argv[1].rstrip('/'), sys.argv[2].rstrip('/')
num = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def grab(text, name):
    m = re.search(r'Real\s+' + re.escape(name) + r'\s*((?:\[[^\]]*\])+)\s*=\s*(\{.*?\});', text, re.S)
    if not m: return None
    dims = [int(d) for d in re.findall(r'\[(\d+)\]', m.group(1))]
    body = re.sub(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', lambda f: repr(float(f.group(1)) / float(f.group(2))), m.group(2))  # '1./8.' -> 0.125
    vals = np.array([float(x) for x in num.findall(body)])
    return vals.reshape(dims) if dims and np.prod(dims) == vals.size else vals

gsp = open(f'{root}/src/core/scheme_coef/scheme_coef_cpr_gsp_gfp_hex.C', errors='replace').read()
dof = open(f'{root}/src/core/scheme_coef/dof_adapt_coef_hex.C', errors='replace').read()

def lag_row(nodes, x):
    n = len(nodes); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w
def interp(lo, hi): return np.array([lag_row(lo, x) for x in hi])            # (n_hi, n_lo)
def mass(nodes):                                                              # exact full mass matrix of the Lagrange basis
    xq, wq = np.polynomial.legendre.leggauss(len(nodes) + 2)
    L = np.array([lag_row(nodes, x) for x in xq])                             # (nq, n)
    return L.T @ (wq[:, None] * L)
def kron3(M): return np.kron(M, np.kron(M, M))                                # i fastest, same 1-D op on every axis

tables = {}
print("== 1-D solution points and weights (scheme_coef_cpr_gsp_gfp_hex.C) ==")
for p in range(6):
    sp, w = grab(gsp, f'sp_1d_p{p}'), grab(gsp, f'mean_1d_p{p}')
    gx, gw = np.polynomial.legendre.leggauss(p + 1)
    tables[f'sp_1d_p{p}'] = sp.tolist(); tables[f'mean_1d_p{p}'] = w.tolist()
    print(f"  p{p}: nodes {np.array2string(sp, precision=6)}  |nodes-GaussLegendre| {np.abs(np.sort(sp)-gx).max():.1e}  |weights-GL| {np.abs(np.sort(w)-np.sort(gw)).max():.1e}")

nodesets = {'gauss': lambda n: np.polynomial.legendre.leggauss(n)[0],
            'lobatto': lambda n: np.array([0.0]) if n == 1 else (np.array([-1.0, 1.0]) if n == 2 else np.concatenate(([-1.0], np.sort(np.polynomial.legendre.Legendre.basis(n - 1).deriv().roots()), [1.0]))),
            'equispaced': lambda n: np.linspace(-1, 1, n) if n > 1 else np.array([0.0])}

print("\n== hex prolongation p_inc_pN (p -> p+1) in dof_adapt_coef_hex.C: which node set? ==")
for p in range(0, 5):
    M = grab(dof, f'p_inc_p{p}'); tables[f'p_inc_p{p}'] = M.tolist()
    line = f"  p_inc_p{p} {M.shape}: "
    for name, ns in nodesets.items():
        lo, hi = ns(p + 1), ns(p + 2)
        cand = kron3(interp(lo, hi)) if p > 0 else np.ones((8, 1))
        line += f"{name} diff {np.abs(M - cand).max():.1e}  "
    print(line)

print("\n== hex restriction p_dec_pN (p -> p-1): which node set and rule? ==")
for p in range(1, 6):
    M = grab(dof, f'p_dec_p{p}'); tables[f'p_dec_p{p}'] = M.tolist()
    line = f"  p_dec_p{p} {M.shape}: "
    for name, ns in nodesets.items():
        lo, hi = ns(p), ns(p + 1)
        P = interp(lo, hi)
        cands = {'L2proj': np.linalg.solve(mass(lo), P.T @ mass(hi)),         # true L2 projection (full mass matrices)
                 'inject': interp(hi, lo),                                     # evaluate fine polynomial at coarse nodes
                 'lstsq': np.linalg.pinv(P)}
        if name == 'gauss':
            wl, wh = np.polynomial.legendre.leggauss(p)[1], np.polynomial.legendre.leggauss(p + 1)[1]
            cands['diagL2'] = (P.T * wh) / wl[:, None]                          # mp-MGARD rp_operators formula
        best = min(((np.abs(M - kron3(C)).max(), rule) for rule, C in cands.items()))
        line += f"{name}: best={best[1]} diff {best[0]:.1e}  "
    print(line)
    Pm = tables[f'p_inc_p{p-1}']
    print(f"      R*P - I (hpMusic p_dec_p{p} * p_inc_p{p-1}): {np.abs(M @ np.array(Pm) - np.eye(M.shape[0])).max():.1e}")


print("\n== Gauss-scheme tables in scheme_coef_cpr_gsp_gfp_hex.C (SchemeCoef::Prolong1D/Restrict1D/Prolong/Restrict) ==")
decl = re.findall(r'Real\s+((?:prolong|restrict)(?:_1d)?_p\d+)\s*((?:\[[^\]]*\])+)', gsp)
print("  declared:", ', '.join(f"{n}{d}" for n, d in decl))
for name, dims in decl:
    M = grab(gsp, name)
    if M is None: continue
    tables[f'gsp_{name}'] = M.tolist()
    p = int(name.rsplit('_p', 1)[1]); one_d = '_1d_' in name; prolong = name.startswith('prolong')
    gx = lambda n: np.polynomial.legendre.leggauss(n)[0]; gw = lambda n: np.polynomial.legendre.leggauss(n)[1]
    line = f"  {name} {M.shape}: "
    for lo_n, hi_n in ([(p + 1, p + 2), (p, p + 1)] if M.ndim == 2 else []):
        if lo_n < 1: continue
        P1 = interp(gx(lo_n), gx(hi_n)); R1 = (P1.T * gw(hi_n)) / gw(lo_n)[:, None]
        for tag, C1 in (('P', P1), ('R', R1)):
            for lab, C in (('1-D', C1), ('3-D kron', kron3(C1))):
                for tr, CC in (('', C), ('^T', C.T)):
                    if CC.shape == M.shape:
                        d = np.abs(M - CC).max()
                        if d < 1e-9: line += f"== Gauss {tag}{tr} {lab} (p{lo_n-1}<->p{hi_n-1}) diff {d:.1e}; "
    print(line if '==' in line else line + "no match with Gauss Lagrange-P / diag-L2-R (checked both orders, 1-D and kron)")
np.savez(f'{out}/hpmusic_hex_tables.npz', **{k: np.array(v) for k, v in tables.items()})
json.dump({k: v for k, v in tables.items() if k.startswith(('sp_1d', 'mean_1d'))}, open(f'{out}/hpmusic_hex_gauss_1d.json', 'w'), indent=1)
print(f"\nsaved {out}/hpmusic_hex_tables.npz (all arrays) and hpmusic_hex_gauss_1d.json (1-D nodes/weights)")
