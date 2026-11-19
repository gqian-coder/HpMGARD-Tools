#!/usr/bin/env python3
"""Compression-only emulation vs MGARD-X at the same absolute pointwise bound (tol = eb * RMS, REL_L2).
usage: bench_compression_only.py <dumpdir> <var> <p> <eb1,eb2,...> <out.csv>
  var: p4 field name (Density_aver, Pressure_aver, uu, ..., U_aver, V_aver, W_aver) or p5b0_Pressure_aver
Scheme: p1 Gauss L2 projection -> corner-represented base coded by MGARD-X 1-D (s=inf, tol_b = 0.2 tol)
        + residual quantised with quantum 2*0.8*tol, coded by zstd-19 / zlib-9 / FRBP-estimate.
Cell-mean variant: base = cell means (MGARD-X 1-D, 0.2 tol) + residual (0.8 tol).
Direct MGARD-X: 1-D flattened, s=inf (pointwise) and s=0 (L2) at tol.
"""
import sys, os, time, subprocess, zlib, numpy as np
d, var, p, ebs, out = sys.argv[1].rstrip('/'), sys.argv[2], int(sys.argv[3]), [float(x) for x in sys.argv[4].split(',')], sys.argv[5]
T = '/lustre/orion/cfd164/proj-shared/gongq/HODG/tools/mgard_test2'
n1 = p + 1
# ---- load field as [cell][k][j][i] ----
if var in ('U_aver', 'V_aver', 'W_aver'):
    c = 'UVW'.index(var[0]); vel = np.fromfile(f'{d}/p4_Velocity_aver_s0.bin').reshape(-1, 125, 3)
    u = np.ascontiguousarray(vel[:, :, c]).reshape(-1, 5, 5, 5); del vel; tag = f'p4_{var}'
elif var == 'p5b0_Pressure_aver':
    u = np.fromfile(f'{d}/p5_Pressure_aver_s0.bin')[:33080 * 216].reshape(33080, 6, 6, 6); tag = var
else:
    u = np.fromfile(f'{d}/p4_{var}_s0.bin').reshape(-1, 5, 5, 5); tag = f'p4_{var}'
NC = u.shape[0]; n = u.size; raw_bytes = n * 8
rawfile = f'{d}/bench_{tag}.bin'
if not os.path.exists(rawfile): u.ravel().tofile(rawfile)

gauss = {1: np.polynomial.legendre.leggauss(2)[0], p: np.polynomial.legendre.leggauss(n1)[0]}
def lag_row(nodes, x):
    m = len(nodes); w = np.ones(m)
    for j in range(m):
        for k in range(m):
            if k != j: w[j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return w
def weights(nodes):
    m = len(nodes); xs, ws = np.polynomial.legendre.leggauss(m + 2)
    return np.array([sum(wq * lag_row(nodes, xq)[j] for xq, wq in zip(xs, ws)) for j in range(m)])
P1 = np.array([lag_row(gauss[1], x) for x in gauss[p]]); R = (P1.T * weights(gauss[p])) / weights(gauss[1])[:, None]
E = np.array([lag_row(gauss[1], -1.0), lag_row(gauss[1], +1.0)]); Pc = np.array([lag_row(np.array([-1.0, 1.0]), x) for x in gauss[p]])

def mgard(arr, abs_tol, s, tagx, want_dec):
    fn = f'{d}/bench_{tag}_{tagx}.bin'; dec = f'{d}/bench_{tag}_{tagx}_dec.bin' if want_dec else '-'
    if fn != rawfile: arr.astype(np.float64).ravel().tofile(fn)
    r = subprocess.run([T, fn, str(arr.size), '0', s, 'default', tagx, dec, f'{abs_tol:.17g}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0: raise RuntimeError(r.stderr[-500:])
    f = r.stdout.strip().split(','); cb = int(f[8]); mx = float(f[10]); rm = float(f[11]); tc = float(f[13]); td = float(f[14])
    decoded = np.fromfile(dec).reshape(arr.shape) if want_dec else None
    if fn != rawfile: os.remove(fn)
    if want_dec: os.remove(dec)
    return cb, mx, rm, tc, td, decoded
def zstd_bytes(b):
    r = subprocess.run(['zstd', '-19', '-q', '-c', '-T4'], input=b, stdout=subprocess.PIPE); return len(r.stdout)
def frbp_bytes(k):
    a = np.abs(k).ravel().astype(np.int64); m = (a.size + 31) // 32; pad = np.zeros(m * 32, dtype=np.int64); pad[:a.size] = a
    bm = pad.reshape(m, 32).max(1); rate = np.where(bm > 0, np.floor(np.log2(np.maximum(bm, 1))).astype(int) + 1, 0)
    return int(m + np.sum(np.where(rate > 0, 4 + 4 * rate, 0)))   # rate byte per block + sign word + rate planes

rms = float(np.sqrt(np.mean(u * u))); rng = float(u.max() - u.min())
rows = []
for eb in ebs:
    tol = eb * rms
    # direct MGARD-X
    cb_i, mx_i, rm_i, tc_i, td_i, _ = mgard(u, tol, 'inf', 'direct_inf', False)
    cb_0, mx_0, rm_0, tc_0, td_0, _ = mgard(u, tol, '0', 'direct_s0', False)
    # scheme: p1 corner base
    t0 = time.time()
    bg = np.einsum('xk,yj,zi,ckji->cxyz', R, R, R, u); bc = np.einsum('ax,by,cz,ixyz->iabc', E, E, E, bg); del bg
    t_ref = time.time() - t0
    cb_b, _, _, tc_b, td_b, bdec = mgard(bc, 0.2 * tol, 'inf', 'p1corner', True)
    t1 = time.time()
    rec = np.einsum('kx,jy,iz,cxyz->ckji', Pc, Pc, Pc, bdec); res = u - rec; q = 2 * 0.8 * tol
    k = np.rint(res / q).astype(np.int32); recon = rec + k * q
    err = np.abs(recon - u); mx_s = float(err.max()); rm_s = float(np.sqrt(np.mean(err * err)))
    t_quant = time.time() - t1
    kb = k.tobytes(); t2 = time.time(); z = zstd_bytes(kb); t_zstd = time.time() - t2; zl = len(zlib.compress(kb, 9)); fr = frbp_bytes(k)
    del rec, res, recon, err
    # cell-mean variant
    means = u.reshape(NC, -1).mean(1)
    cb_m, _, _, tc_m, _, mdec = mgard(means, 0.2 * tol, 'inf', 'cellmean', True)
    resm = u - mdec[:, None, None, None]; km = np.rint(resm / q).astype(np.int32); errm = np.abs(resm - km * q)
    mx_m = float(errm.max()); rm_m = float(np.sqrt(np.mean(errm * errm))); zm = zstd_bytes(km.tobytes()); frm = frbp_bytes(km); del resm, errm
    rows.append(dict(var=tag, p=p, n=n, rms=rms, range=rng, eb_rel_l2=eb, tol_abs=tol,
        CR_mgard_sinf=raw_bytes / cb_i, maxerr_sinf=mx_i, rmse_sinf=rm_i, t_mgard_sinf=tc_i, t_mgard_sinf_dec=td_i,
        CR_mgard_s0=raw_bytes / cb_0, maxerr_s0=mx_0, rmse_s0=rm_0, t_mgard_s0=tc_0,
        CR_p1c_zstd=raw_bytes / (cb_b + z), CR_p1c_zlib=raw_bytes / (cb_b + zl), CR_p1c_frbp=raw_bytes / (cb_b + fr),
        base_bytes=cb_b, res_zstd=z, res_frbp=fr, maxerr_p1c=mx_s, rmse_p1c=rm_s,
        t_scheme_refactor=t_ref, t_scheme_base_mgard=tc_b, t_scheme_quant=t_quant, t_scheme_zstd=t_zstd,
        CR_mean_zstd=raw_bytes / (cb_m + zm), CR_mean_frbp=raw_bytes / (cb_m + frm), maxerr_mean=mx_m, rmse_mean=rm_m))
hdr = list(rows[0].keys())
new = not os.path.exists(out)
with open(out, 'a') as f:
    if new: f.write(','.join(hdr) + '\n')
    for r in rows: f.write(','.join(f'{r[h]:.6g}' if isinstance(r[h], float) else str(r[h]) for h in hdr) + '\n')
print(tag, 'done')
