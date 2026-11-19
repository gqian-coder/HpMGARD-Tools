#!/usr/bin/env python3
"""Summarise a hpmgard_bench CSV (gate benchmark) as a markdown table.
usage: summarize_gate.py gate.csv [out.md]
Rows whose label contains 'codec_' are reported as the codec ablation; rows with 'thread' in the label as
the timing comparison; everything else as the ratio table (fields with tol_abs0 == 0, i.e. constant
fields, are listed but excluded from the geometric means).
"""
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else sys.stdout
f = lambda r, k: float(r[k])
def gmean(v): v = [x for x in v if x > 0]; return math.exp(sum(map(math.log, v)) / len(v)) if v else 0.0

main = [r for r in rows if 'codec_' not in r['label'] and 'thread' not in r['label']]
abl = [r for r in rows if 'codec_' in r['label']]
thr = [r for r in rows if 'thread' in r['label']]

print("## Ratio and time at equal pointwise bound (tol = eb x RMS, REL_L2)\n", file=out)
print("| field | eb | hpMGARD CR | MGARD-X s=inf CR | MGARD-X s=0 CR | hpMGARD / MGARD-X(s=inf) | max err / tol | hpMGARD s (16 thr, incl. tuning) | MGARD-X s (serial) | chosen base | codec |", file=out)
print("|---|---|---|---|---|---|---|---|---|---|---|", file=out)
for eb in sorted({r['tol'] for r in main}, key=float, reverse=True):
    sel = [r for r in main if r['tol'] == eb]
    for r in sel:
        const = f(r, 'tol_abs0') == 0
        ratio = f(r, 'hpm_CR') / f(r, 'mgardx_CR_sinf') if f(r, 'mgardx_CR_sinf') > 0 else 0
        print(f"| {r['label']}{' (constant field)' if const else ''} | {float(eb):g} | {f(r,'hpm_CR'):.0f} | {f(r,'mgardx_CR_sinf'):.1f} | {f(r,'mgardx_CR_s0'):.1f} | {ratio:.1f}x | {f(r,'hpm_maxerr_over_tol'):.3f} | {f(r,'hpm_t_comp_s'):.2f} | {f(r,'mgardx_t_sinf'):.2f} | p{r['base_order']} {r['base_method']} q={float(r['base_quantum']):g} | {r['residual_codec']} |", file=out)
    nz = [r for r in sel if f(r, 'tol_abs0') > 0]
    if nz:
        print(f"| **geomean (non-constant fields), eb={float(eb):g}** | | **{gmean([f(r,'hpm_CR') for r in nz]):.0f}** | **{gmean([f(r,'mgardx_CR_sinf') for r in nz]):.1f}** | **{gmean([f(r,'mgardx_CR_s0') for r in nz]):.1f}** | **{gmean([f(r,'hpm_CR')/f(r,'mgardx_CR_sinf') for r in nz]):.1f}x** | | {sum(f(r,'hpm_t_comp_s') for r in nz)/len(nz):.2f} (mean) | {sum(f(r,'mgardx_t_sinf') for r in nz)/len(nz):.2f} (mean) | | |", file=out)
if thr:
    print("\n## Timing, fixed configuration (p1 cellavg, huffman|zstd:3), p4 pressure, eb 1e-3\n", file=out)
    print("| run | threads | hpMGARD compress s | hpMGARD decompress s | CR | MGARD-X s=inf compress s (serial) | MGARD-X s=0 s |", file=out)
    print("|---|---|---|---|---|---|---|", file=out)
    for r in thr:
        print(f"| {r['label']} | {'1' if '1thread' in r['label'] else '16'} | {f(r,'hpm_t_comp_s'):.2f} | {f(r,'hpm_t_dec_s'):.2f} | {f(r,'hpm_CR'):.0f} | {f(r,'mgardx_t_sinf'):.2f} | {f(r,'mgardx_t_s0'):.2f} |", file=out)
if abl:
    print("\n## Codec ablation, p4 pressure, eb 1e-3, fixed hierarchy (p1 cellavg base, same chain for base and residual)\n", file=out)
    print("| chain | bytes | CR | compress s (16 thr) | decompress s |", file=out)
    print("|---|---|---|---|---|", file=out)
    for r in abl:
        print(f"| `{r['residual_codec']}` | {int(r['hpm_bytes']):,} | {f(r,'hpm_CR'):.0f} | {f(r,'hpm_t_comp_s'):.2f} | {f(r,'hpm_t_dec_s'):.2f} |", file=out)
