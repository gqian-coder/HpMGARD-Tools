#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_gate_hip
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/gate_hip_%j.log
# Gate sweep on an MI250X: hpMGARD tuned (auto codec list, device-resident GPU path) vs MGARD-X HIP 1-D
# (s=inf and s=0 at the same absolute bound), VKI step 0, p4 all fields + p5 pressure, three tolerances.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/gate_hip_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
B=$T/hpmgard_bench
run() { srun -n1 -c56 --gpus=1 $B "$@" --csv $OUT > /dev/null 2>$G/compute/err_gate_$(echo "$*" | tr ' |:/' '____' | cut -c1-90).txt || echo "FAILED $*"; }
for eb in 1e-2 1e-3 1e-4; do
  for v in Density_aver Pressure_aver Rho_ms P_ms uu vv ww uv vw uw; do
    run --raw $D/p4_${v}_s0.bin --ncells 264640 --p 4 --tol $eb --tune --mgardx-direct --mgardx-device hip --device hip --decode-device hip --threads 16 --warmup --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --label "p4_${v}_${eb}"
  done
  run --raw $D/p4_Velocity_aver_s0.bin --ncells 264640 --p 4 --ncomp 3 --tol $eb --tune --mgardx-direct --mgardx-device hip --device hip --decode-device hip --threads 16 --warmup --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --label "p4_Velocity_aver_${eb}"
  [ -f $D/p5_Pressure_aver_s0.bin ] && run --raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --tol $eb --tune --mgardx-direct --mgardx-device hip --device hip --decode-device hip --threads 16 --warmup --label "p5_Pressure_aver_${eb}"
done
echo "== fixed in-house chain on hip, pressure, for reference"
for eb in 1e-2 1e-3 1e-4; do run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol $eb --device hip --decode-device hip --threads 16 --warmup --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "p4_Pressure_inhouse_${eb}"; done
python3 - "$OUT" <<'PY'
import csv,sys,math
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':30s} {'hpm_CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'t_dec':>6s} {'mgx_CR_sinf':>11s} {'mgx_err':>7s} {'mgx_t':>6s} {'mgx_CR_s0':>9s} {'ratio':>6s} codecs")
g={}
for r in rows:
    cr=float(r['hpm_CR']); mx=float(r['mgardx_CR_sinf']); ratio=cr/mx if mx>0 else float('nan')
    print(f"{r['label']:30s} {cr:8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_t_comp_s']):6.3f} {float(r['hpm_t_dec_s']):6.3f} {mx:11.1f} {float(r['mgardx_maxerr_over_tol_sinf']):7.3f} {float(r['mgardx_t_sinf']):6.3f} {float(r['mgardx_CR_s0']):9.1f} {ratio:6.2f} {r['base_codec']}/{r['residual_codec']}")
    eb=r['label'].rsplit('_',1)[-1]
    if mx>0 and not r['label'].startswith('p4_Pressure_inhouse'): g.setdefault(eb,[]).append(ratio)
for eb,v in g.items():
    gm=math.exp(sum(math.log(x) for x in v)/len(v)); print(f"geomean hpMGARD/MGARD-X(s=inf) CR ratio at {eb}: {gm:.2f} over {len(v)} fields")
PY
echo "== done $(date)"
