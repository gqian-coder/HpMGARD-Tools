#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_gate2
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/gate_hip2_%j.log
# Gate v2 on an MI250X: per field and tolerance, hpMGARD with (a) fixed xhuff|zstd:3, (b) fixed huffman|zstd:3,
# (c) tuned; MGARD-X HIP s=inf and s=0 (L2) with achieved RMSE on every row; throughput; dedup geometry cache
# (cold vs warm); zstd worker count. Data: ho_dg_p4_sol step 0 (10 scalars + velocity) and ho_dg_p5_8np_sol pressure.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/gate_hip2_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed"
B=$T/hpmgard_bench
run() { srun -n1 -c56 --gpus=1 $B "$@" --csv $OUT > /dev/null 2>$G/compute/err_gate2_$(echo "$*" | tr ' |:/' '____' | cut -c1-90).txt || echo "FAILED $*"; }
COMMON="--device hip --decode-device hip --threads 16 --warmup --mgardx-direct --mgardx-device hip --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin"
for eb in 1e-2 1e-3 1e-4; do
  for v in Density_aver Pressure_aver Rho_ms P_ms Velocity_aver; do
    NC=1; [ $v = Velocity_aver ] && NC=3
    run --raw $D/p4_${v}_s0.bin --ncells 264640 --p 4 --ncomp $NC --tol $eb $COMMON --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "p4_${v}_${eb}_xhuff"
    run --raw $D/p4_${v}_s0.bin --ncells 264640 --p 4 --ncomp $NC --tol $eb $COMMON --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "p4_${v}_${eb}_inhouse"
    run --raw $D/p4_${v}_s0.bin --ncells 264640 --p 4 --ncomp $NC --tol $eb $COMMON --tune --label "p4_${v}_${eb}_tuned"
  done
  run --raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --tol $eb --device hip --decode-device hip --threads 16 --warmup --mgardx-direct --mgardx-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "p5_Pressure_aver_${eb}_xhuff"
  run --raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --tol $eb --device hip --decode-device hip --threads 16 --warmup --mgardx-direct --mgardx-device hip --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "p5_Pressure_aver_${eb}_inhouse"
  run --raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --tol $eb --device hip --decode-device hip --threads 16 --warmup --mgardx-direct --mgardx-device hip --tune --label "p5_Pressure_aver_${eb}_tuned"
done
echo "== dedup geometry cache: cold (no warm-up) vs warm, pressure 1e-3, dedup base"
run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --device hip --decode-device hip --threads 16 --base-method dedup --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --label "dedup_cold"
run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --device hip --decode-device hip --threads 16 --warmup --base-method dedup --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --label "dedup_warm"
echo "== zstd workers: 1 vs default (8) vs 16, pressure 1e-3, both chains on hip and the in-house chain on cpu"
for w in 1 8 16; do
  HPMGARD_ZSTD_WORKERS=$w run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --device hip --decode-device hip --threads 16 --warmup --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "zstdw${w}_hip_xhuff"
  HPMGARD_ZSTD_WORKERS=$w run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --device hip --decode-device hip --threads 16 --warmup --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "zstdw${w}_hip_inhouse"
  HPMGARD_ZSTD_WORKERS=$w run --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --device cpu --decode-device cpu --threads 16 --warmup --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "zstdw${w}_cpu_inhouse"
done
python3 - "$OUT" <<'PY'
import csv,sys,math
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':32s} {'CR':>8s} {'err/tol':>7s} {'rmse':>9s} {'t_comp':>6s} {'GB/s':>5s} {'t_dec':>6s} {'t_code':>6s} {'t_geom':>6s} {'mgx_CR_inf':>10s} {'mgx_rmse_inf':>12s} {'mgx_CR_s0':>9s} {'mgx_err_s0':>10s} {'mgx_rmse_s0':>11s} {'mgx_t':>6s} codecs")
for r in rows:
    raw=float(r['raw_bytes']); tc=float(r['hpm_t_comp_s'])
    print(f"{r['label']:32s} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_rmse']):9.3g} {tc:6.3f} {raw/tc/1e9:5.2f} {float(r['hpm_t_dec_s']):6.3f} {float(r['t_code']):6.3f} {float(r['t_geom']):6.3f} {float(r['mgardx_CR_sinf']):10.1f} {float(r['mgardx_rmse_sinf']):12.3g} {float(r['mgardx_CR_s0']):9.1f} {float(r['mgardx_maxerr_over_tol_s0']):10.3f} {float(r['mgardx_rmse_s0']):11.3g} {float(r['mgardx_t_sinf']):6.3f} {r['base_codec']}/{r['residual_codec']}")
PY
echo "== done $(date)"
