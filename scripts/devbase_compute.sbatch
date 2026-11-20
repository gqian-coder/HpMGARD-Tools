#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_devbase
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/devbase_%j.log
# Device-resident base path (restrict, split, quantise, decode on the GPU; base streams coded from device memory):
# fixed chains without tuning, stage breakdown and GB/s; dedup base on the GPU; MGARD-X Huffman block size sweep
# (HPMGARD_XHUFF_BLOCK 1024 / 4096 / 20480) for compress and GPU decode time and size.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/devbase_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ./test_compressor 2>&1 | tail -1
B=$T/hpmgard_bench
run() { srun -n1 -c56 --gpus=1 $B "$@" --csv $OUT > /dev/null 2>$G/compute/err_devbase_$(echo "$*" | tr ' |:/' '____' | cut -c1-90).txt || echo "FAILED $*"; }
P4="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --threads 16 --warmup"
V4="--raw $D/p4_Velocity_aver_s0.bin --ncells 264640 --p 4 --ncomp 3 --threads 16 --warmup"
P5="--raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --threads 16 --warmup"
MESH="--conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin"
echo "== fixed chains, no tuning, hip (device base path) and cpu"
for eb in 1e-2 1e-3 1e-4; do
  run $P4 --tol $eb --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "P_hip_xhuff_$eb"
  run $P4 --tol $eb --device hip --decode-device hip --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_hip_inhouse_$eb"
  run $P4 --tol $eb --device cpu --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_cpu_inhouse_$eb"
done
run $V4 --tol 1e-3 --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "V_hip_xhuff_1e-3"
run $V4 --tol 1e-3 --device hip --decode-device hip --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "V_hip_inhouse_1e-3"
run $P5 --tol 1e-3 --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "P5_hip_xhuff_1e-3"
echo "== dedup base on the GPU (device dedup split), warm"
run $P4 --tol 1e-3 --device hip --decode-device hip --base-method dedup $MESH --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "P_hip_dedup_xhuff_1e-3"
run $P4 --tol 1e-3 --device hip --decode-device hip --base-method dedup $MESH --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_hip_dedup_inhouse_1e-3"
run $P4 --tol 1e-3 --device cpu --decode-device cpu --base-method dedup $MESH --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_cpu_dedup_inhouse_1e-3"
echo "== MGARD-X Huffman block size sweep (xhuff|zstd:3, hip compress + hip decode)"
for blk in 1024 4096 20480; do
  HPMGARD_XHUFF_BLOCK=$blk run $P4 --tol 1e-3 --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "P_blk${blk}_1e-3"
  HPMGARD_XHUFF_BLOCK=$blk run $P4 --tol 1e-4 --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "P_blk${blk}_1e-4"
  HPMGARD_XHUFF_BLOCK=$blk run $V4 --tol 1e-3 --device hip --decode-device hip --base-codec "xhuff|zstd:3" --residual-codec "xhuff|zstd:3" --label "V_blk${blk}_1e-3"
done
python3 - "$OUT" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':26s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'GB/s':>5s} {'t_dec':>6s} {'dec GB/s':>8s} {'t_stat':>6s} {'t_refa':>6s} {'t_base':>6s} {'t_res':>6s} {'t_code':>6s} {'t_geom':>6s}")
for r in rows:
    raw=float(r['raw_bytes']); tc=float(r['hpm_t_comp_s']); td=float(r['hpm_t_dec_s'])
    print(f"{r['label']:26s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {tc:6.3f} {raw/tc/1e9:5.2f} {td:6.3f} {raw/td/1e9:8.2f} {float(r['t_stats']):6.3f} {float(r['t_refactor']):6.3f} {float(r['t_base']):6.3f} {float(r['t_residual']):6.3f} {float(r['t_code']):6.3f} {float(r['t_geom']):6.3f}")
PY
echo "== done $(date)"
