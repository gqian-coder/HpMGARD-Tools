#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_gpupath
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/gpu_path_%j.log
# GPU path after: auto codec (xhuff|zstd:3 on hip), no host round trip of the residual symbols, device decode
# into the reconstruction, pinned host scratch, tensor-sweep kernels. Compared with the in-house chain and
# with the CPU backend; decode on hip and on cpu; cross-backend decode.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/gpu_path_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && echo "== ctest HIP build, compress on hip (+ decode on hip inside test_compressor)"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed|\*\*\*"
HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ./test_compressor 2>&1 | tail -1
B=$T/hpmgard_bench; P4="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --threads 16 --warmup"
run() { srun -n1 -c56 --gpus=1 $B $P4 "$@" --csv $OUT > /dev/null 2>$G/compute/err_gpupath_$(echo "$*" | tr ' |:/' '____' | cut -c1-80).txt || echo "FAILED $*"; }
echo "== p4 pressure, three tolerances, hip auto (device path) vs hip in-house vs cpu, decode on cpu and on hip"
for eb in 1e-2 1e-3 1e-4; do
  run --tol $eb --device hip --decode-device cpu --label "hip_auto_deccpu_$eb"
  run --tol $eb --device hip --decode-device hip --label "hip_auto_dechip_$eb"
  run --tol $eb --device hip --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "hip_inhouse_deccpu_$eb"
  run --tol $eb --device cpu --decode-device cpu --label "cpu_auto_$eb"
  run --tol $eb --device cpu --decode-device hip --label "cpu_auto_dechip_$eb"
done
echo "== tuned (auto tune codecs), hip and cpu, 1e-3"
run --tol 1e-3 --device hip --decode-device hip --tune --label "hip_tuned_1e-3"
run --tol 1e-3 --device cpu --decode-device cpu --tune --label "cpu_tuned_1e-3"
echo "== 3-component velocity, hip auto, 1e-3"
srun -n1 -c56 --gpus=1 $B --raw $D/p4_Velocity_aver_s0.bin --ncells 264640 --p 4 --ncomp 3 --tol 1e-3 --threads 16 --warmup --device hip --decode-device hip --label "hip_auto_velocity_1e-3" --csv $OUT > /dev/null 2>&1 || echo "FAILED velocity (dump may be missing)"
python3 - "$OUT" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':28s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'t_dec':>6s} {'t_stat':>6s} {'t_refa':>6s} {'t_base':>6s} {'t_res':>6s} {'t_code':>6s} {'t_tune':>6s} codecs")
for r in rows:
    print(f"{r['label']:28s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_t_comp_s']):6.3f} {float(r['hpm_t_dec_s']):6.3f} {float(r['t_stats']):6.3f} {float(r['t_refactor']):6.3f} {float(r['t_base']):6.3f} {float(r['t_residual']):6.3f} {float(r['t_code']):6.3f} {float(r['hpm_t_tune_s']):6.3f} {r['base_codec']}/{r['residual_codec']}")
PY
echo "== done $(date)"
