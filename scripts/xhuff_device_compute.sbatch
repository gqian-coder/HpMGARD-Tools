#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_xdev
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/xhuff_device_%j.log
# Device-resident MGARD-X lossless stages (xhuff/xhuff:3/xrle/xrans) vs the in-house chain on an MI250X,
# with --warmup so kernel loading and workspace allocation are excluded from the timed run.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/gpu_compute_xdev_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && echo "== ctest HIP build, compress on hip"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed|\*\*\*"
echo "== test_codecs cross-device"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ./test_codecs 2>&1 | grep -E "device 1|checks"
B=$T/hpmgard_bench; R="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 16 --warmup"
echo "== codec sweep p4 pressure 1e-3 (fixed p1 cellavg base), warm-up excluded"
for dev in hip cpu; do for c in "huffman|zstd:3" "raw|zstd:3" "xhuff" "xhuff:3" "xhuff|zstd:3" "xrle" "xrle|zstd:3" "xrans" "xrans|zstd:3"; do
  srun -n1 -c56 --gpus=1 $B $R --device $dev --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "$c" --label "res_${c}_${dev}" --csv $OUT > /dev/null 2>$G/compute/err_xdev_${dev}_${c//[|:]/_}.txt || echo "FAILED $dev $c"
done; done
echo "== verify off (speed of the device path without the decode check)"
for c in "xhuff" "xhuff|zstd:3" "xrle" "xrans"; do
  HPMGARD_XHUFF_VERIFY=0 srun -n1 -c56 --gpus=1 $B $R --device hip --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "$c" --label "res_${c}_hip_noverify" --csv $OUT > /dev/null 2>&1 || echo "FAILED noverify $c"
done
echo "== other tolerances, hip"
for eb in 1e-2 1e-4; do for c in "huffman|zstd:3" "xhuff|zstd:3" "xrle" "xrans"; do
  srun -n1 -c56 --gpus=1 $B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol $eb --threads 16 --warmup --device hip --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "$c" --label "res_${c}_hip_${eb}" --csv $OUT > /dev/null 2>&1 || echo "FAILED $eb $c"
done; done
python3 - "$OUT" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':34s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'t_dec':>6s} {'t_code':>6s} {'t_res':>6s} {'t_base':>6s}")
for r in rows:
    print(f"{r['label']:34s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_t_comp_s']):6.3f} {float(r['hpm_t_dec_s']):6.3f} {float(r['t_code']):6.3f} {float(r['t_residual']):6.3f} {float(r['t_base']):6.3f}")
PY
echo "== done $(date)"
