#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_xhuff
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:12:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/xhuff_compute_%j.log
# Validates the MGARD-X Huffman stage (xhuff) on an MI250X compute node: HIP tests, cross-device probe,
# codec benchmark (in-house huffman|zstd vs MGARD-X Huffman, CPU vs HIP) on VKI p4 pressure step 0.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
export HPMGARD_XHUFF_VERIFY=${HPMGARD_XHUFF_VERIFY:-1}
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/gpu_compute_xhuff_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocminfo 2>/dev/null | grep -m1 -E "^\s+Name:\s+gfx"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && echo "== ctest HIP build, compress on hip"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed|\*\*\*"
echo "== test_codecs cross-device"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ./test_codecs 2>&1 | grep -E "xhuff device|checks"
echo "== probe"; srun -n1 -c56 --gpus=1 $G/probe/probe_hip 2>&1 | grep -v "^\[hpmgard" | head -40
B=$T/hpmgard_bench; R="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 16"
echo "== codec sweep p4 pressure 1e-3 (fixed p1 cellavg base)"
for dev in cpu hip; do for c in "huffman|zstd:3" "xhuff" "xhuff:3" "xhuff|zstd:3" "raw|zstd:3"; do
  srun -n1 -c56 --gpus=1 $B $R --device $dev --decode-device cpu --base-codec "$c" --residual-codec "$c" --label "p4P_${dev}_${c}" --csv $OUT > /dev/null 2>$G/compute/err_${dev}_${c//[|:]/_}.txt || echo "FAILED $dev $c"
done; done
echo "== hip->hip decode, verify off (speed)"; HPMGARD_XHUFF_VERIFY=0 srun -n1 -c56 --gpus=1 $B $R --device hip --decode-device hip --base-codec "xhuff:3" --residual-codec "xhuff:3" --label "p4P_hip_hipdec_noverify_xhuff:3" --csv $OUT > /dev/null 2>&1 || echo "FAILED hipdec"
echo "== MGARD-X direct on hip (reference)"; srun -n1 -c56 --gpus=1 $B $R --device hip --decode-device cpu --mgardx-direct --mgardx-device 1 --label "p4P_hip_mgardx_direct" --csv $OUT > /dev/null 2>&1 || echo "FAILED mgardx direct"
for eb in 1e-2 1e-4; do for c in "huffman|zstd:3" "xhuff:3"; do
  srun -n1 -c56 --gpus=1 $B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol $eb --threads 16 --device hip --decode-device cpu --base-codec "$c" --residual-codec "$c" --label "p4P_hip_${eb}_${c}" --csv $OUT > /dev/null 2>&1 || echo "FAILED $eb $c"
done; done
python3 - "$OUT" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':36s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'t_dec':>6s} {'t_code':>6s} {'t_res':>6s} {'t_base':>6s} mgardx_CR_sinf")
for r in rows:
    print(f"{r['label']:36s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_t_comp_s']):6.3f} {float(r['hpm_t_dec_s']):6.3f} {float(r['t_code']):6.3f} {float(r['t_residual']):6.3f} {float(r['t_base']):6.3f} {r.get('mgardx_CR_sinf','')}")
PY
echo "== done $(date)"
