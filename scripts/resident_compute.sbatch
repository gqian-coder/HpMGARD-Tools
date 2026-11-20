#!/bin/bash
#SBATCH -A cfd164
#SBATCH -J hpmgard_resident
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -o /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/compute/resident_%j.log
# Block-resident GPU path (no host extraction copy: one chunked, pipelined upload of the interleaved block per call,
# strided kernels, one pipelined download into the caller's buffer, DecompressInto): fixed chains, no tuning,
# breakdown incl. upload/download and GB/s; decode on the GPU; dedup; velocity; p5; copy-thread sweep.
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
G=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate
OUT=$G/compute/resident_${SLURM_JOB_ID}.csv
echo "== node $(hostname) $(date)"; rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series"
cd $T && HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ctest 2>&1 | grep -E "tests passed|Failed"; HPMGARD_TEST_DEVICE=hip srun -n1 -c56 --gpus=1 ./test_compressor 2>&1 | tail -1
B=$T/hpmgard_bench
run() { srun -n1 -c56 --gpus=1 $B "$@" --csv $OUT > /dev/null 2>$G/compute/err_resident_$(echo "$*" | tr ' |:/' '____' | cut -c1-90).txt || echo "FAILED $*"; }
P4="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --threads 16 --warmup"
V4="--raw $D/p4_Velocity_aver_s0.bin --ncells 264640 --p 4 --ncomp 3 --threads 16 --warmup"
P5="--raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --threads 16 --warmup"
MESH="--conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin"
echo "== pressure, three tolerances, hip: frbp|zstd:3, xhuff|zstd:3, huffman|zstd:3, frbp; cpu reference"
for eb in 1e-2 1e-3 1e-4; do
  for c in "frbp|zstd:3" "xhuff|zstd:3" "huffman|zstd:3" "frbp"; do
    run $P4 --tol $eb --device hip --decode-device hip --base-codec "$c" --residual-codec "$c" --label "P_hip_${c}_$eb"
  done
  run $P4 --tol $eb --device cpu --decode-device cpu --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_cpu_inhouse_$eb"
  run $P4 --tol $eb --device hip --decode-device cpu --base-codec "frbp|zstd:3" --residual-codec "frbp|zstd:3" --label "P_hip_frbp_deccpu_$eb"
done
echo "== velocity and p5, hip"
for c in "frbp|zstd:3" "xhuff|zstd:3"; do
  run $V4 --tol 1e-3 --device hip --decode-device hip --base-codec "$c" --residual-codec "$c" --label "V_hip_${c}_1e-3"
  run $P5 --tol 1e-3 --device hip --decode-device hip --base-codec "$c" --residual-codec "$c" --label "P5_hip_${c}_1e-3"
done
echo "== dedup base (fused dedup kernels), hip"
for c in "frbp|zstd:3" "xhuff|zstd:3" "huffman|zstd:3"; do
  run $P4 --tol 1e-3 --device hip --decode-device hip --base-method dedup $MESH --base-codec "$c" --residual-codec "$c" --label "P_hip_dedup_${c}_1e-3"
done
run $P4 --tol 1e-3 --device cpu --decode-device cpu --base-method dedup $MESH --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --label "P_cpu_dedup_inhouse_1e-3"
echo "== copy-thread sweep (upload/download staging memcpy), pressure and velocity, frbp|zstd:3"
for th in 8 32 56; do
  run $P4 --tol 1e-3 --device hip --decode-device hip --base-codec "frbp|zstd:3" --residual-codec "frbp|zstd:3" --threads $th --label "P_hip_frbp_th${th}_1e-3"
  run $V4 --tol 1e-3 --device hip --decode-device hip --base-codec "frbp|zstd:3" --residual-codec "frbp|zstd:3" --threads $th --label "V_hip_frbp_th${th}_1e-3"
done
echo "== tuned, hip, 1e-3"
run $P4 --tol 1e-3 --device hip --decode-device hip --tune --tune-codecs "frbp|zstd:3,xhuff|zstd:3,huffman|zstd:3,raw|zstd:3" --label "P_hip_tuned_1e-3"
python3 - "$OUT" <<'PY'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no csv", e); sys.exit()
print(f"{'label':30s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'GB/s':>5s} {'t_dec':>6s} {'dec GB/s':>8s} {'t_stat':>6s} {'t_fused':>7s} {'t_code':>6s} {'t_extr':>6s} {'t_upl':>6s} {'t_ddec':>6s} {'t_down':>6s} {'t_ins':>6s} codecs")
for r in rows:
    raw=float(r['raw_bytes']); tc=float(r['hpm_t_comp_s']); td=float(r['hpm_t_dec_s'])
    print(f"{r['label']:30s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {tc:6.3f} {raw/tc/1e9:5.2f} {td:6.3f} {raw/td/1e9:8.2f} {float(r['t_stats']):6.3f} {float(r['t_refactor']):7.3f} {float(r['t_code']):6.3f} {float(r['t_extract']):6.3f} {float(r['t_upload']):6.3f} {float(r['t_dec_decode']):6.3f} {float(r['t_dec_download']):6.3f} {float(r['t_dec_insert']):6.3f} {r['base_codec']}/{r['residual_codec']}")
PY
echo "== done $(date)"
