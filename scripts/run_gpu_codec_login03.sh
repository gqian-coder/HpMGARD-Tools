#!/bin/bash
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_hip
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
OUT=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/gpu_login03_xhuff.csv
echo "== host $(hostname)  $(rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' | sed 's/.*: //')"
cd $T && echo "== ctest (HIP build, compress on hip)"; HPMGARD_TEST_DEVICE=hip ctest 2>&1 | grep -E "tests passed|Failed|\*\*\*"; HPMGARD_TEST_DEVICE=hip ./test_codecs 2>&1 | grep -E "xhuff device|checks"
rm -f $OUT
B=$T/hpmgard_bench; R="--raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 16"
echo "== fixed hierarchy (p1 cellavg base), codec sweep, CPU 16 threads vs HIP"
for dev in cpu hip; do
  for c in "huffman|zstd:3" "xhuff" "xhuff:3" "xhuff|zstd:3" "raw|zstd:3"; do
    $B $R --device $dev --decode-device cpu --base-codec "$c" --residual-codec "$c" --label "p4P_${dev}_${c}" --csv $OUT > /dev/null 2>&1 || echo "FAILED $dev $c"
  done
done
echo "== hip compress -> hip decode, xhuff:3"
$B $R --device hip --decode-device hip --base-codec "xhuff:3" --residual-codec "xhuff:3" --label "p4P_hip_hipdec_xhuff:3" --csv $OUT > /dev/null 2>&1 || echo "FAILED hipdec"
echo "== tol 1e-2 and 1e-4, hip, huffman|zstd:3 vs xhuff:3"
for eb in 1e-2 1e-4; do for c in "huffman|zstd:3" "xhuff:3"; do
  $B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol $eb --threads 16 --device hip --decode-device cpu --base-codec "$c" --residual-codec "$c" --label "p4P_hip_${eb}_${c}" --csv $OUT > /dev/null 2>&1 || echo "FAILED $eb $c"
done; done
python3 - "$OUT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
print(f"{'label':34s} {'bytes':>9s} {'CR':>8s} {'err/tol':>7s} {'t_comp':>6s} {'t_dec':>6s} {'t_code':>6s} {'t_res':>6s} {'t_base':>6s}")
for r in rows:
    print(f"{r['label']:34s} {int(r['hpm_bytes']):9d} {float(r['hpm_CR']):8.1f} {float(r['hpm_maxerr_over_tol']):7.3f} {float(r['hpm_t_comp_s']):6.3f} {float(r['hpm_t_dec_s']):6.3f} {float(r['t_code']):6.3f} {float(r['t_residual']):6.3f} {float(r['t_base']):6.3f}")
PY
echo XHUFFDONE
