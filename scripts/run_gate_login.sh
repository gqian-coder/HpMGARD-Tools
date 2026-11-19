#!/bin/bash
# M1 gate benchmark: hpMGARD (tuned, 16 threads) vs MGARD-X 1-D (serial) at equal pointwise bound, VKI step 0.
D=/tmp/claude-16377/-lustre-orion-cfd164-proj-shared-gongq/c5fb0353-44df-4de7-93c3-e0d05103b213/scratchpad/hodg
B=/lustre/orion/cfd164/proj-shared/gongq/HODG/HpMGARD-Tools/build_cpu/hpmgard_bench
OUT=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/gate_p4.csv
rm -f $OUT
for eb in 1e-2 1e-3 1e-4; do
  for v in Density_aver Pressure_aver Rho_ms P_ms uu vv ww uv vw uw; do
    $B --raw $D/p4_${v}_s0.bin --ncells 264640 --p 4 --tol $eb --tune --mgardx-direct --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --threads 16 --label p4_${v} --csv $OUT || echo "FAILED $v $eb"
  done
  $B --raw $D/p4_Velocity_aver_s0.bin --ncells 264640 --p 4 --ncomp 3 --tol $eb --tune --mgardx-direct --conn $D/p4_conn_s0.bin --points $D/p4_points_s0.bin --threads 16 --label p4_Velocity_aver --csv $OUT || echo "FAILED Velocity $eb"
  $B --raw $D/p5_Pressure_aver_s0.bin --ncells 264640 --p 5 --tol $eb --tune --mgardx-direct --threads 16 --label p5_Pressure_aver_8blocks --csv $OUT || echo "FAILED p5 $eb"
done
# single-thread timing, fixed configuration (no tuner), pressure, 1e-3
$B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 1 --label p4_Pressure_aver_1thread_fixed --csv $OUT
$B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 16 --label p4_Pressure_aver_16threads_fixed --csv $OUT
# codec ablation on pressure at 1e-3 (fixed hierarchy)
for c in "huffman|zstd:3" "huffman|zstd:19" "huffman" "frbp|zstd:3" "frbp|zstd:19" "frbp" "raw|zstd:3" "raw|zstd:19"; do
  $B --raw $D/p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --threads 16 --base-codec "$c" --residual-codec "$c" --label "p4_Pressure_codec_${c}" --csv $OUT
done
echo GATEDONE
