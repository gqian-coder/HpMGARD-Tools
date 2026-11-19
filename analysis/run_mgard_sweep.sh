#!/bin/bash
# MGARD-X SERIAL sweep on ho_dg_p4_sol/sol_1_aver.bp step-0 Pressure_aver (and Density_aver), login node.
D=/tmp/claude-16377/-lustre-orion-cfd164-proj-shared-gongq/c5fb0353-44df-4de7-93c3-e0d05103b213/scratchpad/hodg
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/tools/mgard_test
OUT=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/mgard_sweep_p4.csv
echo "label,shape,eb,s,levels,n,tol_abs,range,comp_bytes,CR,max_err,rmse,rmse_over_range,t_comp_s,t_decomp_s" > $OUT
jobs=()
for var in Pressure_aver Density_aver; do
  for eb in 5e-3 1e-3; do
    for s in inf 0; do
      for shp in "1d:33080000" "2d:264640,125" "4d:264640,5,5,5"; do
        name=${shp%%:*}; shape=${shp#*:}
        jobs+=("$T $D/p4_${var}_s0.bin $shape $eb $s default ${var}_${name}")
      done
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 8 -I CMD bash -c 'CMD >> '"$OUT"' 2>> '"$OUT.err"
echo DONE >> $OUT.err
