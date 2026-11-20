#!/bin/bash
# Regenerates the step-0 raw dumps used by the gate sweep (the scratchpad copies were cleaned).
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module unload cray-libsci darshan-runtime 2>/dev/null
T=/lustre/orion/cfd164/proj-shared/gongq/HODG/tools/dumpbp
D=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/dumps
P4=/lustre/orion/proj-shared/cfd164/vki_hodg/ho_dg_p4_sol/sol_1_aver.bp
P5=/lustre/orion/proj-shared/cfd164/vki_hodg/ho_dg_p5_8np_sol/sol_1_aver.bp
for v in Density_aver Rho_ms P_ms uu vv ww uv vw uw Velocity_aver; do
  [ -f $D/p4_${v}_s0.bin ] || $T $P4 vtkDGHex/$v 0 $D/p4_${v}_s0.bin > /dev/null 2>&1 || echo "FAILED $v"
done
PV=$(/lustre/orion/cfd164/proj-shared/gongq/Software/ADIOS2/install-adios-serial/bin/bpls $P5 2>/dev/null | grep -i "pressure_aver" | awk '{print $2}' | head -1)
echo "p5 pressure variable: $PV"
[ -n "$PV" ] && { [ -f $D/p5_Pressure_aver_s0.bin ] || $T $P5 $PV 0 $D/p5_Pressure_aver_s0.bin > /dev/null 2>&1 || echo "FAILED p5"; }
ls -la $D/*.bin | awk '{print $5, $9}'; cat $D/p5_Pressure_aver_s0.bin.blocks 2>/dev/null | head -3
