#!/bin/bash
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null; module unload cray-libsci darshan-runtime craype-accel-amd-gfx90a 2>/dev/null
P=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/probe
INST=/lustre/orion/cfd164/proj-shared/gongq/HODG/CompressHpMGARDOperator/install-hip; MG=/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD/install-hip-frontier.mgard17; ROCM=/opt/rocm-7.2.0
cd $P && CC -O2 -std=c++17 -fopenmp probe_hip.cpp -I$INST/include -I$MG/include -L$INST/lib64 -lhpmgard -lhpmgard_xhuff_serial -L$MG/lib64 -lmgard -lprotobuf -L$MG/lib -lzstd -L$ROCM/lib -lamdhip64 -Wl,-rpath,$MG/lib64 -Wl,-rpath,$MG/lib -Wl,-rpath,$ROCM/lib -Wl,--disable-new-dtags -o probe_hip 2>&1 | grep -v warning | grep -i "error" | head -5
echo "== probe on $(hostname)"; ./probe_hip 2>&1 | grep -v "^\[hpmgard xhuff\] MGARD-X lossless compress failed: Cannot" | head -60
