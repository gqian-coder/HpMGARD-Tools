#!/bin/bash
bash /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/probe/run_probe_login03.sh > /dev/null 2>&1   # rebuilds probe_hip
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null
cd /lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/probe
export HPMGARD_XHUFF_VERIFY=0
for kind in 1 2; do for lvl in 0 3; do for mode in A B C D; do
  timeout 120 ./probe_hip $kind $lvl $mode 2>&1 | grep -v "^\[hpmgard xhuff\] MGARD-X lossless compress failed: Cannot\|coredump\|core dump" | tr '\n' ' ' | cut -c1-200; echo
done; done; done
