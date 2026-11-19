#!/bin/bash
# Full header overlays (quote-includes resolve relative to the including file, so the whole tree must be copied).
# E0 unpatched | E1 GenerateCL HIP block width 64 | E2 width 32 | E3 WarpSize forced 32 | E4 width 32 + WarpSize 32
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load rocm/7.2.0 2>/dev/null
P=/lustre/orion/cfd164/proj-shared/gongq/HODG/results/gate/probe; MG=/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD/install-hip-frontier.mgard17; ROCM=/opt/rocm-7.2.0
GCL=mgard/mgard-x/Lossless/ParallelHuffman/GenerateCL.hpp; DAH=mgard/mgard-x/RuntimeX/DeviceAdapters/DeviceAdapterHip.h
for V in E0 E1 E2 E3 E4; do
  O=$P/overlay_$V; rm -rf $O; mkdir -p $O; cp -r $MG/include $O/include
  case $V in
    E1) sed -i 's/      return 512;/      return 64;/' $O/include/$GCL ;;
    E2) sed -i 's/      return 512;/      return 32;/' $O/include/$GCL ;;
    E3) sed -i 's/\(WarpSize\[d\] = \).*;/\132;/' $O/include/$DAH ;;
    E4) sed -i 's/      return 512;/      return 32;/' $O/include/$GCL; sed -i 's/\(WarpSize\[d\] = \).*;/\132;/' $O/include/$DAH ;;
  esac
  echo "$V: GenerateCL width line: $(grep -m1 -n 'return 512;\|return 64;\|return 32;' $O/include/$GCL | tr -d ' ') | WarpSize: $(grep -m1 'WarpSize\[d\] = ' $O/include/$DAH | tr -d ' ')"
  ( $ROCM/bin/hipcc -O2 --offload-arch=gfx90a -std=c++17 -DMGARDX_COMPILE_HIP -D__HIP_PLATFORM_AMD__ -I$O/include $P/xhuff_min.hip -L$MG/lib64 -lmgard -lprotobuf -L$MG/lib -lzstd -Wl,--disable-new-dtags -Wl,-rpath,$MG/lib64 -Wl,-rpath,$MG/lib -Wl,-rpath,$ROCM/lib -o $P/xhuff_min_$V 2>&1 | grep -E "error" | head -3; echo "built $V" ) &
done
wait
md5sum $P/xhuff_min_E? | awk '{print substr($1,1,8), $2}'
