# HpMGARD-Tools

Drivers, tests, benchmarks, scripts and documents for `CompressHpMGARDOperator` (the hpMGARD
compression library and its ADIOS2 plugin). Same role as `Unstructured-ReMesh` plays for the centroid
operator: nothing here is needed to use the operator, everything here is needed to develop and validate it.

```
hpmgard_bench.cpp        raw-dump benchmark: hpMGARD (fixed or tuned) vs MGARD-X at the same absolute bound, error check, CSV
tests/                   unit tests (ctest): quadrature, operators, codecs, compressor round trips and bound checks
analysis/                dump/analysis scripts used for the HODG format review and the emulation studies
documents/               design (code-design.md), format review, benchmark memos
scripts/                 Frontier job scripts
build_scripts_cpu.sh     configure + build against an installed CompressHpMGARDOperator
```

Build: `HPMGARD_PREFIX=<CompressHpMGARDOperator install prefix> bash build_scripts_cpu.sh`, then
`ctest --test-dir build_cpu --output-on-failure`.

Benchmark example (p4 VKI pressure dumped with `analysis/dumpbp`):

```bash
build_cpu/hpmgard_bench --raw p4_Pressure_aver_s0.bin --ncells 264640 --p 4 --tol 1e-3 --tune --mgardx-direct \
  --conn p4_conn_s0.bin --points p4_points_s0.bin --threads 16 --csv results.csv
```
