## Ratio and time at equal pointwise bound (tol = eb x RMS, REL_L2)

| field | eb | hpMGARD CR | MGARD-X s=inf CR | MGARD-X s=0 CR | hpMGARD / MGARD-X(s=inf) | max err / tol | hpMGARD s (16 thr, incl. tuning) | MGARD-X s (serial) | chosen base | codec |
|---|---|---|---|---|---|---|---|---|---|---|
| p4_Density_aver | 0.01 | 8118 | 250.6 | 350.0 | 32.4x | 1.000 | 0.62 | 3.99 | p1 cellavg q=0.5 | huffman|zstd:3 |
| p4_Pressure_aver | 0.01 | 7598 | 242.4 | 324.6 | 31.3x | 1.000 | 0.61 | 3.97 | p1 cellavg q=1 | huffman|zstd:3 |
| p4_Rho_ms | 0.01 | 6469 | 209.9 | 288.1 | 30.8x | 1.000 | 0.61 | 3.99 | p1 cellavg q=1 | huffman|zstd:3 |
| p4_P_ms | 0.01 | 5023 | 202.3 | 270.4 | 24.8x | 1.000 | 0.64 | 3.90 | p2 cellavg q=0.5 | huffman|zstd:3 |
| p4_uu (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.07 | p0 plain q=0.5 | frbp |
| p4_vv (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.07 | p0 plain q=0.5 | frbp |
| p4_ww (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 4.98 | p0 plain q=0.5 | frbp |
| p4_uv (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.16 | p0 plain q=0.5 | frbp |
| p4_vw (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.02 | p0 plain q=0.5 | frbp |
| p4_uw (constant field) | 0.01 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.23 | p0 plain q=0.5 | frbp |
| p4_Velocity_aver | 0.01 | 1302 | 149.9 | 192.6 | 8.7x | 1.000 | 1.36 | 9.12 | p0 plain q=1 | huffman|zstd:3 |
| p5_Pressure_aver_8blocks | 0.01 | 9879 | 506.7 | 673.3 | 19.5x | 1.000 | 0.95 | 7.36 | p1 cellavg q=0.5 | huffman|zstd:3 |
| **geomean (non-constant fields), eb=0.01** | | **5435** | **241.0** | **323.6** | **22.6x** | | 0.80 (mean) | 5.39 (mean) | | |
| p4_Density_aver | 0.001 | 2208 | 136.2 | 167.9 | 16.2x | 1.000 | 0.61 | 4.25 | p1 cellavg q=2 | huffman|zstd:3 |
| p4_Pressure_aver | 0.001 | 2115 | 130.6 | 159.4 | 16.2x | 1.000 | 0.63 | 4.10 | p1 cellavg q=1 | huffman|zstd:3 |
| p4_Rho_ms | 0.001 | 1742 | 115.4 | 141.3 | 15.1x | 1.000 | 0.67 | 4.19 | p1 cellavg q=1 | huffman|zstd:3 |
| p4_P_ms | 0.001 | 1613 | 111.2 | 136.2 | 14.5x | 1.000 | 0.64 | 4.26 | p1 cellavg q=2 | huffman|zstd:3 |
| p4_uu (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.30 | p0 plain q=0.5 | frbp |
| p4_vv (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.40 | p0 plain q=0.5 | frbp |
| p4_ww (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.26 | p0 plain q=0.5 | frbp |
| p4_uv (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.35 | p0 plain q=0.5 | frbp |
| p4_vw (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.37 | p0 plain q=0.5 | frbp |
| p4_uw (constant field) | 0.001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.09 | 5.28 | p0 plain q=0.5 | frbp |
| p4_Velocity_aver | 0.001 | 505 | 79.7 | 97.9 | 6.3x | 1.000 | 1.42 | 9.14 | p0 plain q=0.25 | huffman|zstd:3 |
| p5_Pressure_aver_8blocks | 0.001 | 2903 | 277.0 | 328.0 | 10.5x | 1.000 | 0.97 | 7.36 | p1 cellavg q=1 | huffman|zstd:3 |
| **geomean (non-constant fields), eb=0.001** | | **1637** | **130.9** | **159.6** | **12.5x** | | 0.82 (mean) | 5.55 (mean) | | |
| p4_Density_aver | 0.0001 | 734 | 35.9 | 39.2 | 20.4x | 1.000 | 0.64 | 4.25 | p0 plain q=0.25 | huffman|zstd:3 |
| p4_Pressure_aver | 0.0001 | 675 | 36.0 | 38.7 | 18.8x | 1.000 | 0.65 | 4.18 | p0 plain q=0.25 | huffman|zstd:3 |
| p4_Rho_ms | 0.0001 | 529 | 32.3 | 36.8 | 16.4x | 1.000 | 0.66 | 4.25 | p0 plain q=2 | huffman|zstd:3 |
| p4_P_ms | 0.0001 | 557 | 32.4 | 36.4 | 17.2x | 1.000 | 0.78 | 4.36 | p1 cellavg q=0.25 | huffman|zstd:3 |
| p4_uu (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.10 | 5.07 | p0 plain q=0.5 | frbp |
| p4_vv (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.11 | 5.33 | p0 plain q=0.5 | frbp |
| p4_ww (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.10 | 5.06 | p0 plain q=0.5 | frbp |
| p4_uv (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.10 | 5.03 | p0 plain q=0.5 | frbp |
| p4_vw (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.10 | 5.06 | p0 plain q=0.5 | frbp |
| p4_uw (constant field) | 0.0001 | 1400212 | 7.9 | 7.9 | 178280.1x | 0.000 | 0.10 | 5.08 | p0 plain q=0.5 | frbp |
| p4_Velocity_aver | 0.0001 | 217 | 44.1 | 51.2 | 4.9x | 1.000 | 2.78 | 9.41 | p4 plain q=2 | huffman|zstd:3 |
| p5_Pressure_aver_8blocks | 0.0001 | 867 | 44.1 | 48.5 | 19.7x | 1.000 | 0.97 | 7.35 | p1 cellavg q=1 | huffman|zstd:3 |
| **geomean (non-constant fields), eb=0.0001** | | **549** | **37.2** | **41.4** | **14.8x** | | 1.08 (mean) | 5.63 (mean) | | |

## Timing, fixed configuration (p1 cellavg, huffman|zstd:3), p4 pressure, eb 1e-3

| run | threads | hpMGARD compress s | hpMGARD decompress s | CR | MGARD-X s=inf compress s (serial) | MGARD-X s=0 s |
|---|---|---|---|---|---|---|
| p4_Pressure_aver_1thread_fixed | 1 | 0.61 | 0.35 | 2048 | 0.00 | 0.00 |
| p4_Pressure_aver_16threads_fixed | 16 | 0.31 | 0.30 | 2048 | 0.00 | 0.00 |

## Codec ablation, p4 pressure, eb 1e-3, fixed hierarchy (p1 cellavg base, same chain for base and residual)

| chain | bytes | CR | compress s (16 thr) | decompress s |
|---|---|---|---|---|
| `huffman|zstd:3` | 129,189 | 2048 | 0.32 | 0.32 |
| `huffman|zstd:19` | 115,942 | 2283 | 0.55 | 0.31 |
| `huffman` | 8,182,108 | 32 | 0.31 | 0.27 |
| `frbp|zstd:3` | 304,247 | 870 | 0.23 | 0.13 |
| `frbp|zstd:19` | 163,421 | 1619 | 0.50 | 0.12 |
| `frbp` | 6,572,219 | 40 | 0.24 | 0.10 |
| `raw|zstd:3` | 155,492 | 1702 | 0.28 | 0.13 |
| `raw|zstd:19` | 65,311 | 4052 | 0.48 | 0.12 |
