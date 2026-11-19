# hpMGARD bitstream, file version 1.0.0

All integers are little-endian fixed width; doubles are IEEE-754 binary64; strings are `uint16 length`
followed by the bytes. Implemented in `src/hpmgard/Header.cpp` and `src/hpmgard/codec/`.

## 1. Block = header + payload

```
preamble (22 bytes)
  char[4]  "HPMG"                 uint8  operator_type = 54
  uint8[3] software_version        uint8[3] file_version       uint8 endianness (1 = little)
  uint32   metadata_size           uint32 metadata_crc32 (CRC-32/IEEE over the metadata bytes)
  uint16   flags   bit0 raw block (raw doubles/floats follow the header)
                   bit1 mesh identity section present   bit2 geometry CRC section present   bit3 stream directory present
metadata (metadata_size bytes)
  uint64 blockid   uint64 ndims   uint64 dims[ndims]   uint8 data_type (0 double, 1 float)
  uint8 cell_type (3 quad, 4 tet, 5 hex, 6 prism, 7 pyramid)   uint8 p   uint16 nsp   uint8 ncomp
  uint8 dof_layout (0 = lexicographic, first axis fastest, components interleaved per point)
  uint8 node_set (0 Gauss-Legendre, 1 Gauss-Lobatto)
  uint8 base_order   uint8 base_representation (0 data nodes, 1 closed)   uint8 base_method (0 plain, 1 cellavg, 2 dedup_sfc)
  uint8 norm (0 abs, 1 rel_l2, 2 rel_range)   f64 tolerance (as requested)
  per component: f64 var_scale, f64 tol_abs (pointwise bound of the output), f64 base_quantum, f64 residual_quantum
  f64 amp_inf (infinity norm of the base->data prolongation)
  str base_codec   str residual_codec   uint8 direct_mgardx
  [bit1] str connectivity_variable   str coordinates_variable   uint8 corner_perm_code   uint64 num_vertex_groups
  [bit2] uint32 crc32(vertex group ids)   uint32 crc32(Hilbert permutation)
  str writer   uint8[3] mgardx_version   uint8 writer_backend (0 CPU, 1 HIP, 2 CUDA)
  [bit3] uint32 nstreams; per stream: uint8 component, uint8 kind, uint8 codec, uint64 offset, uint64 bytes, uint64 count
payload
  the streams, at their offsets from the start of the payload
```

Stream kinds: 0 avg (cell averages, vertex means, or the plain base), 1 remainder (base minus its
average, Hilbert-ordered for dedup), 2 residual (data minus the prolonged decoded base), 3 direct
(MGARD-X on the whole component), 4 const (one double, constant component), 5 raw doubles.
Stream codecs: 0 symbol chain container (section 2), 1 MGARD-X stream, 2 raw doubles, 3 const.

Decoding rule: `component = P_c(avg_dec + remainder_dec) + residual_dec`, with `P_c` the interpolation
from the stored base nodes (closed nodes of order base_order, or data nodes) to the data nodes, and
`x_dec = symbol * quantum`. The pointwise error is bounded by `residual_quantum / 2 = tol_abs`.

Readers accept `file_version` up to 1.x.y; a newer major version is rejected with both version numbers
in the message. `metadata_size` lets a reader skip to the payload.

## 2. Symbol chain container (StreamCodec 0)

```
uint8 'P'   uint8 nstages   per stage: uint8 id (0 raw, 1 frbp, 2 huffman, 3 zstd), uint8 param, uint64 out_bytes
uint64 nsymbols   bytes (output of the last stage)
```
Stage 0 turns int32 symbols into bytes; every later stage is bytes -> bytes (zstd). Chains are written
as spec strings: `raw`, `frbp`, `huffman[:dict_bits]`, `...|zstd[:level]` (e.g. `huffman|zstd:3`).

### 2.1 raw
int32 symbols, little-endian, 4 bytes each.

### 2.2 FRBP (mp-MGARD fixed-rate bit packing, FIXED mode)
```
uint32 magic 0x46524250 ("FRBP")   uint32 version 1   uint64 n   f64 eb (informational)   uint64 num_blocks = ceil(n/32)   uint64 body_bytes
uint8 rate[num_blocks]  (padded to a multiple of 4)
per block with rate > 0: uint32 sign_word (bit i set = element i negative), rate x uint32 plane words
  (plane word b, bit i = bit b of |symbol_i|)
```
rate = number of bits of the largest |symbol| in the block (0 for an all-zero block).

### 2.3 Huffman (canonical, zigzag symbols, escaped outliers)
```
uint32 magic "HUF1"   uint64 n   uint8 dict_bits   uint32 table_count   table_count x (uint32 symbol, uint8 length)
uint64 outliers   uint64 bitstream_bytes   bitstream (MSB first)
```
Symbols are zigzag-mapped (`(v << 1) ^ (v >> 31)`). Values below `2^dict_bits` are coded by the table,
larger values by the escape symbol `2^dict_bits` followed by the 32-bit value. Codes are canonical:
assigned in (length, symbol) order, limited to 32 bits (frequencies are halved and the tree rebuilt until
the limit holds). The table lists only symbols with non-zero length, in canonical order.

### 2.4 zstd
A standard zstd frame of the previous stage's bytes; `param` is the compression level; the decoded size
is the previous stage's `out_bytes`.

## 3. MGARD-X streams (StreamCodec 1)
Unmodified `mgard_x::compress` output of a 1-D double array (self-describing MGARD-X header), produced
with an absolute pointwise bound and `s = inf`.
