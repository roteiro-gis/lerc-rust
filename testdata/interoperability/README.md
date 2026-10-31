# Interoperability Fixtures

This directory holds the small vendored fixtures used for LERC decode
interoperability, parity, and benchmark coverage.

Included fixtures:

- `world.lerc1`: official Esri Lerc1 sample with an external validity mask
- `california_400_400_1_float.lerc2`: official masked float Lerc2 sample
- `bluemarble_256_256_3_byte.lerc2`: official concatenated 3-band byte Lerc2
  sample whose first blob exercises libLerc's integer Huffman encoding
- `esri_js_sanity_u8_3d.csv`: byte-for-byte fixture from Esri's JavaScript
  sanity test covering `depth > 1`
- `gdal-3.12-f32-masked-depth.lerc2`, `gdal-3.12-f32-centimetre-grid.lerc2`:
  single-tile LERC payloads written by GDAL 3.12.2 for the polynomial surfaces
  in `test-support/surfaces.rs`; the writer's GDAL profile must reproduce them
  byte for byte

These files are exercised by:

- `lerc-reader/tests/interop.rs`
- `lerc-reader/tests/reference_parity.rs`
- `lerc-reader/benches/reference_compare_bench.rs`
- `lerc-writer/tests/gdal_profile.rs`
