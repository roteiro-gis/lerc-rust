# lerc-rust

Pure-Rust LERC decoding for raster and elevation data. No C/C++ FFI, no build
scripts, and a decoder-first API designed to be reused by container crates such
as `geotiff-rust`.

## Crates

| Crate | Description |
|---|---|
| `lerc-core` | Shared types and errors for LERC blobs, decoded pixels, masks, and ndarray conversion |
| `lerc-reader` | Pure-Rust LERC inspection and decode paths for Lerc1 and Lerc2 blobs |

## Usage

```rust
use lerc_reader::{decode_ndarray, decode_mask_ndarray, get_blob_info};

let blob = std::fs::read("elevation.lerc2")?;
let info = get_blob_info(&blob)?;
println!(
    "version={:?} size={}x{} depth={} dtype={:?}",
    info.version, info.width, info.height, info.depth, info.data_type
);

let raster: ndarray::ArrayD<f32> = decode_ndarray(&blob)?;
let mask = decode_mask_ndarray(&blob)?;
println!("shape={:?} has_mask={}", raster.shape(), mask.is_some());
```

Concatenated band sets decode to bands-last arrays:

```rust
let rgb: ndarray::ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob)?;
assert_eq!(rgb.shape(), &[height, width, bands]);
```

## Supported Now

- Lerc1 header parsing, mask decoding, tiled block decode, and concatenated
  shared-mask band sets
- Lerc2 header parsing, Fletcher32 verification, mask decoding, constant/raw,
  tiled, bit-stuffed, and Huffman decode paths
- Native typed decode and type-promoting `f64` decode
- Direct `ndarray::ArrayD` conversion for rasters, band sets, and masks
- Shape helpers and shared metadata types in `lerc-core`

## Testing

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Interop fixtures are vendored under
[`testdata/interoperability`](testdata/interoperability/README.md). The default
test suite covers:

- synthetic decoder-path unit tests embedded in `lerc-reader`
- official Esri fixtures for Lerc1, masked Lerc2, and concatenated multi-band
  Lerc2
- an Esri JavaScript sanity fixture for `depth > 1`

Reference-library parity tests compare `lerc-reader` against Esri's official
`libLerc` decoder when a compiled helper path is configured; otherwise they
self-skip:

```sh
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-reader --test reference_parity
```

For a reproducible reference environment, run the Docker harness:

```sh
./scripts/run-reference-parity.sh
```

Criterion comparison benches against `libLerc` live in
`lerc-reader/benches/reference_compare_bench.rs`:

```sh
./scripts/run-reference-benchmarks.sh
```

For methodology and current benchmark notes, see
[docs/benchmark-report.md](docs/benchmark-report.md).

## License

MIT OR Apache-2.0
