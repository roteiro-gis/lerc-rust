# lerc-rust

Pure-Rust LERC decoding for raster and elevation data. No C/C++ FFI, no build
scripts, and a decoder-first API designed to be reused by container crates such
as `geotiff-rust`.

## Crates

| Crate | Description |
|---|---|
| `lerc-core` | Codec-neutral shared types, errors, typed raster views, and sample helpers |
| `lerc-reader` | Pure-Rust LERC inspection and decode paths for Lerc1 and Lerc2 blobs |
| `lerc-writer` | Pure-Rust Lerc2 single-blob writer with mask, depth, checksum, and tiled emit paths |

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

```rust
use lerc_core::RasterView;
use lerc_writer::{encode, EncodeOptions};

let pixels = vec![1u8, 2, 3, 4];
let blob = encode(
    RasterView::new(2, 2, 1, &pixels)?,
    None,
    EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 8,
    },
)?;
```

Single-blob entry points are strict. If you intentionally want first-blob
inspection or decode from a concatenated payload, use `inspect_first()` or
`decode_first()`.

Concatenated band sets decode to bands-last arrays by default, and can also be
requested in BSQ order:

```rust
let rgb: ndarray::ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob)?;
assert_eq!(rgb.shape(), &[height, width, bands]);

let (info, bsq): (_, Vec<u8>) =
    lerc_reader::decode_band_set_vec(&blob, lerc_core::BandLayout::Bsq)?;
assert_eq!(bsq.len(), (info.width() * info.height() * info.band_count() as u32) as usize);
```

## Supported Now

- Lerc1 header parsing, mask decoding, tiled block decode, and concatenated
  shared-mask band sets
- Lerc2 header parsing, Fletcher32 verification, mask decoding, constant/raw,
  tiled, bit-stuffed, and Huffman decode paths
- Lerc2 single-blob writes with optional masks, depth metadata, checksum, and
  constant/raw/bit-stuffed tiles
- Native typed decode and type-promoting `f64` decode
- Strict single-blob APIs plus permissive first-blob adapters for concatenated payloads
- Direct `ndarray::ArrayD` conversion for rasters, band sets, and masks, with selectable band layout
- Codec-neutral typed raster views and shared metadata types in `lerc-core`

## Testing

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Interop fixtures are vendored in the repository under
[`testdata/interoperability`](https://github.com/i-norden/lerc-rust/tree/main/testdata/interoperability).
The default test suite covers:

- synthetic decoder-path unit tests embedded in `lerc-reader`
- official Esri fixtures for Lerc1, masked Lerc2, and concatenated multi-band
  Lerc2
- an Esri JavaScript sanity fixture for `depth > 1`
- malformed-input regression coverage for strict parsing, mask RLE, Huffman tables,
  bit-stuffed payloads, concatenated parsing, and writer roundtrips/output sizing

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

For methodology and current benchmark notes, see the repository copy of
[docs/benchmark-report.md](https://github.com/i-norden/lerc-rust/blob/main/docs/benchmark-report.md).

`cargo-fuzz` targets for decoder hardening live under [`fuzz/`](./fuzz):

```sh
cargo fuzz run headers
cargo fuzz run mask_rle
cargo fuzz run huffman_tables
cargo fuzz run bitstuff_blocks
cargo fuzz run concatenated
cargo fuzz run encode_roundtrip
```

## License

MIT OR Apache-2.0
