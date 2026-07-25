# lerc-rust

[![lerc-core crates.io](https://img.shields.io/crates/v/lerc-core.svg)](https://crates.io/crates/lerc-core)
[![lerc-core docs.rs](https://docs.rs/lerc-core/badge.svg)](https://docs.rs/lerc-core)
[![lerc-reader crates.io](https://img.shields.io/crates/v/lerc-reader.svg)](https://crates.io/crates/lerc-reader)
[![lerc-reader docs.rs](https://docs.rs/lerc-reader/badge.svg)](https://docs.rs/lerc-reader)
[![lerc-writer crates.io](https://img.shields.io/crates/v/lerc-writer.svg)](https://crates.io/crates/lerc-writer)
[![lerc-writer docs.rs](https://docs.rs/lerc-writer/badge.svg)](https://docs.rs/lerc-writer)

Pure-Rust LERC encoding and decoding for raster and elevation data. No C/C++
FFI, no build scripts, and shared codec primitives designed to be reused by
higher-level raster/container crates such as `geotiff-rust`.

## Crates

| Crate | Description |
|---|---|
| `lerc-core` | Codec-neutral shared types, errors, typed raster views, and sample helpers |
| `lerc-reader` | Pure-Rust LERC inspection and decode paths for Lerc1 and Lerc2 blobs |
| `lerc-writer` | Pure-Rust Lerc2 writer for single blobs and concatenated band sets |

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
    EncodeOptions::new().with_max_z_error(0.5),
)?;
```

### Features

`lerc-reader` enables `ndarray` by default. Disable default features for a
minimal slice/vector decoder, or opt into parallel multi-band decoding:

```toml
lerc-reader = { version = "0.5", default-features = false }
# or
lerc-reader = { version = "0.5", features = ["ndarray", "rayon"] }
```

The `rayon` feature resolves concatenated blob boundaries and inherited masks
sequentially, then decodes independent bands in parallel. BSQ output is written
directly to disjoint slices; interleaved output is scattered deterministically.

### Version Support

`lerc-reader` reads Lerc1 and Lerc2 blobs, including Lerc2 revisions through
v6. An existing blob's version can be inspected with `get_blob_info(&blob)?.version`.

`lerc-writer` writes Lerc2 only and does not expose a manual revision selector, instead it
depends on the encoded byte stream:

- v4 for ordinary output
- v5 for depth rasters when slices compress better as per-tile differences
  from the previous depth slice
- v6 when `EncodeOptions::no_data_value` is set for a depth raster

### Blob Entry Points

Single-blob entry points are strict. If you intentionally want first-blob
inspection or decode from a concatenated payload, use `inspect_first()` or
`decode_first()`.
Use `InspectOptions::new().with_compute_value_range(false)` when Lerc1 header
metadata is sufficient and exact range scanning would be unnecessary work.
For Lerc1 shared-mask or Lerc2 external-mask blobs, use the corresponding
single-blob `*_with_mask()` entry points.
For concatenated Lerc2 band sets whose first blob uses an external mask, use
the matching band-set `*_with_mask()` entry points.

Concatenated band sets decode to bands-last arrays by default, and can also be
requested in BSQ order:

```rust
let rgb: ndarray::ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob)?;
assert_eq!(rgb.shape(), &[height, width, bands]);

let (info, bsq): (_, Vec<u8>) =
    lerc_reader::decode_band_set_vec(&blob, lerc_core::BandLayout::Bsq)?;
assert_eq!(bsq.len(), (info.width() * info.height() * info.band_count() as u32) as usize);
```

`decode_from_reader` reads exactly one blob and leaves subsequent bytes unread.
`decode_band_set_from_reader` continues through clean EOF, enabling bounded
blob-by-blob decoding from files, sockets, and other `std::io::Read` sources.

## Supported Now

- Lerc1 header parsing, mask decoding, tiled block decode, and concatenated
  shared-mask band sets
- Lerc2 header parsing, Fletcher32 verification, mask decoding, constant/raw,
  tiled, bit-stuffed, and Huffman decode paths
- Lerc2 writes for single blobs and concatenated band sets with optional masks,
  depth metadata, checksum, constant/per-depth constant, one-sweep, tiled,
  Huffman, v5 diff-tile encode paths, and v6/no-data emission
- Lerc2 v6/no-data decode support
- Native typed decode and type-promoting `f64` decode
- Exact one-blob and EOF-terminated band-set streaming from `std::io::Read`
- Optional deterministic Rayon parallelism across concatenated bands
- Strict single-blob APIs plus permissive first-blob adapters for concatenated payloads
- Feature-gated `ndarray::ArrayD` conversion for rasters, band sets, and masks, with selectable band layout
- Codec-neutral typed raster views and shared metadata types in `lerc-core`

## Testing

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test -p lerc-reader --no-default-features
cargo test -p lerc-reader --no-default-features --features rayon
```

For coordinated workspace publishes, see
[docs/release-checklist.md](./docs/release-checklist.md).

Interop fixtures are vendored in the repository under
[`testdata/interoperability`](https://github.com/i-norden/lerc-rust/tree/main/testdata/interoperability).
The default test suite covers:

- synthetic decoder-path unit tests embedded in `lerc-reader`
- official Esri fixtures for Lerc1, masked Lerc2, and concatenated multi-band
  Lerc2
- an Esri JavaScript sanity fixture for `depth > 1`
- malformed-input regression coverage for strict parsing, mask RLE, Huffman tables,
  bit-stuffed payloads, concatenated parsing, and writer roundtrips/output sizing

Reference-library parity tests compare both decode and writer output against
Esri's official `libLerc` decoder when a compiled helper path is configured;
otherwise they self-skip:

```sh
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-reader --test reference_parity
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-writer --test reference_parity
```

For a reproducible reference environment, run the Docker harness:

```sh
./scripts/run-reference-parity.sh
```

Criterion benchmark entry points live in `lerc-reader` for decode-vs-`libLerc`
comparison and in `lerc-writer` for encode / encode+decode throughput:

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
cargo fuzz run lerc2_bitstuff_blocks
cargo fuzz run concatenated
cargo fuzz run encode_roundtrip
```

## License

MIT OR Apache-2.0
