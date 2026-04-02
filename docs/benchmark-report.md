# Benchmark Report

Date: 2026-04-02

This report summarizes the current Criterion comparison suite for
`lerc-reader` against Esri's official native `libLerc` decoder.

## System Under Test

- Machine: Apple Silicon Mac
- CPU topology: 8 logical CPUs
- Memory: 16 GiB
- OS: macOS 13.0
- Architecture: `arm64`
- Rust toolchain: `rustc 1.92.0`
- Reference environment: native `libLerc` v4.1.0 helper

These numbers reflect this host and should be treated as a local baseline, not
as universal throughput claims.

## Scope

The current suite measures:

- masked single-band Lerc1 decode (`world.lerc1`)
- masked single-band Lerc2 decode (`california_400_400_1_float.lerc2`)
- concatenated multi-band Lerc2 decode (`bluemarble_256_256_3_byte.lerc2`)

Each benchmark validates decoded byte-hash parity against `libLerc` before
timing and then compares full decode throughput for the same fixture.

## Methodology

Commands used for this report:

```sh
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-reader --test reference_parity
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo bench -p lerc-reader --bench reference_compare_bench -- --noplot
```

Notes:

- The benchmark harness normalizes `libLerc`'s decoded layout to match
  `lerc-reader`'s public API before hashing and timing.
- The helper links directly against the native Esri library and uses
  `lerc_decode_4D` so parity and benchmark coverage exercise the same decoder
  path.
- Concatenated band sets are compared in bands-last layout because that is the
  public shape exposed by `lerc-reader`.
- Both implementations include the decode-to-checksum path during timing so the
  benchmark validates real decoded output rather than parser-only work.

## Current Results

| fixture | `lerc-rust` time | `libLerc` time | result |
| --- | ---: | ---: | --- |
| `world.lerc1` | 439-549 us | 1.06-1.20 ms | `lerc-rust` faster |
| `california_400_400_1_float.lerc2` | 705 us-1.07 ms | 1.36-1.49 ms | `lerc-rust` faster |
| `bluemarble_256_256_3_byte.lerc2` | 1.83-2.01 ms | 1.88-2.20 ms | near parity, slight `lerc-rust` edge |

Representative throughput ranges from the same run:

| fixture | `lerc-rust` throughput | `libLerc` throughput |
| --- | ---: | ---: |
| `world.lerc1` | 459-575 MiB/s | 209-239 MiB/s |
| `california_400_400_1_float.lerc2` | 570-866 MiB/s | 410-449 MiB/s |
| `bluemarble_256_256_3_byte.lerc2` | 93.1-102.6 MiB/s | 85.1-99.8 MiB/s |

## Interpretation

- `lerc-reader` stays clearly ahead on the two masked single-band fixtures.
- The concatenated 3-band byte case is much tighter and effectively near parity
  on this host, with a small median edge for `lerc-reader`.
- The benchmark is now a codec-to-codec comparison with no Python or NumPy
  boundary overhead in the reference timing.

## Commands

Run the full reference benchmark suite in Docker:

```sh
./scripts/run-reference-benchmarks.sh
```

Run locally when `libLerc` is installed or discoverable via
`LERC_REFERENCE_LIB_DIR`:

```sh
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo bench -p lerc-reader --bench reference_compare_bench -- --noplot
```
