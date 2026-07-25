# Benchmark Report

Last methodology update: 2026-07-10

This report summarizes the current Criterion comparison suite for
`lerc-reader` against Esri's official native `libLerc` decoder and records the
repo-level benchmark entry points that now include `lerc-writer`.

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
- writer analysis/planning/emission after cached mask, diff-plan, and bit-packer optimizations

Each benchmark validates decoded byte-hash parity against `libLerc` before
timing and then reports two Rust measurements for the same fixture:

- `decode-only`: fixture bytes are preloaded in memory before timing
- `load-plus-decode`: fixture loading remains inside the timed loop

The current native `libLerc` comparison remains on the end-to-end
`load-plus-decode` path via the helper.

## Methodology

Commands used for this report:

```sh
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-reader --test reference_parity
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo test -p lerc-writer --test reference_parity
LERC_READER_REFERENCE_HELPER="$(./scripts/build-reference-helper.sh)" \
  cargo bench -p lerc-reader --bench reference_compare_bench -- --noplot
cargo bench -p lerc-writer --bench encode_bench -- --noplot
```

Notes:

- The benchmark harness normalizes `libLerc`'s decoded layout to match
  `lerc-reader`'s public API before hashing and timing.
- The helper links directly against the native Esri library and uses
  `lerc_decode_4D` so parity and benchmark coverage exercise the same decoder
  path.
- Concatenated band sets are compared in bands-last layout because that is the
  public shape exposed by `lerc-reader`.
- The Rust harness now publishes a pure in-memory `decode-only` group and a
  separate `load-plus-decode` group so codec speed is not conflated with file I/O.
- Both implementations include the decode-to-checksum path during timing so the
  benchmark validates real decoded output rather than parser-only work.
- The writer bench is operationalized through the same repo harness, but this
  report does not yet publish encode baseline numbers from a fixed host run.

## Current Results

The figures below are the last recorded fixed-host baseline (2026-04-02).
They remain historical comparison points; run the commands above to measure
the 0.5.0 implementation on the current host.

A 2026-07-10 local 0.5.0 writer smoke run on Apple Silicon recorded:

| benchmark | time |
| --- | ---: |
| `encode/u8-bitstuff` | 1.460-1.481 ms |
| `encode-plus-decode/f32` | 1.587-1.601 ms |

The figures below summarize the `load-plus-decode` comparison, because that is
the only group with a directly comparable `libLerc` number today.

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

The Criterion report also includes separate `decode-only` groups for the Rust
implementation. Those are the numbers to use when making codec-speed claims.

## Interpretation

- `lerc-reader` stays clearly ahead on the two masked single-band fixtures.
- The concatenated 3-band byte case is much tighter and effectively near parity
  on this host, with a small median edge for `lerc-reader`.
- The benchmark is now a codec-to-codec comparison with no Python or NumPy
  boundary overhead in the reference timing.
- Decode-only and end-to-end timings are now intentionally separated, so read
  the correct group for the claim you want to make.

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
