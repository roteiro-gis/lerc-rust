# Changelog

## Unreleased

- fix Lerc1 remainder-tile decoding and inspection for legal grids whose final tile is larger than the base tile, and replace unchecked reader size arithmetic
- align integer zero-error encoding and multidimensional no-data filtering/remapping with libLerc semantics; constrain micro-block sizes to 2 through 64
- fix signed reduced offsets and absolute-range clamping for bit-stuffed v5 difference tiles
- redesign the public error taxonomy to distinguish caller arguments, corrupt blobs, size overflow, checksum failure, internal invariants, and stream I/O
- split the writer into analysis, headers, masks, tiles, bit-stuffing, Huffman, and options modules; remove the duplicate band-materialization crate and duplicate owned decode walks
- cache mask payloads and tile decisions, eliminate cloned plans and rebuilt diff vectors, fuse Huffman histograms into analysis, and replace scalar bit packing with an accumulator
- make `ndarray` optional, add complete layout-aware `f64` ndarray APIs and borrowing conversions, and add the optional deterministic `rayon` band decoder
- add exact one-blob and EOF-terminated band-set decoding from `std::io::Read`, plus configurable Lerc1 range inspection
- add encoded-byte snapshots, all-datatype property tests, expanded no-data/type fuzzing, workspace lints, public API documentation, and broader CI coverage
- bump the coordinated workspace crates to 0.5.0; `EncodeOptions` and `Error` are now non-exhaustive and use forward-compatible builders/variants

## 0.4.3 - 2026-06-25

- reject malformed Lerc1 and Lerc2 headers with zero dimensions, invalid block geometry, zero Lerc2 micro-block sizes, negative or non-finite error tolerances, and non-finite range/no-data values
- fix direct band-set decode APIs so returned metadata includes decoded Lerc1 value ranges and Lerc2 per-depth ranges
- validate zero-sized band-set decode payloads instead of returning before malformed payload checks
- keep the locked dev-dependency graph compatible with Rust 1.77 and add CI coverage for MSRV, Miri, CodeQL, and libLerc parity on pull requests
- document unsafe-code invariants for direct band materialization and typed band-set conversions

## 0.4.2 - 2026-05-17

- reject oversized Lerc2 mask bitsets, decoded masks, constant pixel buffers, one-sweep outputs, tile outputs, and Huffman outputs before allocating
- reject oversized Lerc1 block tables, decoded masks, pixel buffers, and block buffers before allocating
- reject oversized decoded band masks and band materializer buffers before allocating

## 0.4.1 - 2026-05-17

- reject malformed Lerc2 bit-stuffed payloads without panicking
- reject malformed Huffman table metadata before oversized allocation or invalid bit reads
- fix direct band materialization when tiled decode changes write-order hints after sparse/default-filled paths
- add malformed codec regression tests and a structured Lerc2 bitstuff fuzz target

## 0.4.0 - 2026-05-16

- add Lerc2 v6/no-data writer emission for depth rasters
- add `EncodeOptions::no_data_value` so callers can request v6/no-data output
- add generated writer/libLerc parity coverage for v6/no-data blobs
- strengthen generated band-set roundtrip coverage for direct decode APIs across layouts, masks, zero blocks, Huffman tiles, and v5 diff tiles
- fix direct tiled decode materialization for all-valid zero/empty tile blocks
- fix `encoded_len_upper_bound*` helpers so v6/no-data output is sized with the v6 header and validated consistently with `encode*`
- fix Lerc1 compact unsigned offset decoding to match libLerc
- reduce duplicate tiled decode logic between allocating and caller-provided decode paths
- cache tiled writer block planning to avoid duplicate analysis during encoding

## 0.3.0 - 2026-04-17

- add masked single-blob Lerc2 decode APIs
- add public external-mask metadata and reader support for Lerc2 v6/no-data blobs
- add band-set Lerc2 writer APIs with shared-mask emission
- add writer planning for one-sweep, Huffman, and v5 diff tile bodies
- add parity coverage for v6/no-data, external masks, signed Huffman, and f64 fixtures
- add band-set external-mask APIs
- reject zero-depth Lerc2 headers
- fix Lerc1 with-mask parsing behavior
- document release publish order
- document that writer output was limited to v5 at the time of release

## 0.2.0 - 2026-04-04

- add the `lerc-writer` crate for Lerc2 encoding
- add direct mask and band decode paths
- add writer buffer APIs
- refactor Lerc decoding validation
- consolidate band materialization and shared raster core behavior
- optimize band-set, f64 decode, and writer tile paths
- fix metadata scanning and validation regressions
- validate materializer finalization and cleanup behavior

## 0.1.0 - 2026-04-02

- initial public release
- add the initial pure-Rust LERC workspace
- add Lerc1 decoder support
- add first-class ndarray support
- add multi-band decode support and official interoperability fixtures
- add documentation, parity checks, benchmarks, and CI
