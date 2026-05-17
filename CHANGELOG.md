# Changelog

## 0.4.1 - 2026-05-17

### Fixed

- Rejected malformed Lerc2 bit-stuffed payloads without panicking.
- Rejected malformed Huffman table metadata before oversized allocation or invalid bit reads.
- Fixed direct band materialization when tiled decode changes write-order hints after sparse/default-filled paths.

### Added

- Added malformed codec regression tests and a structured Lerc2 bitstuff fuzz target.

## 0.4.0 - 2026-05-16

### Added

- Added Lerc2 v6/no-data writer emission for depth rasters.
- Added `EncodeOptions::no_data_value` so callers can request v6/no-data output.
- Added generated writer/libLerc parity coverage for v6/no-data blobs.
- Strengthened generated band-set roundtrip coverage for direct decode APIs across layouts, masks, zero blocks, Huffman tiles, and v5 diff tiles.

### Fixed

- Fixed direct tiled decode materialization for all-valid zero/empty tile blocks.
- Fixed `encoded_len_upper_bound*` helpers so v6/no-data output is sized with the v6 header and validated consistently with `encode*`.
- Fixed Lerc1 compact unsigned offset decoding to match libLerc.

### Changed

- Reduced duplicate tiled decode logic between allocating and caller-provided decode paths.
- Cached tiled writer block planning to avoid duplicate analysis during encoding.

## 0.3.0 - 2026-04-17

### Added

- Added masked single-blob Lerc2 decode APIs.
- Added public external-mask metadata and reader support for Lerc2 v6/no-data blobs.
- Added band-set Lerc2 writer APIs with shared-mask emission.
- Added writer planning for one-sweep, Huffman, and v5 diff tile bodies.
- Added parity coverage for v6/no-data, external masks, signed Huffman, and f64 fixtures.
- Added band-set external-mask APIs.

### Fixed

- Rejected zero-depth Lerc2 headers.
- Fixed Lerc1 with-mask parsing behavior.

### Documentation

- Documented release publish order.
- Documented that writer output was limited to v5 at the time of release.

## 0.2.0 - 2026-04-04

### Added

- Added the `lerc-writer` crate for Lerc2 encoding.
- Added direct mask and band decode paths.
- Added writer buffer APIs.

### Changed

- Refactored and hardened Lerc decoding.
- Consolidated band materialization and shared raster core behavior.
- Optimized band-set, f64 decode, and writer tile paths.

### Fixed

- Fixed metadata scanning and validation regressions.
- Hardened materializer finalization and cleanup behavior.

## 0.1.0 - 2026-04-02

### Added

- Added the initial pure-Rust LERC workspace.
- Added Lerc1 decoder support.
- Added first-class ndarray support.
- Added multi-band decode support and official interoperability fixtures.
- Added documentation, parity checks, benchmarks, and CI.
