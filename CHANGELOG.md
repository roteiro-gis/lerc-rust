# Changelog

## Unreleased

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
