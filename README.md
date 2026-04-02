# lerc-rust

`lerc-rust` is a pure-Rust implementation of the LERC raster codec.

This crate is intended to be the Rust counterpart to Esri's LERC repository,
with a Rust-native API and no C or C++ FFI layer.

Workspace layout:

- `lerc-core`: shared types and errors
- `lerc-reader`: pure-Rust LERC inspection and decode paths

Design goals:

- no FFI, no generated bindings, no C++ dependency
- stable shared metadata and pixel buffer types in `lerc-core`
- decoder-first architecture in `lerc-reader`
- first-class `ndarray::ArrayD` integration for downstream engines
- clean separation so container crates such as `geotiff-rust` depend on this
  workspace instead of embedding codec logic

Implemented in `lerc-reader` today:

- Lerc1 header parsing
- Lerc1 mask decoding
- Lerc1 tiled block decode
- Lerc1 shared-mask concatenated band counting
- Lerc2 header parsing
- concatenated Lerc2 band counting
- pure-Rust Fletcher32 checksum verification
- Lerc2 mask decoding
- Lerc2 constant-surface decode
- Lerc2 one-sweep raw decode
- Lerc2 tiled decode
- Lerc2 bit-stuffed block decode
- Lerc2 Huffman decode
- public inspection and decode entry points for native and `f64` output buffers
- direct decode helpers into `ndarray::ArrayD`
- shape helpers for raster and mask arrays

Verified coverage:

- synthetic unit fixtures for Lerc1 bit-stuffed blocks and shared-mask
  concatenated bands
- synthetic unit fixtures for constant, one-sweep, concatenated-band, and
  per-depth-range cases
- ndarray conversion tests for 2D rasters, 3D rasters, and masks
- an interoperability fixture from Esri's JavaScript sanity test exercising a
  real upstream Lerc2 blob

The crate is designed so those remaining decode paths can be added without
breaking the public metadata or pixel-buffer APIs.

Planned next:

- additional interoperability fixtures covering more tiled and masked blobs
- writer support in a future `lerc-writer` crate
