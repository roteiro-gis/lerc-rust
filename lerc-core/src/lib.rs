//! Shared types and errors for pure-Rust LERC codecs.

mod error;
mod types;

pub use error::{Error, Result};
pub use types::{
    BandLayout, BandSetInfo, BlobInfo, DataType, Decoded, DecodedBandSet, DecodedF64,
    NdArrayElement, PixelData, Version,
};
