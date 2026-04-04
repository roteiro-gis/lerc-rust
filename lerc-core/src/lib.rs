//! Shared types and errors for pure-Rust LERC codecs.

mod error;
mod types;

pub use error::{Error, Result};
pub use types::{
    copy_band_values_into_slice, BandLayout, BandMaterializer, BandSetInfo, BlobInfo, DataType,
    Decoded, DecodedBandSet, DecodedF64, NdArrayElement, PixelData, Version,
};
