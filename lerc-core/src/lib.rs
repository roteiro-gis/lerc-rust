//! Shared types and errors for pure-Rust LERC codecs.

mod error;
mod types;

pub use error::{Error, Result};
pub use types::{BlobInfo, DataType, Decoded, DecodedF64, NdArrayElement, PixelData, Version};
