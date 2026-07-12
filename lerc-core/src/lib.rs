#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Shared types and errors for pure-Rust LERC codecs.

mod error;
mod raster;
mod types;

pub use error::{Error, Result};
pub use raster::{
    append_value_as, bits_required, coerce_f64_to_data_type, count_valid_in_block, fletcher32,
    output_value, pixel_count_from_dims, read_scalar, read_typed_values, read_values_as,
    sample_count_from_dims, sample_index, words_from_padded, BandSetView, MaskView, RasterView,
    Sample,
};
pub use types::{BandLayout, BandSetInfo, BlobInfo, DataType, MaskEncoding, PixelData, Version};
