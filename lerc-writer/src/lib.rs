#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Pure-Rust Lerc2 writer for single-blob rasters and concatenated band sets.

mod lerc2;

pub use lerc2::{
    encode, encode_band_set, encode_band_set_into, encode_into, encoded_band_set_len_upper_bound,
    encoded_len_upper_bound, EncodeOptions,
};
pub use lerc_core::{BandSetView, MaskView, RasterView};
