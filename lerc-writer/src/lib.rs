//! Pure-Rust Lerc2 writer for single-blob rasters.

mod lerc2;

pub use lerc2::{encode, encode_into, encoded_len_upper_bound, EncodeOptions};
pub use lerc_core::{MaskView, RasterView};
