#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! Pure-Rust LERC decoder.
//!
//! The public API distinguishes strict single-blob entry points from
//! concatenated-band helpers:
//!
//! - inspect a single blob with [`get_blob_info`]
//! - inspect only the first blob with [`inspect_first`]
//! - pass an explicit shared or external mask with the `*_with_mask` single-blob variants
//! - count concatenated blobs with [`get_band_count`]
//! - decode a single blob with [`decode`]
//! - decode only the first blob with [`decode_first`]
//! - decode concatenated band sets with [`decode_band_set`]
//! - seed a first-band external mask with the band-set `*_with_mask` variants
//! - decode promoted `f64` buffers with [`decode_to_f64`]
//! - decode only the first promoted `f64` blob with [`decode_first_to_f64`]
//! - decode directly into `ndarray::ArrayD` with the default `ndarray` feature

mod allocation;
mod bitstuff;
mod huffman;
mod io;
mod lerc1;
mod lerc2;
mod materialize;
mod pixel;
mod stream;
mod types;

#[cfg(test)]
mod tests;

use lerc_core::{BandLayout, BandSetInfo, BlobInfo, Error, Result};
use materialize::BandSink;
#[cfg(feature = "rayon")]
use materialize::{copy_band_values_into_slice, PixelDataWriter};
#[cfg(feature = "ndarray")]
use ndarray::ArrayD;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::allocation::default_vec;
use crate::pixel::Sample;
#[cfg(feature = "ndarray")]
pub use crate::types::NdArrayElement;
pub use crate::types::{BandElement, BandElementKind, Decoded, DecodedBandSet, DecodedF64};

/// Controls optional work performed while inspecting a blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct InspectOptions {
    /// Whether Lerc1 blocks are decoded far enough to compute their exact value range.
    pub compute_value_range: bool,
}

impl InspectOptions {
    /// Creates options that preserve the exact-range behavior of [`inspect_first`].
    pub const fn new() -> Self {
        Self {
            compute_value_range: true,
        }
    }

    /// Enables or disables exact Lerc1 value-range computation.
    pub const fn with_compute_value_range(mut self, compute_value_range: bool) -> Self {
        self.compute_value_range = compute_value_range;
        self
    }
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! dispatch_band_element {
    ($target:ty, |$concrete:ident| $body:block) => {
        match <$target as BandElement>::KIND {
            BandElementKind::I8 => {
                type $concrete = i8;
                $body
            }
            BandElementKind::U8 => {
                type $concrete = u8;
                $body
            }
            BandElementKind::I16 => {
                type $concrete = i16;
                $body
            }
            BandElementKind::U16 => {
                type $concrete = u16;
                $body
            }
            BandElementKind::I32 => {
                type $concrete = i32;
                $body
            }
            BandElementKind::U32 => {
                type $concrete = u32;
                $body
            }
            BandElementKind::F32 => {
                type $concrete = f32;
                $body
            }
            BandElementKind::F64 => {
                type $concrete = f64;
                $body
            }
        }
    };
}

/// Inspects the first blob and computes exact Lerc1 value ranges.
///
/// # Errors
/// Returns an error when the first blob is malformed or unsupported.
pub fn inspect_first(blob: &[u8]) -> Result<BlobInfo> {
    inspect_first_with_options(blob, InspectOptions::default())
}

/// Inspects the first blob with configurable Lerc1 range computation.
///
/// # Errors
/// Returns an error when the first blob is malformed or unsupported.
pub fn inspect_first_with_options(blob: &[u8], options: InspectOptions) -> Result<BlobInfo> {
    if lerc1::is_lerc1(blob) {
        return lerc1::inspect_with_mask_options(blob, None, options.compute_value_range)
            .map(|(info, _)| info);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::inspect(blob, None);
    }
    Err(Error::InvalidMagic)
}

/// Inspects the first blob using `mask` when it omits its own validity mask.
///
/// # Errors
/// Returns an error for malformed input or a mismatched external mask.
pub fn inspect_first_with_mask(blob: &[u8], mask: &[u8]) -> Result<BlobInfo> {
    let (info, _) = inspect_first_mask_with_info(blob, Some(mask), Some(mask))?;
    Ok(info)
}

/// Strictly inspects one blob and rejects trailing bytes.
///
/// # Errors
/// Returns an error for malformed input or any trailing data.
pub fn get_blob_info(blob: &[u8]) -> Result<BlobInfo> {
    let info = inspect_first(blob)?;
    ensure_single_blob_consumed(blob.len(), info.blob_size, "get_blob_info", "inspect_first")?;
    Ok(info)
}

/// Strictly inspects one externally masked blob and rejects trailing bytes.
///
/// # Errors
/// Returns an error for malformed input, a mismatched mask, or trailing data.
pub fn get_blob_info_with_mask(blob: &[u8], mask: &[u8]) -> Result<BlobInfo> {
    let info = inspect_first_with_mask(blob, mask)?;
    ensure_single_blob_consumed(
        blob.len(),
        info.blob_size,
        "get_blob_info_with_mask",
        "inspect_first_with_mask",
    )?;
    Ok(info)
}

/// Counts validated blobs in a concatenated band payload.
///
/// # Errors
/// Returns an error if any blob boundary, header, or inherited mask is invalid.
pub fn get_band_count(blob: &[u8]) -> Result<usize> {
    let mut offset = 0usize;
    let mut count = 0usize;
    let mut lerc1_mask: Option<Vec<u8>> = None;

    while offset < blob.len() {
        let slice = &blob[offset..];
        let next_len = if lerc1::is_lerc1(slice) {
            let parsed = lerc1::parse(slice, lerc1_mask.as_deref())?;
            let next_len = parsed.info.blob_size;
            lerc1_mask = parsed.mask;
            next_len
        } else if lerc2::is_lerc2(slice) {
            let (parsed, _) = lerc2::parse(slice)?;
            parsed.info.blob_size
        } else {
            return Err(Error::InvalidMagic);
        };

        offset = checked_next_offset(offset, next_len, blob.len())?;
        count += 1;
    }

    Ok(count)
}

/// Decodes the first blob and permits trailing bytes.
///
/// # Errors
/// Returns an error when the first blob is malformed or unsupported.
pub fn decode_first(blob: &[u8]) -> Result<Decoded> {
    decode_first_with_masks(blob, None, None)
}

/// Decodes the first blob with an initial external mask and permits trailing bytes.
///
/// # Errors
/// Returns an error for malformed input or a mismatched mask.
pub fn decode_first_with_mask(blob: &[u8], mask: &[u8]) -> Result<Decoded> {
    decode_first_with_masks(blob, Some(mask), Some(mask))
}

/// Strictly decodes exactly one blob into its native sample type.
///
/// # Errors
/// Returns an error for malformed input, unsupported features, or trailing bytes.
pub fn decode(blob: &[u8]) -> Result<Decoded> {
    let decoded = decode_first(blob)?;
    ensure_single_blob_consumed(blob.len(), decoded.info.blob_size, "decode", "decode_first")?;
    Ok(decoded)
}

/// Strictly decodes one blob with an initial external mask.
///
/// # Errors
/// Returns an error for malformed input, a mismatched mask, or trailing bytes.
pub fn decode_with_mask(blob: &[u8], mask: &[u8]) -> Result<Decoded> {
    let decoded = decode_first_with_mask(blob, mask)?;
    ensure_single_blob_consumed(
        blob.len(),
        decoded.info.blob_size,
        "decode_with_mask",
        "decode_first_with_mask",
    )?;
    Ok(decoded)
}

/// Reads and decodes exactly one blob, leaving subsequent stream bytes unread.
///
/// # Errors
/// Returns an error for I/O failure, truncation, or invalid encoded data.
pub fn decode_from_reader<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<Decoded> {
    let blob = read_required_stream_blob(reader, None)?;
    decode(&blob)
}

/// Reads and decodes one blob with an initial external mask.
///
/// # Errors
/// Returns an error for I/O failure, truncation, invalid data, or a mismatched mask.
pub fn decode_from_reader_with_mask<R: std::io::Read + ?Sized>(
    reader: &mut R,
    mask: &[u8],
) -> Result<Decoded> {
    let blob = read_required_stream_blob(reader, Some(mask))?;
    decode_with_mask(&blob, mask)
}

/// Decodes every concatenated blob as a native-typed band set.
///
/// # Errors
/// Returns an error if any band is invalid or band shapes disagree.
pub fn decode_band_set(blob: &[u8]) -> Result<DecodedBandSet> {
    decode_band_set_with_lerc2_mask(blob, None)
}

/// Decodes a band set, seeding the first blob with an external mask.
///
/// # Errors
/// Returns an error for invalid blobs, masks, or mismatched band shapes.
pub fn decode_band_set_with_mask(blob: &[u8], mask: &[u8]) -> Result<DecodedBandSet> {
    decode_band_set_with_lerc2_mask(blob, Some(mask))
}

/// Reads concatenated blobs through clean EOF and decodes them as a band set.
///
/// # Errors
/// Returns an error for I/O failure, truncation, invalid blobs, or mismatched shapes.
pub fn decode_band_set_from_reader<R: std::io::Read + ?Sized>(
    reader: &mut R,
) -> Result<DecodedBandSet> {
    decode_band_set_from_reader_impl(reader, None)
}

/// Streams a band set through EOF with an initial external mask.
///
/// # Errors
/// Returns an error for I/O failure, invalid data, masks, or band shapes.
pub fn decode_band_set_from_reader_with_mask<R: std::io::Read + ?Sized>(
    reader: &mut R,
    mask: &[u8],
) -> Result<DecodedBandSet> {
    decode_band_set_from_reader_impl(reader, Some(mask))
}

/// Decodes a band set into a homogeneous vector in the requested layout.
///
/// # Errors
/// Returns an error for invalid input or an incompatible output element type.
pub fn decode_band_set_vec<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    decode_band_set_owned(blob, layout, None)
}

/// Decodes an externally masked band set into a homogeneous vector.
///
/// # Errors
/// Returns an error for invalid input, mask mismatch, or incompatible element type.
pub fn decode_band_set_vec_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    decode_band_set_owned(blob, layout, Some(mask))
}

/// Decodes a band set directly into an exactly sized output slice.
///
/// # Errors
/// Returns an error for invalid input, incompatible types, or the wrong output length.
pub fn decode_band_set_into<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    decode_band_set_into_direct(blob, layout, None, out)
}

/// Decodes an externally masked band set directly into an output slice.
///
/// # Errors
/// Returns an error for invalid input, mask mismatch, type mismatch, or wrong length.
pub fn decode_band_set_into_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    decode_band_set_into_direct(blob, layout, Some(mask), out)
}

#[cfg(feature = "ndarray")]
/// Decodes a band set into an interleaved ndarray.
///
/// # Errors
/// Returns an error for invalid data, incompatible types, or shape construction failure.
pub fn decode_band_set_ndarray<T: BandElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

#[cfg(feature = "ndarray")]
/// Decodes an externally masked band set into an interleaved ndarray.
///
/// # Errors
/// Returns an error for invalid data, mask mismatch, type mismatch, or shape failure.
pub fn decode_band_set_ndarray_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_and_mask(blob, BandLayout::Interleaved, mask)
}

#[cfg(feature = "ndarray")]
/// Decodes a band set into an ndarray using `layout`.
///
/// # Errors
/// Returns an error for invalid data, incompatible types, or shape failure.
pub fn decode_band_set_ndarray_with_layout<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_impl(blob, layout, None)
}

#[cfg(feature = "ndarray")]
/// Decodes an externally masked band set into an ndarray using `layout`.
///
/// # Errors
/// Returns an error for invalid data, masks, types, or shape construction.
pub fn decode_band_set_ndarray_with_layout_and_mask<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    mask: &[u8],
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_impl(blob, layout, Some(mask))
}

#[cfg(feature = "ndarray")]
/// Decodes and promotes a band set into an interleaved `f64` ndarray.
///
/// # Errors
/// Returns an error for invalid data or shape construction failure.
pub fn decode_band_set_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_layout(blob, BandLayout::Interleaved)
}

#[cfg(feature = "ndarray")]
/// Decodes and promotes an externally masked band set into an `f64` ndarray.
///
/// # Errors
/// Returns an error for invalid data, mask mismatch, or shape failure.
pub fn decode_band_set_ndarray_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_layout_and_mask(blob, BandLayout::Interleaved, mask)
}

#[cfg(feature = "ndarray")]
/// Decodes and promotes a band set into an `f64` ndarray using `layout`.
///
/// # Errors
/// Returns an error for invalid data or shape construction failure.
pub fn decode_band_set_ndarray_f64_with_layout(
    blob: &[u8],
    layout: BandLayout,
) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_optional_mask(blob, layout, None)
}

#[cfg(feature = "ndarray")]
/// Decodes and promotes an externally masked band set using `layout`.
///
/// # Errors
/// Returns an error for invalid data, mask mismatch, or shape failure.
pub fn decode_band_set_ndarray_f64_with_layout_and_mask(
    blob: &[u8],
    layout: BandLayout,
    mask: &[u8],
) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_optional_mask(blob, layout, Some(mask))
}

#[cfg(feature = "ndarray")]
/// Returns decoded band masks as an ndarray without decoding pixel values.
///
/// # Errors
/// Returns an error for invalid blobs, inconsistent masks, or shape failure.
pub fn decode_band_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, band_masks) = inspect_band_masks(blob, None)?;
    crate::types::band_masks_into_ndarray(info, band_masks)
}

#[cfg(feature = "ndarray")]
/// Returns band masks seeded by an external mask as an ndarray.
///
/// # Errors
/// Returns an error for invalid blobs, masks, or shape construction.
pub fn decode_band_mask_ndarray_with_mask(blob: &[u8], mask: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, band_masks) = inspect_band_masks(blob, Some(mask))?;
    crate::types::band_masks_into_ndarray(info, band_masks)
}

fn decode_band_set_with_lerc2_mask(
    blob: &[u8],
    initial_mask: Option<&[u8]>,
) -> Result<DecodedBandSet> {
    let mut offset = 0usize;
    let mut bands = Vec::new();
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask = initial_mask.map(<[u8]>::to_vec);
    let mut lerc2_mask = initial_mask.map(<[u8]>::to_vec);

    while offset < blob.len() {
        let decoded = decode_first_with_masks(
            &blob[offset..],
            lerc1_mask.as_deref(),
            lerc2_mask.as_deref(),
        )?;

        if lerc1::is_lerc1(&blob[offset..]) {
            lerc1_mask = decoded.mask.clone();
            lerc2_mask = None;
        } else {
            lerc2_mask = decoded.mask.clone();
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, decoded.info.blob_size, blob.len())?;
        infos.push(decoded.info);
        bands.push(decoded.pixels);
        band_masks.push(decoded.mask);
    }

    Ok(DecodedBandSet {
        info: BandSetInfo::new(infos)?,
        bands,
        band_masks,
    })
}

#[cfg(feature = "ndarray")]
fn decode_band_set_ndarray_with_layout_impl<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
) -> Result<ArrayD<T>> {
    let (info, values) = decode_band_set_owned(blob, layout, initial_mask)?;
    let shape = info.ndarray_shape_for_layout(layout);
    ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values).map_err(|e| {
        Error::invalid_blob(format!(
            "failed to build ndarray from decoded band set: {e}"
        ))
    })
}

#[cfg(feature = "ndarray")]
fn decode_band_set_ndarray_f64_with_optional_mask(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
) -> Result<ArrayD<f64>> {
    let band_info = decode_band_set_to_f64_info(blob, layout, initial_mask)?;
    let shape = band_info.0.ndarray_shape_for_layout(layout);
    ArrayD::from_shape_vec(ndarray::IxDyn(&shape), band_info.1).map_err(|e| {
        Error::invalid_blob(format!(
            "failed to build ndarray from decoded band set: {e}"
        ))
    })
}

/// Strictly decodes one blob and promotes every sample to `f64`.
///
/// # Errors
/// Returns an error for malformed data, unsupported features, or trailing bytes.
pub fn decode_to_f64(blob: &[u8]) -> Result<DecodedF64> {
    let decoded = decode_first_to_f64(blob)?;
    ensure_single_blob_consumed(
        blob.len(),
        decoded.info.blob_size,
        "decode_to_f64",
        "decode_first_to_f64",
    )?;
    Ok(decoded)
}

/// Strictly decodes one externally masked blob as promoted `f64` samples.
///
/// # Errors
/// Returns an error for invalid input, mask mismatch, or trailing bytes.
pub fn decode_to_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<DecodedF64> {
    let decoded = decode_first_to_f64_with_mask(blob, mask)?;
    ensure_single_blob_consumed(
        blob.len(),
        decoded.info.blob_size,
        "decode_to_f64_with_mask",
        "decode_first_to_f64_with_mask",
    )?;
    Ok(decoded)
}

/// Reads and decodes one blob as `f64`, leaving later stream bytes unread.
///
/// # Errors
/// Returns an error for I/O failure, truncation, or invalid data.
pub fn decode_to_f64_from_reader<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<DecodedF64> {
    let blob = read_required_stream_blob(reader, None)?;
    decode_to_f64(&blob)
}

/// Reads and decodes one externally masked blob as `f64`.
///
/// # Errors
/// Returns an error for I/O failure, invalid data, or mask mismatch.
pub fn decode_to_f64_from_reader_with_mask<R: std::io::Read + ?Sized>(
    reader: &mut R,
    mask: &[u8],
) -> Result<DecodedF64> {
    let blob = read_required_stream_blob(reader, Some(mask))?;
    decode_to_f64_with_mask(&blob, mask)
}

/// Decodes the first blob as promoted `f64` samples and permits trailing bytes.
///
/// # Errors
/// Returns an error when the first blob is malformed or unsupported.
pub fn decode_first_to_f64(blob: &[u8]) -> Result<DecodedF64> {
    decode_first_f64(blob)
}

/// Decodes the first externally masked blob as `f64` and permits trailing bytes.
///
/// # Errors
/// Returns an error for invalid data or a mismatched mask.
pub fn decode_first_to_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<DecodedF64> {
    decode_first_f64_with_masks(blob, Some(mask), Some(mask))
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one native-typed blob into an ndarray.
///
/// # Errors
/// Returns an error for invalid data, incompatible output type, or shape failure.
pub fn decode_ndarray<T: NdArrayElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode(blob)?.into_ndarray()
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one externally masked blob into an ndarray.
///
/// # Errors
/// Returns an error for invalid data, masks, types, or shape construction.
pub fn decode_ndarray_with_mask<T: NdArrayElement>(blob: &[u8], mask: &[u8]) -> Result<ArrayD<T>> {
    decode_with_mask(blob, mask)?.into_ndarray()
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one blob into a promoted `f64` ndarray.
///
/// # Errors
/// Returns an error for invalid data or shape construction failure.
pub fn decode_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_to_f64(blob)?.into_ndarray()
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one externally masked blob into an `f64` ndarray.
///
/// # Errors
/// Returns an error for invalid data, mask mismatch, or shape failure.
pub fn decode_ndarray_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<ArrayD<f64>> {
    decode_to_f64_with_mask(blob, mask)?.into_ndarray()
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one blob's validity mask as an ndarray.
///
/// # Errors
/// Returns an error for invalid data, trailing bytes, or shape construction.
pub fn decode_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, mask) = inspect_first_mask_with_info(blob, None, None)?;
    ensure_single_blob_consumed(
        blob.len(),
        info.blob_size,
        "decode_mask_ndarray",
        "inspect_first",
    )?;
    mask_to_ndarray(&info, mask)
}

#[cfg(feature = "ndarray")]
/// Strictly decodes one blob's externally seeded mask as an ndarray.
///
/// # Errors
/// Returns an error for invalid data, mask mismatch, trailing bytes, or shape failure.
pub fn decode_mask_ndarray_with_mask(blob: &[u8], mask: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, decoded_mask) = inspect_first_mask_with_info(blob, Some(mask), Some(mask))?;
    ensure_single_blob_consumed(
        blob.len(),
        info.blob_size,
        "decode_mask_ndarray_with_mask",
        "inspect_first_with_mask",
    )?;
    mask_to_ndarray(&info, decoded_mask)
}

fn decode_first_with_masks(
    blob: &[u8],
    lerc1_shared_mask: Option<&[u8]>,
    lerc2_shared_mask: Option<&[u8]>,
) -> Result<Decoded> {
    if lerc1::is_lerc1(blob) {
        return lerc1::decode(blob, lerc1_shared_mask);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::decode(blob, lerc2_shared_mask);
    }
    Err(Error::InvalidMagic)
}

fn inspect_first_mask_with_info(
    blob: &[u8],
    lerc1_shared_mask: Option<&[u8]>,
    lerc2_shared_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    if lerc1::is_lerc1(blob) {
        return lerc1::inspect_mask(blob, lerc1_shared_mask);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::inspect_with_mask(blob, lerc2_shared_mask);
    }
    Err(Error::InvalidMagic)
}

fn decode_first_f64(blob: &[u8]) -> Result<DecodedF64> {
    decode_first_f64_with_masks(blob, None, None)
}

fn decode_first_f64_with_masks(
    blob: &[u8],
    lerc1_shared_mask: Option<&[u8]>,
    lerc2_shared_mask: Option<&[u8]>,
) -> Result<DecodedF64> {
    if lerc1::is_lerc1(blob) {
        return lerc1::decode_f64(blob, lerc1_shared_mask);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::decode_f64(blob, lerc2_shared_mask);
    }
    Err(Error::InvalidMagic)
}

fn decode_band_set_owned<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    let band_info = scan_band_infos(blob, initial_mask)?;
    decode_band_set_owned_direct(blob, layout, band_info, initial_mask)
}

fn decode_band_set_into_direct<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
    out: &mut [T],
) -> Result<BandSetInfo> {
    dispatch_band_element!(T, |Concrete| {
        decode_band_set_into_impl::<Concrete>(
            blob,
            layout,
            initial_mask,
            cast_slice_mut::<T, Concrete>(out),
        )
    })
}

fn decode_band_set_owned_direct<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    band_info: BandSetInfo,
    initial_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    dispatch_band_element!(T, |Concrete| {
        decode_band_set_owned_direct_impl::<Concrete>(blob, layout, band_info, initial_mask)
            .map(|(info, values)| (info, cast_vec::<T, Concrete>(values)))
    })
}

fn decode_band_set_into_impl<T: Sample + BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let band_info = scan_band_infos(blob, initial_mask)?;
    decode_band_set_into_impl_with_info(blob, layout, initial_mask, &band_info, out)
}

fn decode_band_set_into_impl_with_info<T: Sample + BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
    band_info: &BandSetInfo,
    out: &mut [T],
) -> Result<BandSetInfo> {
    #[cfg(feature = "rayon")]
    {
        if band_info.band_count() > 1 {
            return decode_band_set_into_parallel(blob, layout, initial_mask, band_info, out);
        }
    }
    decode_band_set_into_sequential(blob, layout, initial_mask, band_info, out)
}

fn decode_band_set_into_sequential<T: Sample + BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
    band_info: &BandSetInfo,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let band_count = band_info.band_count();
    let expected_len = band_info.value_count()?;
    if out.len() != expected_len {
        return Err(Error::InvalidArgument(
            "output slice length does not match decoded band set length",
        ));
    }

    let pixel_count = band_info.bands[0].pixel_count()?;
    let depth = band_info.depth() as usize;
    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut lerc1_mask = initial_mask.map(<[u8]>::to_vec);
    let mut lerc2_mask = initial_mask.map(<[u8]>::to_vec);
    let mut decoded_infos = Vec::with_capacity(band_count);

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let mut sink = BandSink::new(out, pixel_count, depth, band_index, band_count, layout);
        let (info, mask) = if is_lerc1 {
            lerc1::decode_into(slice, lerc1_mask.as_deref(), &mut sink)?
        } else if lerc2::is_lerc2(slice) {
            lerc2::decode_into(slice, lerc2_mask.as_deref(), &mut sink)?
        } else {
            return Err(Error::InvalidMagic);
        };

        if is_lerc1 {
            lerc1_mask = mask;
            lerc2_mask = None;
        } else {
            lerc2_mask = mask;
            lerc1_mask = None;
        }

        let blob_size = info.blob_size;
        decoded_infos.push(info);
        offset = checked_next_offset(offset, blob_size, blob.len())?;
        band_index += 1;
    }

    BandSetInfo::new(decoded_infos)
}

#[cfg(feature = "rayon")]
#[derive(Debug, Clone, Copy)]
enum BandFormat {
    Lerc1,
    Lerc2,
}

#[cfg(feature = "rayon")]
#[derive(Debug)]
struct ParallelBand<'a> {
    blob: &'a [u8],
    format: BandFormat,
    inherited_mask: Option<std::sync::Arc<[u8]>>,
}

#[cfg(feature = "rayon")]
fn scan_parallel_bands<'a>(
    blob: &'a [u8],
    initial_mask: Option<&[u8]>,
) -> Result<Vec<ParallelBand<'a>>> {
    use std::sync::Arc;

    let mut offset = 0usize;
    let mut bands = Vec::new();
    let mut lerc1_mask = initial_mask.map(Arc::<[u8]>::from);
    let mut lerc2_mask = initial_mask.map(Arc::<[u8]>::from);

    while offset < blob.len() {
        let slice = &blob[offset..];
        let format = if lerc1::is_lerc1(slice) {
            BandFormat::Lerc1
        } else if lerc2::is_lerc2(slice) {
            BandFormat::Lerc2
        } else {
            return Err(Error::InvalidMagic);
        };
        let inherited_mask = match format {
            BandFormat::Lerc1 => lerc1_mask.clone(),
            BandFormat::Lerc2 => lerc2_mask.clone(),
        };
        let (info, decoded_mask) =
            inspect_first_mask_with_info(slice, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;
        let next = checked_next_offset(offset, info.blob_size, blob.len())?;

        bands.push(ParallelBand {
            blob: &blob[offset..next],
            format,
            inherited_mask,
        });

        match format {
            BandFormat::Lerc1 => {
                lerc1_mask = decoded_mask.map(Arc::from);
                lerc2_mask = None;
            }
            BandFormat::Lerc2 => {
                lerc2_mask = decoded_mask.map(Arc::from);
                lerc1_mask = None;
            }
        }
        offset = next;
    }

    Ok(bands)
}

#[cfg(feature = "rayon")]
fn decode_parallel_band<T: Sample>(
    band: &ParallelBand<'_>,
    out: &mut [T],
    depth: usize,
) -> Result<BlobInfo> {
    let mut writer = PixelDataWriter::new(out, depth);
    let (info, _) = match band.format {
        BandFormat::Lerc1 => {
            lerc1::decode_into(band.blob, band.inherited_mask.as_deref(), &mut writer)?
        }
        BandFormat::Lerc2 => {
            lerc2::decode_into(band.blob, band.inherited_mask.as_deref(), &mut writer)?
        }
    };
    Ok(info)
}

#[cfg(feature = "rayon")]
fn decode_band_set_into_parallel<T: Sample + BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
    band_info: &BandSetInfo,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let expected_len = band_info.value_count()?;
    if out.len() != expected_len {
        return Err(Error::InvalidArgument(
            "output slice length does not match decoded band set length",
        ));
    }

    let bands = scan_parallel_bands(blob, initial_mask)?;
    let band_count = band_info.band_count();
    if bands.len() != band_count {
        return Err(Error::Internal(
            "parallel band scan disagrees with decoded band metadata",
        ));
    }
    let pixel_count = band_info.bands[0].pixel_count()?;
    let depth = band_info.depth() as usize;
    let band_len = band_info.bands[0].sample_count()?;

    let infos = match layout {
        BandLayout::Bsq => out
            .par_chunks_mut(band_len)
            .zip(bands.par_iter())
            .map(|(band_out, band)| decode_parallel_band(band, band_out, depth))
            .collect::<Result<Vec<_>>>()?,
        BandLayout::Interleaved => {
            let decoded = bands
                .par_iter()
                .map(|band| {
                    let mut values = default_vec(band_len, "parallel decoded band")?;
                    let info = decode_parallel_band(band, &mut values, depth)?;
                    Ok((info, values))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut infos = Vec::with_capacity(band_count);
            for (band_index, (info, values)) in decoded.into_iter().enumerate() {
                copy_band_values_into_slice(
                    out,
                    &values,
                    pixel_count,
                    depth,
                    band_index,
                    band_count,
                    layout,
                )?;
                infos.push(info);
            }
            infos
        }
    };

    BandSetInfo::new(infos)
}

#[cfg(feature = "ndarray")]
fn decode_band_set_to_f64_info(
    blob: &[u8],
    layout: BandLayout,
    initial_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<f64>)> {
    let band_info = scan_band_infos(blob, initial_mask)?;
    decode_band_set_owned_direct_impl::<f64>(blob, layout, band_info, initial_mask)
}

#[cfg(feature = "ndarray")]
fn inspect_band_masks(
    blob: &[u8],
    initial_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<Option<Vec<u8>>>)> {
    let mut offset = 0usize;
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask = initial_mask.map(<[u8]>::to_vec);
    let mut lerc2_mask = initial_mask.map(<[u8]>::to_vec);

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let (info, mask) =
            inspect_first_mask_with_info(slice, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;

        if is_lerc1 {
            lerc1_mask = mask.clone();
            lerc2_mask = None;
        } else {
            lerc2_mask = mask.clone();
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, info.blob_size, blob.len())?;
        infos.push(info);
        band_masks.push(mask);
    }

    Ok((BandSetInfo::new(infos)?, band_masks))
}

fn scan_band_infos(blob: &[u8], initial_mask: Option<&[u8]>) -> Result<BandSetInfo> {
    let mut offset = 0usize;
    let mut infos = Vec::new();
    let mut lerc1_mask = initial_mask.map(<[u8]>::to_vec);

    while offset < blob.len() {
        let slice = &blob[offset..];
        let (info, next_lerc1_mask) = if lerc1::is_lerc1(slice) {
            let parsed = lerc1::parse(slice, lerc1_mask.as_deref())?;
            let info = parsed.info;
            let next_mask = parsed.mask;
            (info, next_mask)
        } else if lerc2::is_lerc2(slice) {
            let (parsed, _) = lerc2::parse(slice)?;
            (parsed.info, None)
        } else {
            return Err(Error::InvalidMagic);
        };

        offset = checked_next_offset(offset, info.blob_size, blob.len())?;
        lerc1_mask = next_lerc1_mask;
        infos.push(info);
    }

    BandSetInfo::new(infos)
}

fn read_required_stream_blob<R: std::io::Read + ?Sized>(
    reader: &mut R,
    lerc1_shared_mask: Option<&[u8]>,
) -> Result<Vec<u8>> {
    stream::read_next_blob(reader, lerc1_shared_mask)?.ok_or(Error::Truncated {
        offset: 0,
        needed: 10,
        available: 0,
    })
}

fn decode_band_set_from_reader_impl<R: std::io::Read + ?Sized>(
    reader: &mut R,
    initial_mask: Option<&[u8]>,
) -> Result<DecodedBandSet> {
    let mut bands = Vec::new();
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask = initial_mask.map(<[u8]>::to_vec);
    let mut lerc2_mask = initial_mask.map(<[u8]>::to_vec);

    while let Some(blob) = stream::read_next_blob(reader, lerc1_mask.as_deref())? {
        let is_lerc1 = lerc1::is_lerc1(&blob);
        let decoded = decode_first_with_masks(&blob, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;

        if is_lerc1 {
            lerc1_mask = decoded.mask.clone();
            lerc2_mask = None;
        } else {
            lerc2_mask = decoded.mask.clone();
            lerc1_mask = None;
        }
        infos.push(decoded.info);
        bands.push(decoded.pixels);
        band_masks.push(decoded.mask);
    }

    Ok(DecodedBandSet {
        info: BandSetInfo::new(infos)?,
        bands,
        band_masks,
    })
}

fn ensure_single_blob_consumed(
    blob_len: usize,
    decoded_len: usize,
    strict_api: &str,
    permissive_api: &str,
) -> Result<()> {
    if blob_len == decoded_len {
        return Ok(());
    }
    Err(Error::invalid_blob(format!(
        "{strict_api} only accepts a single LERC blob; found {} trailing bytes, use {permissive_api} for first-blob decoding or decode_band_set for concatenated rasters",
        blob_len - decoded_len
    )))
}

fn checked_next_offset(offset: usize, next_len: usize, total_len: usize) -> Result<usize> {
    let next = offset
        .checked_add(next_len)
        .ok_or(Error::SizeOverflow("concatenated band offset"))?;
    if next <= offset || next > total_len {
        return Err(Error::invalid_blob("invalid concatenated band blob size"));
    }
    Ok(next)
}

#[cfg(feature = "ndarray")]
fn mask_to_ndarray(info: &BlobInfo, mask: Option<Vec<u8>>) -> Result<Option<ArrayD<u8>>> {
    let shape = info.mask_ndarray_shape();
    mask.map(|mask| {
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape), mask).map_err(|e| {
            Error::invalid_blob(format!("failed to build ndarray from decoded mask: {e}"))
        })
    })
    .transpose()
}

fn decode_band_set_owned_direct_impl<T: Sample + BandElement + Copy + Default>(
    blob: &[u8],
    layout: BandLayout,
    band_info: BandSetInfo,
    initial_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    let mut values = default_vec(band_info.value_count()?, "decoded band set")?;
    let decoded_info =
        decode_band_set_into_impl_with_info(blob, layout, initial_mask, &band_info, &mut values)?;
    Ok((decoded_info, values))
}

fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    // SAFETY: callers only reach this helper through dispatch_band_element!, where
    // T and U are the same concrete primitive type selected by BandElementKind.
    unsafe { &mut *(slice as *mut [T] as *mut [U]) }
}

fn cast_vec<T, U>(values: Vec<U>) -> Vec<T> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let len = values.len();
    let cap = values.capacity();
    let ptr = values.as_ptr() as *mut T;
    std::mem::forget(values);
    // SAFETY: callers only reach this helper through dispatch_band_element!, where
    // T and U are the same concrete primitive type. The allocation, length, and
    // capacity therefore remain valid for Vec<T>.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}
