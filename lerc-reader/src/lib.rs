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
//! - decode directly into `ndarray::ArrayD` with [`decode_ndarray`]

mod allocation;
mod bitstuff;
mod huffman;
mod io;
mod lerc1;
mod lerc2;
mod pixel;
mod types;

#[cfg(test)]
mod tests;

use lerc_band_materialize::{BandLayout as MaterializeLayout, BandMaterializer, BandSink};
use lerc_core::{BandLayout, BandSetInfo, BlobInfo, Error, Result};
use ndarray::ArrayD;

use crate::pixel::Sample;
pub use crate::types::{
    into_band_mask_ndarray, BandElement, BandElementKind, Decoded, DecodedBandSet, DecodedF64,
    NdArrayElement,
};

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

pub fn inspect_first(blob: &[u8]) -> Result<BlobInfo> {
    if lerc1::is_lerc1(blob) {
        return lerc1::inspect(blob, None);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::inspect(blob, None);
    }
    Err(Error::InvalidMagic)
}

pub fn inspect_first_with_mask(blob: &[u8], mask: &[u8]) -> Result<BlobInfo> {
    let (info, _) = inspect_first_mask_with_info(blob, Some(mask), Some(mask))?;
    Ok(info)
}

pub fn get_blob_info(blob: &[u8]) -> Result<BlobInfo> {
    let info = inspect_first(blob)?;
    ensure_single_blob_consumed(blob.len(), info.blob_size, "get_blob_info", "inspect_first")?;
    Ok(info)
}

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

pub fn decode_first(blob: &[u8]) -> Result<Decoded> {
    decode_first_with_masks(blob, None, None)
}

pub fn decode_first_with_mask(blob: &[u8], mask: &[u8]) -> Result<Decoded> {
    decode_first_with_masks(blob, Some(mask), Some(mask))
}

pub fn decode(blob: &[u8]) -> Result<Decoded> {
    let decoded = decode_first(blob)?;
    ensure_single_blob_consumed(blob.len(), decoded.info.blob_size, "decode", "decode_first")?;
    Ok(decoded)
}

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

pub fn decode_band_set(blob: &[u8]) -> Result<DecodedBandSet> {
    decode_band_set_with_lerc2_mask(blob, None)
}

pub fn decode_band_set_with_mask(blob: &[u8], mask: &[u8]) -> Result<DecodedBandSet> {
    decode_band_set_with_lerc2_mask(blob, Some(mask))
}

pub fn decode_band_set_vec<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    decode_band_set_owned(blob, layout, None)
}

pub fn decode_band_set_vec_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    decode_band_set_owned(blob, layout, Some(mask))
}

pub fn decode_band_set_into<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    decode_band_set_into_direct(blob, layout, None, out)
}

pub fn decode_band_set_into_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    decode_band_set_into_direct(blob, layout, Some(mask), out)
}

pub fn decode_band_set_ndarray<T: BandElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

pub fn decode_band_set_ndarray_with_mask<T: BandElement>(
    blob: &[u8],
    mask: &[u8],
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_and_mask(blob, BandLayout::Interleaved, mask)
}

pub fn decode_band_set_ndarray_with_layout<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_impl(blob, layout, None)
}

pub fn decode_band_set_ndarray_with_layout_and_mask<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    mask: &[u8],
) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout_impl(blob, layout, Some(mask))
}

pub fn decode_band_set_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_optional_mask(blob, None)
}

pub fn decode_band_set_ndarray_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_f64_with_optional_mask(blob, Some(mask))
}

pub fn decode_band_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, band_masks) = inspect_band_masks(blob, None)?;
    into_band_mask_ndarray(info, band_masks)
}

pub fn decode_band_mask_ndarray_with_mask(blob: &[u8], mask: &[u8]) -> Result<Option<ArrayD<u8>>> {
    let (info, band_masks) = inspect_band_masks(blob, Some(mask))?;
    into_band_mask_ndarray(info, band_masks)
}

fn decode_band_set_with_lerc2_mask(
    blob: &[u8],
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<DecodedBandSet> {
    let mut offset = 0usize;
    let mut bands = Vec::new();
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask = initial_lerc2_mask.map(|mask| mask.to_vec());

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

fn decode_band_set_ndarray_with_layout_impl<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<ArrayD<T>> {
    let (info, values) = decode_band_set_owned(blob, layout, initial_lerc2_mask)?;
    let shape = info.ndarray_shape_for_layout(layout);
    ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values).map_err(|e| {
        Error::InvalidBlob(format!(
            "failed to build ndarray from decoded band set: {e}"
        ))
    })
}

fn decode_band_set_ndarray_f64_with_optional_mask(
    blob: &[u8],
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<ArrayD<f64>> {
    let band_info = decode_band_set_to_f64_info(blob, BandLayout::Interleaved, initial_lerc2_mask)?;
    let shape = band_info
        .0
        .ndarray_shape_for_layout(BandLayout::Interleaved);
    ArrayD::from_shape_vec(ndarray::IxDyn(&shape), band_info.1).map_err(|e| {
        Error::InvalidBlob(format!(
            "failed to build ndarray from decoded band set: {e}"
        ))
    })
}

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

pub fn decode_first_to_f64(blob: &[u8]) -> Result<DecodedF64> {
    decode_first_f64(blob)
}

pub fn decode_first_to_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<DecodedF64> {
    decode_first_f64_with_masks(blob, Some(mask), Some(mask))
}

pub fn decode_ndarray<T: NdArrayElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode(blob)?.into_ndarray()
}

pub fn decode_ndarray_with_mask<T: NdArrayElement>(blob: &[u8], mask: &[u8]) -> Result<ArrayD<T>> {
    decode_with_mask(blob, mask)?.into_ndarray()
}

pub fn decode_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_to_f64(blob)?.into_ndarray()
}

pub fn decode_ndarray_f64_with_mask(blob: &[u8], mask: &[u8]) -> Result<ArrayD<f64>> {
    decode_to_f64_with_mask(blob, mask)?.into_ndarray()
}

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
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    let band_info = scan_band_infos(blob)?;
    decode_band_set_owned_direct(blob, layout, band_info, initial_lerc2_mask)
}

fn decode_band_set_into_direct<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_lerc2_mask: Option<&[u8]>,
    out: &mut [T],
) -> Result<BandSetInfo> {
    dispatch_band_element!(T, |Concrete| {
        decode_band_set_into_impl::<Concrete>(
            blob,
            layout,
            initial_lerc2_mask,
            cast_slice_mut::<T, Concrete>(out),
        )
    })
}

fn decode_band_set_owned_direct<T: BandElement>(
    blob: &[u8],
    layout: BandLayout,
    band_info: BandSetInfo,
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    dispatch_band_element!(T, |Concrete| {
        decode_band_set_owned_direct_impl::<Concrete>(blob, layout, band_info, initial_lerc2_mask)
            .map(|(info, values)| (info, cast_vec::<T, Concrete>(values)))
    })
}

fn decode_band_set_into_impl<T: Sample + NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_lerc2_mask: Option<&[u8]>,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let band_info = scan_band_infos(blob)?;
    decode_band_set_into_impl_with_info(blob, layout, initial_lerc2_mask, &band_info, out)?;
    Ok(band_info)
}

fn decode_band_set_into_impl_with_info<T: Sample + NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
    initial_lerc2_mask: Option<&[u8]>,
    band_info: &BandSetInfo,
    out: &mut [T],
) -> Result<()> {
    let band_count = band_info.band_count();
    let expected_len = band_info.value_count()?;
    if out.len() != expected_len {
        return Err(Error::InvalidBlob(format!(
            "output slice length {} does not match decoded band set length {}",
            out.len(),
            expected_len
        )));
    }

    let pixel_count = band_info.bands[0].pixel_count()?;
    let depth = band_info.depth() as usize;
    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask = initial_lerc2_mask.map(|mask| mask.to_vec());

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let mut sink = BandSink::new(
            out,
            pixel_count,
            depth,
            band_index,
            band_count,
            materialize_layout(layout),
        );
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

        offset = checked_next_offset(offset, info.blob_size, blob.len())?;
        band_index += 1;
    }

    Ok(())
}

fn decode_band_set_to_f64_info(
    blob: &[u8],
    layout: BandLayout,
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<f64>)> {
    let band_info = scan_band_infos(blob)?;
    decode_band_set_owned_direct_impl::<f64>(blob, layout, band_info, initial_lerc2_mask)
}

fn inspect_band_masks(
    blob: &[u8],
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<Option<Vec<u8>>>)> {
    let mut offset = 0usize;
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask = initial_lerc2_mask.map(|mask| mask.to_vec());

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

fn scan_band_infos(blob: &[u8]) -> Result<BandSetInfo> {
    let mut offset = 0usize;
    let mut infos = Vec::new();
    let mut lerc1_mask: Option<Vec<u8>> = None;

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

fn ensure_single_blob_consumed(
    blob_len: usize,
    decoded_len: usize,
    strict_api: &str,
    permissive_api: &str,
) -> Result<()> {
    if blob_len == decoded_len {
        return Ok(());
    }
    Err(Error::InvalidBlob(format!(
        "{strict_api} only accepts a single LERC blob; found {} trailing bytes, use {permissive_api} for first-blob decoding or decode_band_set for concatenated rasters",
        blob_len - decoded_len
    )))
}

fn checked_next_offset(offset: usize, next_len: usize, total_len: usize) -> Result<usize> {
    let next = offset
        .checked_add(next_len)
        .ok_or_else(|| Error::InvalidBlob("band offset overflow".into()))?;
    if next <= offset || next > total_len {
        return Err(Error::InvalidBlob(
            "invalid concatenated band blob size".into(),
        ));
    }
    Ok(next)
}

fn materialize_layout(layout: BandLayout) -> MaterializeLayout {
    match layout {
        BandLayout::Interleaved => MaterializeLayout::Interleaved,
        BandLayout::Bsq => MaterializeLayout::Bsq,
    }
}

fn materialize_error(err: lerc_band_materialize::MaterializeError) -> Error {
    Error::InvalidBlob(err.to_string())
}

fn mask_to_ndarray(info: &BlobInfo, mask: Option<Vec<u8>>) -> Result<Option<ArrayD<u8>>> {
    let shape = info.mask_ndarray_shape();
    mask.map(|mask| {
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape), mask).map_err(|e| {
            Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {e}"))
        })
    })
    .transpose()
}

fn decode_band_set_owned_direct_impl<T: Sample + NdArrayElement + Copy + Default>(
    blob: &[u8],
    layout: BandLayout,
    band_info: BandSetInfo,
    initial_lerc2_mask: Option<&[u8]>,
) -> Result<(BandSetInfo, Vec<T>)> {
    let expected_len = band_info.value_count()?;
    if expected_len == 0 {
        return Ok((band_info, Vec::new()));
    }

    let pixel_count = band_info.bands[0].pixel_count()?;
    let depth = band_info.depth() as usize;
    let band_count = band_info.band_count();
    let mut materializer =
        BandMaterializer::new(pixel_count, depth, band_count, materialize_layout(layout))
            .map_err(materialize_error)?;
    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask = initial_lerc2_mask.map(|mask| mask.to_vec());

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let (info, mask) = {
            let mut writer = materializer
                .band_writer(band_index)
                .map_err(materialize_error)?;
            let decoded = if is_lerc1 {
                lerc1::decode_into(slice, lerc1_mask.as_deref(), &mut writer)?
            } else if lerc2::is_lerc2(slice) {
                lerc2::decode_into(slice, lerc2_mask.as_deref(), &mut writer)?
            } else {
                return Err(Error::InvalidMagic);
            };
            writer.finish().map_err(materialize_error)?;
            decoded
        };

        if is_lerc1 {
            lerc1_mask = mask;
            lerc2_mask = None;
        } else {
            lerc2_mask = mask;
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, info.blob_size, blob.len())?;
        band_index += 1;
    }

    Ok((band_info, materializer.finish().map_err(materialize_error)?))
}

fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    unsafe { &mut *(slice as *mut [T] as *mut [U]) }
}

fn cast_vec<T, U>(values: Vec<U>) -> Vec<T> {
    let len = values.len();
    let cap = values.capacity();
    let ptr = values.as_ptr() as *mut T;
    std::mem::forget(values);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}
