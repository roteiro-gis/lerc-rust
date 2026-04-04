//! Pure-Rust LERC decoder.
//!
//! The public API distinguishes strict single-blob entry points from
//! concatenated-band helpers:
//!
//! - inspect a single blob with [`get_blob_info`]
//! - inspect only the first blob with [`inspect_first`]
//! - count concatenated blobs with [`get_band_count`]
//! - decode a single blob with [`decode`]
//! - decode only the first blob with [`decode_first`]
//! - decode concatenated band sets with [`decode_band_set`]
//! - decode promoted `f64` buffers with [`decode_to_f64`]
//! - decode only the first promoted `f64` blob with [`decode_first_to_f64`]
//! - decode directly into `ndarray::ArrayD` with [`decode_ndarray`]

mod bitstuff;
mod huffman;
mod io;
mod lerc1;
mod lerc2;
mod pixel;

#[cfg(test)]
mod tests;

use lerc_band_materialize::{
    copy_band_values_into_slice, BandLayout as MaterializeLayout, BandMaterializer,
};
use lerc_core::{
    BandLayout, BandSetInfo, BlobInfo, Decoded, DecodedBandSet, DecodedF64, Error, NdArrayElement,
    Result,
};
use ndarray::ArrayD;

pub fn inspect_first(blob: &[u8]) -> Result<BlobInfo> {
    if lerc1::is_lerc1(blob) {
        return lerc1::inspect(blob, None);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::inspect(blob, None);
    }
    Err(Error::InvalidMagic)
}

pub fn get_blob_info(blob: &[u8]) -> Result<BlobInfo> {
    let info = inspect_first(blob)?;
    ensure_single_blob_consumed(blob.len(), info.blob_size, "get_blob_info", "inspect_first")?;
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
            let (info, _) = lerc2::parse(slice)?;
            info.blob_size
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

pub fn decode(blob: &[u8]) -> Result<Decoded> {
    let decoded = decode_first(blob)?;
    ensure_single_blob_consumed(blob.len(), decoded.info.blob_size, "decode", "decode_first")?;
    Ok(decoded)
}

pub fn decode_band_set(blob: &[u8]) -> Result<DecodedBandSet> {
    let mut offset = 0usize;
    let mut bands = Vec::new();
    let mut infos = Vec::new();
    let mut band_masks = Vec::new();
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask: Option<Vec<u8>> = None;

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

pub fn decode_band_set_vec<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    decode_band_set_owned(blob, layout)
}

pub fn decode_band_set_into<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let band_info = scan_band_infos(blob)?;
    let band_count = band_info.band_count();
    let expected_len = band_info.value_count()?;
    if out.len() != expected_len {
        return Err(Error::InvalidBlob(format!(
            "output slice length {} does not match decoded band set length {}",
            out.len(),
            expected_len
        )));
    }

    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask: Option<Vec<u8>> = None;

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let decoded = decode_first_with_masks(slice, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;
        let pixel_count = decoded.info.pixel_count()?;
        let depth = decoded.info.depth as usize;

        let values = T::from_pixel_data(decoded.pixels)?;
        copy_band_values_into_slice(
            out,
            &values,
            pixel_count,
            depth,
            band_index,
            band_count,
            materialize_layout(layout),
        )
        .map_err(materialize_error)?;

        if is_lerc1 {
            lerc1_mask = decoded.mask;
            lerc2_mask = None;
        } else {
            lerc2_mask = decoded.mask;
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, decoded.info.blob_size, blob.len())?;
        band_index += 1;
    }

    Ok(band_info)
}

pub fn decode_band_set_ndarray<T: NdArrayElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

pub fn decode_band_set_ndarray_with_layout<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<ArrayD<T>> {
    let (info, values) = decode_band_set_owned(blob, layout)?;
    let shape = info.ndarray_shape_for_layout(layout);
    ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values).map_err(|e| {
        Error::InvalidBlob(format!(
            "failed to build ndarray from decoded band set: {e}"
        ))
    })
}

pub fn decode_band_set_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

pub fn decode_band_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    decode_band_set(blob)?.into_band_mask_ndarray()
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

pub fn decode_first_to_f64(blob: &[u8]) -> Result<DecodedF64> {
    decode_first_f64(blob)
}

pub fn decode_ndarray<T: NdArrayElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode(blob)?.into_ndarray()
}

pub fn decode_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_to_f64(blob)?.into_ndarray()
}

pub fn decode_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    decode(blob)?.into_mask_ndarray()
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

fn decode_first_f64(blob: &[u8]) -> Result<DecodedF64> {
    if lerc1::is_lerc1(blob) {
        return lerc1::decode_f64(blob, None);
    }
    if lerc2::is_lerc2(blob) {
        return lerc2::decode_f64(blob, None);
    }
    Err(Error::InvalidMagic)
}

fn decode_band_set_owned<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<(BandSetInfo, Vec<T>)> {
    let band_info = scan_band_infos(blob)?;
    let band_count = band_info.band_count();
    let expected_len = band_info.value_count()?;
    if expected_len == 0 {
        return Ok((band_info, Vec::new()));
    }

    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask: Option<Vec<u8>> = None;
    let mut materializer = if band_count > 1 {
        Some(
            BandMaterializer::new(
                band_info.bands[0].pixel_count()?,
                band_info.depth() as usize,
                band_count,
                materialize_layout(layout),
            )
            .map_err(materialize_error)?,
        )
    } else {
        None
    };

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let decoded = decode_first_with_masks(slice, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;
        let values = T::from_pixel_data(decoded.pixels)?;

        if band_count == 1 {
            return Ok((band_info, values));
        }

        materializer
            .as_mut()
            .unwrap()
            .copy_band(band_index, &values)
            .map_err(materialize_error)?;

        if is_lerc1 {
            lerc1_mask = decoded.mask;
            lerc2_mask = None;
        } else {
            lerc2_mask = decoded.mask;
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, decoded.info.blob_size, blob.len())?;
        band_index += 1;
    }

    Ok((
        band_info,
        materializer.unwrap().finish().map_err(materialize_error)?,
    ))
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
            let (info, _) = lerc2::parse(slice)?;
            (info, None)
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
