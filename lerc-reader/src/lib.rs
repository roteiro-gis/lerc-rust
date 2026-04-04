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
//! - decode directly into `ndarray::ArrayD` with [`decode_ndarray`]

mod bitstuff;
mod huffman;
mod io;
mod lerc1;
mod lerc2;
mod pixel;

#[cfg(test)]
mod tests;

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
    let mut lerc2_mask: Option<Vec<u8>> = None;

    while offset < blob.len() {
        let slice = &blob[offset..];
        let next_len = if lerc1::is_lerc1(slice) {
            let parsed = lerc1::parse(slice, lerc1_mask.as_deref())?;
            lerc1_mask = parsed.mask.clone();
            lerc2_mask = None;
            parsed.eof_offset
        } else if lerc2::is_lerc2(slice) {
            let decoded = lerc2::decode(slice, lerc2_mask.as_deref())?;
            lerc2_mask = decoded.mask;
            lerc1_mask = None;
            decoded.info.blob_size
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
    let decoded = decode_band_set(blob)?;
    let info = decoded.info.clone();
    let values = decoded.into_vec_with_layout(layout)?;
    Ok((info, values))
}

pub fn decode_band_set_into<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
    out: &mut [T],
) -> Result<BandSetInfo> {
    let band_count = get_band_count(blob)?;
    let mut offset = 0usize;
    let mut band_index = 0usize;
    let mut infos = Vec::with_capacity(band_count);
    let mut lerc1_mask: Option<Vec<u8>> = None;
    let mut lerc2_mask: Option<Vec<u8>> = None;

    while offset < blob.len() {
        let slice = &blob[offset..];
        let is_lerc1 = lerc1::is_lerc1(slice);
        let decoded = decode_first_with_masks(slice, lerc1_mask.as_deref(), lerc2_mask.as_deref())?;
        let pixel_count = decoded.info.pixel_count()?;
        let depth = decoded.info.depth as usize;
        let expected_len = band_set_value_len(&decoded.info, band_count)?;
        if out.len() != expected_len {
            return Err(Error::InvalidBlob(format!(
                "output slice length {} does not match decoded band set length {}",
                out.len(),
                expected_len
            )));
        }

        let values = T::from_pixel_data(decoded.pixels)?;
        write_band_into_slice(
            out,
            &values,
            pixel_count,
            depth,
            band_index,
            band_count,
            layout,
        )?;

        if is_lerc1 {
            lerc1_mask = decoded.mask;
            lerc2_mask = None;
        } else {
            lerc2_mask = decoded.mask;
            lerc1_mask = None;
        }

        offset = checked_next_offset(offset, decoded.info.blob_size, blob.len())?;
        infos.push(decoded.info);
        band_index += 1;
    }

    BandSetInfo::new(infos)
}

pub fn decode_band_set_ndarray<T: NdArrayElement>(blob: &[u8]) -> Result<ArrayD<T>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

pub fn decode_band_set_ndarray_with_layout<T: NdArrayElement>(
    blob: &[u8],
    layout: BandLayout,
) -> Result<ArrayD<T>> {
    decode_band_set(blob)?.into_ndarray_with_layout(layout)
}

pub fn decode_band_set_ndarray_f64(blob: &[u8]) -> Result<ArrayD<f64>> {
    decode_band_set_ndarray_with_layout(blob, BandLayout::Interleaved)
}

pub fn decode_band_mask_ndarray(blob: &[u8]) -> Result<Option<ArrayD<u8>>> {
    decode_band_set(blob)?.into_band_mask_ndarray()
}

pub fn decode_to_f64(blob: &[u8]) -> Result<DecodedF64> {
    let decoded = decode(blob)?;
    Ok(DecodedF64 {
        info: decoded.info,
        pixels: decoded.pixels.to_f64(),
        mask: decoded.mask,
    })
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

fn band_set_value_len(info: &BlobInfo, band_count: usize) -> Result<usize> {
    info.pixel_count()?
        .checked_mul(info.depth as usize)
        .and_then(|value_count| value_count.checked_mul(band_count))
        .ok_or_else(|| Error::InvalidBlob("decoded band set length overflows usize".into()))
}

fn write_band_into_slice<T: NdArrayElement>(
    out: &mut [T],
    values: &[T],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
) -> Result<()> {
    let band_len = pixel_count
        .checked_mul(depth)
        .ok_or_else(|| Error::InvalidBlob("decoded band length overflows usize".into()))?;
    if values.len() != band_len {
        return Err(Error::InvalidBlob(
            "decoded band length does not match its metadata".into(),
        ));
    }

    match layout {
        BandLayout::Interleaved => {
            if depth <= 1 {
                for pixel in 0..pixel_count {
                    out[pixel * band_count + band_index] = values[pixel].clone();
                }
            } else {
                for pixel in 0..pixel_count {
                    let src_base = pixel * depth;
                    let dst_base = (pixel * band_count + band_index) * depth;
                    out[dst_base..dst_base + depth]
                        .clone_from_slice(&values[src_base..src_base + depth]);
                }
            }
        }
        BandLayout::Bsq => {
            let dst_base = band_index * band_len;
            out[dst_base..dst_base + band_len].clone_from_slice(values);
        }
    }

    Ok(())
}
