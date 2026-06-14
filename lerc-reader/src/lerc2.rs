use lerc_core::{BlobInfo, DataType, Error, MaskEncoding, PixelData, Result, Version};

use crate::allocation::{check_allocation, checked_mul, default_vec};
use crate::bitstuff::{data_type_used, decode_bits, read_typed_scalar};
use crate::huffman::{decode_huffman, decode_huffman_into};
use crate::io::Cursor;
use crate::pixel::{
    count_valid_in_block, fletcher32, output_value, read_typed_values, read_values_as,
    sample_index, Sample,
};
use crate::{Decoded, DecodedF64};
use lerc_band_materialize::{BandWriteOrder, BandWriter};

const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";

#[derive(Debug, Clone)]
pub(crate) struct DepthRanges {
    pub(crate) min_values: Vec<f64>,
    pub(crate) max_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedLerc2 {
    pub(crate) info: BlobInfo,
    pub(crate) mask_num_bytes: usize,
    raw_z_min: f64,
    raw_z_max: f64,
    encoded_no_data_value: Option<f64>,
}

pub(crate) fn is_lerc2(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC2)
}

pub(crate) fn inspect(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<BlobInfo> {
    let (parsed, mut cursor) = parse(blob)?;
    let mut info = parsed.info.clone();
    if parsed.info.uses_external_mask() {
        if let Some(mask) = inherited_mask {
            validate_external_mask(&parsed.info, mask)?;
        }
    } else {
        skip_mask(&mut cursor, &parsed)?;
    }

    read_public_depth_ranges(&mut cursor, &parsed, &mut info)?;
    Ok(info)
}

pub(crate) fn inspect_with_mask(
    blob: &[u8],
    inherited_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let (parsed, mut cursor) = parse(blob)?;
    let mut info = parsed.info.clone();
    let mask = read_mask(&mut cursor, &parsed, inherited_mask)?;
    read_public_depth_ranges(&mut cursor, &parsed, &mut info)?;
    Ok((info, mask))
}

pub(crate) fn decode(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<Decoded> {
    let (parsed, mut cursor) = parse(blob)?;
    let mut info = parsed.info.clone();
    let mask = read_mask(&mut cursor, &parsed, inherited_mask)?;
    let depth_ranges = read_public_depth_ranges(&mut cursor, &parsed, &mut info)?;
    let mut decode_info = info.clone();
    decode_info.z_min = parsed.raw_z_min;
    decode_info.z_max = parsed.raw_z_max;
    let mut pixels = decode_pixels(
        &mut cursor,
        &decode_info,
        depth_ranges.as_ref(),
        mask.as_deref(),
    )?;
    remap_pixel_data_no_data(
        &mut pixels,
        &info,
        mask.as_deref(),
        parsed.encoded_no_data_value,
    );
    Ok(Decoded { info, pixels, mask })
}

pub(crate) fn decode_f64(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<DecodedF64> {
    let (parsed, mut cursor) = parse(blob)?;
    let mut info = parsed.info.clone();
    let mask = read_mask(&mut cursor, &parsed, inherited_mask)?;
    let depth_ranges = read_public_depth_ranges(&mut cursor, &parsed, &mut info)?;
    let mut decode_info = info.clone();
    decode_info.z_min = parsed.raw_z_min;
    decode_info.z_max = parsed.raw_z_max;

    let mut pixels = match decode_pixels_typed::<f64>(
        &mut cursor,
        &decode_info,
        depth_ranges.as_ref(),
        mask.as_deref(),
    )? {
        PixelData::F64(values) => values,
        _ => unreachable!("f64 decode must produce an f64 buffer"),
    };
    remap_no_data_values(
        &mut pixels,
        &info,
        mask.as_deref(),
        parsed.encoded_no_data_value,
    );
    Ok(DecodedF64 { info, pixels, mask })
}

pub(crate) fn decode_into<T: Sample, W: BandWriter<T>>(
    blob: &[u8],
    inherited_mask: Option<&[u8]>,
    out: &mut W,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let (parsed, mut cursor) = parse(blob)?;
    let mut info = parsed.info.clone();
    let mask = read_mask(&mut cursor, &parsed, inherited_mask)?;
    let depth_ranges = read_public_depth_ranges(&mut cursor, &parsed, &mut info)?;
    let mut decode_info = info.clone();
    decode_info.z_min = parsed.raw_z_min;
    decode_info.z_max = parsed.raw_z_max;

    if info.valid_pixel_count != info.pixel_count()? as u32 {
        out.fill_default();
    }
    decode_pixels_into_typed(
        &mut cursor,
        &decode_info,
        depth_ranges.as_ref(),
        mask.as_deref(),
        out,
    )?;
    remap_written_no_data(out, &info, mask.as_deref(), parsed.encoded_no_data_value);
    Ok((info, mask))
}

pub(crate) fn parse(blob: &[u8]) -> Result<(ParsedLerc2, Cursor<'_>)> {
    let mut cursor = Cursor::new(blob);
    let magic = cursor.read_bytes(6)?;
    if magic != MAGIC_LERC2 {
        return Err(Error::InvalidMagic);
    }

    let version = cursor.read_i32()?;
    if version < 1 {
        return Err(Error::UnsupportedVersion(version as u32));
    }
    let version = version as u32;
    if version > 6 {
        return Err(Error::UnsupportedVersion(version));
    }

    let checksum = if version >= 3 {
        Some(cursor.read_u32()?)
    } else {
        None
    };

    let height = cursor.read_u32()?;
    let width = cursor.read_u32()?;
    let depth = if version >= 4 { cursor.read_u32()? } else { 1 };

    let valid_pixel_count = cursor.read_u32()?;
    let micro_block_size = cursor.read_i32()?;
    let blob_size = cursor.read_i32()?;
    let image_type = cursor.read_i32()?;
    let n_blobs_more = if version >= 6 { cursor.read_i32()? } else { 0 };
    let pass_no_data_values = if version >= 6 { cursor.read_u8()? } else { 0 };
    let _is_all_int = if version >= 6 { cursor.read_u8()? } else { 0 };
    let _reserved3 = if version >= 6 { cursor.read_u8()? } else { 0 };
    let _reserved4 = if version >= 6 { cursor.read_u8()? } else { 0 };
    let max_z_error = cursor.read_f64()?;
    let raw_z_min = cursor.read_f64()?;
    let raw_z_max = cursor.read_f64()?;
    let encoded_no_data_value = if version >= 6 {
        Some(cursor.read_f64()?)
    } else {
        None
    };
    let original_no_data_value = if version >= 6 {
        Some(cursor.read_f64()?)
    } else {
        None
    };

    if micro_block_size < 0 {
        return Err(Error::InvalidHeader("negative micro block size"));
    }
    if depth == 0 {
        return Err(Error::InvalidHeader("depth must be greater than zero"));
    }
    if n_blobs_more < 0 {
        return Err(Error::InvalidHeader("negative appended blob count"));
    }
    if version >= 6 && pass_no_data_values > 1 {
        return Err(Error::InvalidHeader("invalid no-data flag"));
    }
    if pass_no_data_values != 0 && depth <= 1 {
        return Err(Error::InvalidHeader(
            "no-data values require depth greater than one",
        ));
    }
    if blob_size <= 0 {
        return Err(Error::InvalidHeader("non-positive blob size"));
    }
    let blob_size = blob_size as usize;
    if blob_size > blob.len() {
        return Err(Error::Truncated {
            offset: 0,
            needed: blob_size,
            available: blob.len(),
        });
    }

    if let Some(expected) = checksum {
        if blob_size < 14 {
            return Err(Error::InvalidHeader(
                "blob size is smaller than checksum range",
            ));
        }
        let actual = fletcher32(&blob[14..blob_size]);
        if actual != expected {
            return Err(Error::ChecksumMismatch { expected, actual });
        }
    }

    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
    let num_valid = valid_pixel_count as usize;
    if num_valid > pixel_count {
        return Err(Error::InvalidBlob(format!(
            "valid pixel count {} exceeds pixel count {}",
            num_valid, pixel_count
        )));
    }

    let mask_num_bytes = cursor.read_u32()? as usize;
    let mask_encoding = infer_mask_encoding(valid_pixel_count, pixel_count, mask_num_bytes)?;

    let no_data_value = if pass_no_data_values != 0 {
        original_no_data_value
    } else {
        None
    };
    let encoded_no_data_value = if pass_no_data_values != 0 {
        encoded_no_data_value
    } else {
        None
    };
    let (z_min, z_max) = if no_data_value.is_some() {
        (-1.0, -1.0)
    } else {
        (raw_z_min, raw_z_max)
    };

    let info = BlobInfo {
        version: Version::Lerc2(version),
        data_type: DataType::from_code(image_type)?,
        width,
        height,
        depth,
        min_values: None,
        max_values: None,
        valid_pixel_count,
        micro_block_size: micro_block_size as u32,
        blob_size,
        max_z_error,
        z_min,
        z_max,
        mask_encoding,
        no_data_value,
    };

    let current_offset = cursor.offset();
    let mut limited_cursor = Cursor::new(&blob[..blob_size]);
    limited_cursor.skip(current_offset)?;

    Ok((
        ParsedLerc2 {
            info,
            mask_num_bytes,
            raw_z_min,
            raw_z_max,
            encoded_no_data_value,
        },
        limited_cursor,
    ))
}

fn infer_mask_encoding(
    valid_pixel_count: u32,
    pixel_count: usize,
    mask_num_bytes: usize,
) -> Result<MaskEncoding> {
    let num_valid = valid_pixel_count as usize;
    if num_valid == pixel_count {
        if mask_num_bytes != 0 {
            return Err(Error::InvalidBlob(
                "full-valid LERC blob unexpectedly contains mask bytes".into(),
            ));
        }
        return Ok(MaskEncoding::None);
    }

    if num_valid == 0 {
        return Ok(if mask_num_bytes == 0 {
            MaskEncoding::ImplicitAllInvalid
        } else {
            MaskEncoding::Explicit
        });
    }

    Ok(if mask_num_bytes == 0 {
        MaskEncoding::External
    } else {
        MaskEncoding::Explicit
    })
}

fn should_read_depth_ranges(parsed: &ParsedLerc2) -> bool {
    matches!(parsed.info.version, Version::Lerc2(version) if version >= 4)
        && parsed.info.valid_pixel_count != 0
        && parsed.raw_z_min != parsed.raw_z_max
}

fn read_public_depth_ranges(
    cursor: &mut Cursor<'_>,
    parsed: &ParsedLerc2,
    info: &mut BlobInfo,
) -> Result<Option<DepthRanges>> {
    if !should_read_depth_ranges(parsed) {
        return Ok(None);
    }

    let ranges = read_depth_ranges(cursor, &parsed.info)?;
    if !info.uses_no_data_value() {
        info.min_values = Some(ranges.min_values.clone());
        info.max_values = Some(ranges.max_values.clone());
    }
    Ok(Some(ranges))
}

fn skip_mask(cursor: &mut Cursor<'_>, parsed: &ParsedLerc2) -> Result<()> {
    cursor.skip(parsed.mask_num_bytes)
}

fn validate_external_mask(info: &BlobInfo, mask: &[u8]) -> Result<()> {
    let num_pixels = info.pixel_count()?;
    if mask.len() != num_pixels {
        return Err(Error::InvalidBlob(
            "inherited mask length does not match the current LERC blob".into(),
        ));
    }

    let inherited_valid = mask.iter().filter(|&&value| value != 0).count();
    if inherited_valid != info.valid_pixel_count as usize {
        return Err(Error::InvalidBlob(
            "inherited mask valid count does not match the current LERC blob".into(),
        ));
    }

    Ok(())
}

fn read_mask(
    cursor: &mut Cursor<'_>,
    parsed: &ParsedLerc2,
    inherited_mask: Option<&[u8]>,
) -> Result<Option<Vec<u8>>> {
    let info = &parsed.info;
    let num_pixels = info.pixel_count()?;
    match parsed.info.mask_encoding {
        MaskEncoding::None => Ok(None),
        MaskEncoding::ImplicitAllInvalid => {
            cursor.skip(parsed.mask_num_bytes)?;
            Ok(Some(default_vec(num_pixels, "Lerc2 implicit mask")?))
        }
        MaskEncoding::External => {
            let mask = inherited_mask.ok_or(Error::UnsupportedFeature(
                "Lerc2 external masks require a caller-supplied mask via the *_with_mask APIs",
            ))?;
            validate_external_mask(info, mask)?;
            Ok(Some(mask.to_vec()))
        }
        MaskEncoding::Explicit => {
            let bitset = decode_mask_rle(
                cursor.read_bytes(parsed.mask_num_bytes)?,
                num_pixels.div_ceil(8),
            )?;
            Ok(Some(unpack_mask_bitset(&bitset, num_pixels)?))
        }
    }
}

pub(crate) fn decode_mask_rle(encoded: &[u8], bitset_len: usize) -> Result<Vec<u8>> {
    let mut bitset = default_vec(bitset_len, "Lerc2 mask bitset")?;
    let mut inner = Cursor::new(encoded);
    let mut out = 0usize;

    loop {
        let count = inner.read_i16()?;
        if count == i16::MIN {
            break;
        }
        if count > 0 {
            let count = count as usize;
            let run = inner.read_bytes(count)?;
            if out
                .checked_add(count)
                .ok_or_else(|| Error::InvalidBlob("mask RLE overflow".into()))?
                > bitset.len()
            {
                return Err(Error::InvalidBlob(
                    "mask RLE expands past bitset length".into(),
                ));
            }
            bitset[out..out + count].copy_from_slice(run);
            out += count;
        } else {
            let count = (-count) as usize;
            let value = inner.read_u8()?;
            if out
                .checked_add(count)
                .ok_or_else(|| Error::InvalidBlob("mask RLE overflow".into()))?
                > bitset.len()
            {
                return Err(Error::InvalidBlob(
                    "mask RLE expands past bitset length".into(),
                ));
            }
            bitset[out..out + count].fill(value);
            out += count;
        }
    }

    if out != bitset.len() {
        return Err(Error::InvalidBlob(
            "mask RLE ended before filling bitset".into(),
        ));
    }
    if inner.offset() != encoded.len() {
        return Err(Error::InvalidBlob(
            "mask RLE contains trailing bytes after sentinel".into(),
        ));
    }

    Ok(bitset)
}

pub(crate) fn unpack_mask_bitset(bitset: &[u8], num_pixels: usize) -> Result<Vec<u8>> {
    let mut mask = default_vec(num_pixels, "Lerc2 unpacked mask")?;
    for (i, item) in mask.iter_mut().enumerate() {
        let byte = bitset[i >> 3];
        let bit = 7 - (i & 7);
        *item = (byte >> bit) & 1;
    }
    Ok(mask)
}

fn read_depth_ranges(cursor: &mut Cursor<'_>, info: &BlobInfo) -> Result<DepthRanges> {
    let depth = info.depth as usize;
    let range_bytes = depth
        .checked_mul(info.data_type.byte_len())
        .ok_or_else(|| Error::InvalidBlob("range byte length overflows usize".into()))?;
    let min_values = read_typed_values(cursor.read_bytes(range_bytes)?, info.data_type)?;
    let max_values = read_typed_values(cursor.read_bytes(range_bytes)?, info.data_type)?;
    Ok(DepthRanges {
        min_values,
        max_values,
    })
}

fn typed_no_data_mapping<T: Sample>(
    info: &BlobInfo,
    encoded_no_data_value: Option<f64>,
) -> Option<(T, T)> {
    let original = info.no_data_value?;
    let encoded = encoded_no_data_value?;
    let encoded = output_value::<T>(encoded, info.data_type);
    let original = output_value::<T>(original, info.data_type);
    if encoded.to_f64() == original.to_f64() {
        None
    } else {
        Some((encoded, original))
    }
}

fn remap_no_data_values<T: Sample>(
    values: &mut [T],
    info: &BlobInfo,
    mask: Option<&[u8]>,
    encoded_no_data_value: Option<f64>,
) {
    let Some((encoded, original)) = typed_no_data_mapping::<T>(info, encoded_no_data_value) else {
        return;
    };

    let depth = info.depth as usize;
    match mask {
        Some(mask) => {
            for (pixel, &is_valid) in mask.iter().enumerate() {
                if is_valid == 0 {
                    continue;
                }
                for dim in 0..depth {
                    let index = sample_index(pixel, depth, dim);
                    if values[index].to_f64() == encoded.to_f64() {
                        values[index] = original;
                    }
                }
            }
        }
        None => {
            for value in values {
                if value.to_f64() == encoded.to_f64() {
                    *value = original;
                }
            }
        }
    }
}

fn remap_pixel_data_no_data(
    pixels: &mut PixelData,
    info: &BlobInfo,
    mask: Option<&[u8]>,
    encoded_no_data_value: Option<f64>,
) {
    match pixels {
        PixelData::I8(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::U8(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::I16(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::U16(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::I32(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::U32(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::F32(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
        PixelData::F64(values) => remap_no_data_values(values, info, mask, encoded_no_data_value),
    }
}

fn remap_written_no_data<T: Sample, W: BandWriter<T>>(
    out: &mut W,
    info: &BlobInfo,
    mask: Option<&[u8]>,
    encoded_no_data_value: Option<f64>,
) {
    let Some((encoded, original)) = typed_no_data_mapping::<T>(info, encoded_no_data_value) else {
        return;
    };

    let depth = info.depth as usize;
    let num_pixels = info.width as usize * info.height as usize;
    match mask {
        Some(mask) => {
            for (pixel, &is_valid) in mask.iter().enumerate() {
                if is_valid == 0 {
                    continue;
                }
                for dim in 0..depth {
                    if out.read(pixel, dim).to_f64() == encoded.to_f64() {
                        out.write(pixel, dim, original);
                    }
                }
            }
        }
        None => {
            for pixel in 0..num_pixels {
                for dim in 0..depth {
                    if out.read(pixel, dim).to_f64() == encoded.to_f64() {
                        out.write(pixel, dim, original);
                    }
                }
            }
        }
    }
}

fn decode_pixels(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
) -> Result<PixelData> {
    match info.data_type {
        DataType::I8 => decode_pixels_typed::<i8>(cursor, info, depth_ranges, mask),
        DataType::U8 => decode_pixels_typed::<u8>(cursor, info, depth_ranges, mask),
        DataType::I16 => decode_pixels_typed::<i16>(cursor, info, depth_ranges, mask),
        DataType::U16 => decode_pixels_typed::<u16>(cursor, info, depth_ranges, mask),
        DataType::I32 => decode_pixels_typed::<i32>(cursor, info, depth_ranges, mask),
        DataType::U32 => decode_pixels_typed::<u32>(cursor, info, depth_ranges, mask),
        DataType::F32 => decode_pixels_typed::<f32>(cursor, info, depth_ranges, mask),
        DataType::F64 => decode_pixels_typed::<f64>(cursor, info, depth_ranges, mask),
    }
}

fn decode_pixels_typed<T: Sample>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
) -> Result<PixelData> {
    let num_pixels = info.pixel_count()?;
    let sample_count = info.sample_count()?;

    if info.valid_pixel_count == 0 {
        return Ok(T::into_pixel_data(default_vec(
            sample_count,
            "Lerc2 empty pixel buffer",
        )?));
    }

    if info.z_min == info.z_max {
        return Ok(T::into_pixel_data(constant_pixels::<T>(
            info.data_type,
            num_pixels,
            info.depth as usize,
            mask,
            ConstantValues::Single(info.z_max),
        )?));
    }

    if let Some(ranges) = depth_ranges {
        if ranges.min_values == ranges.max_values {
            return Ok(T::into_pixel_data(constant_pixels::<T>(
                info.data_type,
                num_pixels,
                info.depth as usize,
                mask,
                ConstantValues::PerDepth(&ranges.max_values),
            )?));
        }
    }

    let one_sweep = cursor.read_u8()? != 0;
    if one_sweep {
        return decode_one_sweep::<T>(cursor, info, mask);
    }

    let version = match info.version {
        Version::Lerc2(version) => version,
        _ => unreachable!("Lerc2 decode called with a non-Lerc2 blob"),
    };

    if needs_encode_mode_flag(info, version) {
        let encode_mode = read_encode_mode(cursor, version)?;
        if encode_mode != 0 {
            if !supports_integer_huffman(info) {
                return Err(Error::UnsupportedFeature(
                    "Lerc2 floating-point Huffman decode",
                ));
            }
            return decode_huffman::<T>(cursor, info, mask, encode_mode == 1);
        }
    }

    decode_tiles::<T>(cursor, info, depth_ranges, mask)
}

fn decode_pixels_into_typed<T: Sample, W: BandWriter<T>>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
    out: &mut W,
) -> Result<()> {
    if info.valid_pixel_count == 0 {
        return Ok(());
    }

    if info.z_min == info.z_max {
        write_constant_pixels(
            out,
            info.data_type,
            info.pixel_count()?,
            info.depth as usize,
            mask,
            ConstantValues::Single(info.z_max),
        );
        return Ok(());
    }

    if let Some(ranges) = depth_ranges {
        if ranges.min_values == ranges.max_values {
            write_constant_pixels(
                out,
                info.data_type,
                info.pixel_count()?,
                info.depth as usize,
                mask,
                ConstantValues::PerDepth(&ranges.max_values),
            );
            return Ok(());
        }
    }

    let one_sweep = cursor.read_u8()? != 0;
    if one_sweep {
        return decode_one_sweep_into(cursor, info, mask, out);
    }

    let version = match info.version {
        Version::Lerc2(version) => version,
        _ => unreachable!("Lerc2 decode called with a non-Lerc2 blob"),
    };

    if needs_encode_mode_flag(info, version) {
        let encode_mode = read_encode_mode(cursor, version)?;
        if encode_mode != 0 {
            if !supports_integer_huffman(info) {
                return Err(Error::UnsupportedFeature(
                    "Lerc2 floating-point Huffman decode",
                ));
            }
            return decode_huffman_into(cursor, info, mask, encode_mode == 1, out);
        }
    }

    decode_tiles_into(cursor, info, depth_ranges, mask, out)
}

enum ConstantValues<'a> {
    Single(f64),
    PerDepth(&'a [f64]),
}

fn needs_encode_mode_flag(info: &BlobInfo, version: u32) -> bool {
    supports_integer_huffman(info)
        || (version >= 6
            && matches!(info.data_type, DataType::F32 | DataType::F64)
            && info.max_z_error == 0.0)
}

fn supports_integer_huffman(info: &BlobInfo) -> bool {
    matches!(info.data_type, DataType::I8 | DataType::U8) && (info.max_z_error - 0.5).abs() < 1e-5
}

fn read_encode_mode(cursor: &mut Cursor<'_>, version: u32) -> Result<u8> {
    let encode_mode = cursor.read_u8()?;
    if encode_mode > 3 || (version < 6 && encode_mode > 2) || (version < 4 && encode_mode > 1) {
        return Err(Error::InvalidBlob(format!(
            "invalid Lerc2 encode mode flag {encode_mode}"
        )));
    }
    Ok(encode_mode)
}

fn constant_pixels<T: Sample>(
    data_type: DataType,
    num_pixels: usize,
    depth: usize,
    mask: Option<&[u8]>,
    values: ConstantValues<'_>,
) -> Result<Vec<T>> {
    let sample_count = checked_mul(num_pixels, depth, "Lerc2 sample count")?;
    let mut out = default_vec(sample_count, "Lerc2 constant pixel buffer")?;
    match values {
        ConstantValues::Single(value) => {
            let value = output_value::<T>(value, data_type);
            if let Some(mask) = mask {
                for (pixel, &mask_value) in mask.iter().enumerate().take(num_pixels) {
                    if mask_value != 0 {
                        let base = pixel * depth;
                        out[base..base + depth].fill(value);
                    }
                }
            } else {
                out.fill(value);
            }
        }
        ConstantValues::PerDepth(values) => {
            if let Some(mask) = mask {
                for (pixel, &mask_value) in mask.iter().enumerate().take(num_pixels) {
                    if mask_value != 0 {
                        let base = pixel * depth;
                        for (offset, &value) in values.iter().enumerate() {
                            out[base + offset] = output_value::<T>(value, data_type);
                        }
                    }
                }
            } else {
                for pixel in 0..num_pixels {
                    let base = pixel * depth;
                    for (offset, &value) in values.iter().enumerate() {
                        out[base + offset] = output_value::<T>(value, data_type);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn write_constant_pixels<T: Sample, W: BandWriter<T>>(
    out: &mut W,
    data_type: DataType,
    num_pixels: usize,
    depth: usize,
    mask: Option<&[u8]>,
    values: ConstantValues<'_>,
) {
    match values {
        ConstantValues::Single(value) => {
            let value = output_value::<T>(value, data_type);
            if let Some(mask) = mask {
                for (pixel, &mask_value) in mask.iter().enumerate().take(num_pixels) {
                    if mask_value != 0 {
                        for dim in 0..depth {
                            out.write(pixel, dim, value);
                        }
                    }
                }
            } else {
                for pixel in 0..num_pixels {
                    for dim in 0..depth {
                        out.write(pixel, dim, value);
                    }
                }
            }
        }
        ConstantValues::PerDepth(values) => {
            if let Some(mask) = mask {
                for (pixel, &mask_value) in mask.iter().enumerate().take(num_pixels) {
                    if mask_value != 0 {
                        for (dim, &value) in values.iter().enumerate() {
                            out.write(pixel, dim, output_value::<T>(value, data_type));
                        }
                    }
                }
            } else {
                for pixel in 0..num_pixels {
                    for (dim, &value) in values.iter().enumerate() {
                        out.write(pixel, dim, output_value::<T>(value, data_type));
                    }
                }
            }
        }
    }
}

fn decode_one_sweep<T: Sample>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    mask: Option<&[u8]>,
) -> Result<PixelData> {
    let num_pixels = info.pixel_count()?;
    let num_valid = info.valid_pixel_count as usize;
    let depth = info.depth as usize;
    let sample_len = info
        .data_type
        .byte_len()
        .checked_mul(num_valid)
        .and_then(|v| v.checked_mul(depth))
        .ok_or_else(|| Error::InvalidBlob("one-sweep byte count overflows usize".into()))?;
    check_allocation::<T>(
        checked_mul(num_valid, depth, "Lerc2 one-sweep sample count")?,
        "Lerc2 one-sweep raw values",
    )?;
    let raw = read_values_as::<T>(cursor.read_bytes(sample_len)?, info.data_type)?;

    if num_valid == num_pixels {
        return Ok(T::into_pixel_data(raw));
    }

    let mask = mask.ok_or(Error::InvalidBlob(
        "partial-valid one-sweep block is missing its decoded mask".into(),
    ))?;
    let mut out = default_vec(info.sample_count()?, "Lerc2 one-sweep output")?;
    let mut src = 0usize;
    for (pixel, &is_valid) in mask.iter().enumerate() {
        if is_valid == 0 {
            continue;
        }
        let base = pixel * depth;
        let src_end = src + depth;
        if src_end > raw.len() {
            return Err(Error::InvalidBlob(
                "one-sweep valid sample payload is too short".into(),
            ));
        }
        out[base..base + depth].copy_from_slice(&raw[src..src_end]);
        src = src_end;
    }
    if src != raw.len() {
        return Err(Error::InvalidBlob(
            "one-sweep valid sample payload contains trailing values".into(),
        ));
    }
    Ok(T::into_pixel_data(out))
}

fn decode_one_sweep_into<T: Sample, W: BandWriter<T>>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    mask: Option<&[u8]>,
    out: &mut W,
) -> Result<()> {
    let num_pixels = info.pixel_count()?;
    let num_valid = info.valid_pixel_count as usize;
    let depth = info.depth as usize;
    let sample_len = info
        .data_type
        .byte_len()
        .checked_mul(num_valid)
        .and_then(|v| v.checked_mul(depth))
        .ok_or_else(|| Error::InvalidBlob("one-sweep byte count overflows usize".into()))?;
    check_allocation::<T>(
        checked_mul(num_valid, depth, "Lerc2 one-sweep sample count")?,
        "Lerc2 one-sweep raw values",
    )?;
    let raw = read_values_as::<T>(cursor.read_bytes(sample_len)?, info.data_type)?;

    if num_valid == num_pixels {
        let mut src = 0usize;
        for pixel in 0..num_pixels {
            for dim in 0..depth {
                out.write(pixel, dim, raw[src]);
                src += 1;
            }
        }
        return Ok(());
    }

    let mask = mask.ok_or(Error::InvalidBlob(
        "partial-valid one-sweep block is missing its decoded mask".into(),
    ))?;
    let mut src = 0usize;
    for (pixel, &is_valid) in mask.iter().enumerate() {
        if is_valid == 0 {
            continue;
        }
        for dim in 0..depth {
            if src >= raw.len() {
                return Err(Error::InvalidBlob(
                    "one-sweep valid sample payload is too short".into(),
                ));
            }
            out.write(pixel, dim, raw[src]);
            src += 1;
        }
    }
    if src != raw.len() {
        return Err(Error::InvalidBlob(
            "one-sweep valid sample payload contains trailing values".into(),
        ));
    }
    Ok(())
}

struct PixelDataWriter<'a, T> {
    values: &'a mut [T],
    depth: usize,
}

impl<T: Copy> PixelDataWriter<'_, T> {
    fn index(&self, pixel: usize, dim: usize) -> usize {
        pixel * self.depth + dim
    }
}

impl<T: Sample> BandWriter<T> for PixelDataWriter<'_, T> {
    fn fill_default(&mut self) {
        self.values.fill(T::default());
    }

    fn write(&mut self, pixel: usize, dim: usize, value: T) {
        let index = self.index(pixel, dim);
        self.values[index] = value;
    }

    fn read(&self, pixel: usize, dim: usize) -> T {
        self.values[self.index(pixel, dim)]
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

fn decode_tiles<T: Sample>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
) -> Result<PixelData> {
    if info.micro_block_size == 0 {
        return Err(Error::InvalidBlob(
            "micro block size must be greater than zero".into(),
        ));
    }
    let mut result = default_vec(info.sample_count()?, "Lerc2 tile output")?;
    let depth = info.depth as usize;
    let mut writer = PixelDataWriter {
        values: &mut result,
        depth,
    };
    decode_tiles_into(cursor, info, depth_ranges, mask, &mut writer)?;
    Ok(T::into_pixel_data(result))
}

fn decode_tiles_into<T: Sample, W: BandWriter<T>>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
    out: &mut W,
) -> Result<()> {
    out.set_write_order(BandWriteOrder::Arbitrary);
    let width = info.width as usize;
    let height = info.height as usize;
    let depth = info.depth as usize;
    let micro_block_size = info.micro_block_size as usize;
    if micro_block_size == 0 {
        return Err(Error::InvalidBlob(
            "micro block size must be greater than zero".into(),
        ));
    }

    let num_blocks_x = width.div_ceil(micro_block_size);
    let num_blocks_y = height.div_ceil(micro_block_size);
    let last_block_width = if width % micro_block_size == 0 {
        micro_block_size
    } else {
        width % micro_block_size
    };
    let last_block_height = if height % micro_block_size == 0 {
        micro_block_size
    } else {
        height % micro_block_size
    };

    let max_block_width = width.min(micro_block_size);
    let max_block_height = height.min(micro_block_size);
    let block_samples = checked_mul(
        max_block_width,
        max_block_height,
        "Lerc2 micro-block sample count",
    )?;
    let mut block_buffer = default_vec(block_samples, "Lerc2 micro-block buffer")?;
    let version = match info.version {
        Version::Lerc2(version) => version,
        _ => unreachable!("Lerc2 tile decode called with a non-Lerc2 blob"),
    };
    let file_version_check_num = if version >= 5 { 14u8 } else { 15u8 };

    for block_y in 0..num_blocks_y {
        let this_block_height = if block_y + 1 == num_blocks_y {
            last_block_height
        } else {
            micro_block_size
        };
        for block_x in 0..num_blocks_x {
            let this_block_width = if block_x + 1 == num_blocks_x {
                last_block_width
            } else {
                micro_block_size
            };

            for dim in 0..depth {
                let header_byte = cursor.read_u8()?;
                let is_diff_encoding = version >= 5 && (header_byte & 4) != 0;
                let bits67 = header_byte >> 6;
                let test_code = (header_byte >> 2) & file_version_check_num;
                let expected_code =
                    (((block_x * micro_block_size) >> 3) as u8) & file_version_check_num;
                if test_code != expected_code {
                    return Err(Error::InvalidBlob(
                        "tile block integrity check failed".into(),
                    ));
                }
                if is_diff_encoding && dim == 0 {
                    return Err(Error::InvalidBlob(
                        "diff-encoded tile block encountered in first dimension".into(),
                    ));
                }

                let block_encoding = header_byte & 3;
                let z_max = depth_ranges
                    .and_then(|ranges| ranges.max_values.get(dim))
                    .copied()
                    .unwrap_or(info.z_max);

                match block_encoding {
                    2 => {
                        for row in 0..this_block_height {
                            let pixel_row = block_y * micro_block_size + row;
                            for col in 0..this_block_width {
                                let pixel = pixel_row * width + block_x * micro_block_size + col;
                                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                    let value = if is_diff_encoding {
                                        out.read(pixel, dim - 1)
                                    } else {
                                        T::default()
                                    };
                                    out.write(pixel, dim, value);
                                }
                            }
                        }
                    }
                    0 => {
                        if is_diff_encoding {
                            return Err(Error::InvalidBlob(
                                "uncompressed diff-encoded tile block is invalid".into(),
                            ));
                        }
                        let values_needed = if let Some(mask) = mask {
                            count_valid_in_block(
                                mask,
                                width,
                                block_x * micro_block_size,
                                block_y * micro_block_size,
                                this_block_width,
                                this_block_height,
                            )
                        } else {
                            this_block_width * this_block_height
                        };
                        let byte_len = values_needed
                            .checked_mul(info.data_type.byte_len())
                            .ok_or_else(|| {
                                Error::InvalidBlob(
                                    "uncompressed block byte length overflows usize".into(),
                                )
                            })?;
                        let raw_values =
                            read_values_as::<T>(cursor.read_bytes(byte_len)?, info.data_type)?;
                        let mut src = 0usize;
                        for row in 0..this_block_height {
                            let pixel_row = block_y * micro_block_size + row;
                            for col in 0..this_block_width {
                                let pixel = pixel_row * width + block_x * micro_block_size + col;
                                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                    out.write(pixel, dim, raw_values[src]);
                                    src += 1;
                                }
                            }
                        }
                        if src != raw_values.len() {
                            return Err(Error::InvalidBlob(
                                "uncompressed tile block payload contains trailing values".into(),
                            ));
                        }
                    }
                    1 | 3 => {
                        let offset_type = data_type_used(
                            if is_diff_encoding && info.data_type.code() < 6 {
                                DataType::I32
                            } else {
                                info.data_type
                            },
                            bits67,
                        )?;
                        let offset = read_typed_scalar(cursor, offset_type)?;

                        if block_encoding == 3 {
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        let value = if is_diff_encoding {
                                            (out.read(pixel, dim - 1).to_f64() + offset).min(z_max)
                                        } else {
                                            offset
                                        };
                                        out.write(
                                            pixel,
                                            dim,
                                            output_value::<T>(value, info.data_type),
                                        );
                                    }
                                }
                            }
                        } else {
                            block_buffer.fill(0.0);
                            let block_values = decode_bits(
                                cursor,
                                version,
                                &mut block_buffer,
                                Some(offset),
                                info.max_z_error,
                                z_max,
                            )?;
                            let mut src = 0usize;
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        let value = if is_diff_encoding {
                                            block_buffer[src] + out.read(pixel, dim - 1).to_f64()
                                        } else {
                                            block_buffer[src]
                                        };
                                        out.write(
                                            pixel,
                                            dim,
                                            output_value::<T>(value, info.data_type),
                                        );
                                        src += 1;
                                    }
                                }
                            }
                            if src != block_values {
                                return Err(Error::InvalidBlob(
                                    "bit-stuffed tile block value count does not match the block mask".into(),
                                ));
                            }
                        }
                    }
                    _ => {
                        return Err(Error::InvalidBlob(format!(
                            "invalid tile block encoding {block_encoding}"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}
