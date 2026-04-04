use lerc_core::{BlobInfo, DataType, Decoded, DecodedF64, Error, PixelData, Result, Version};

use crate::bitstuff::{data_type_used, decode_bits, read_typed_scalar};
use crate::huffman::{decode_huffman, decode_huffman_into};
use crate::io::Cursor;
use crate::pixel::{
    count_valid_in_block, fletcher32, output_value, read_typed_values, read_values_as,
    sample_index, Sample,
};
use lerc_band_materialize::BandWriter;

const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";

#[derive(Debug, Clone)]
pub(crate) struct DepthRanges {
    pub(crate) min_values: Vec<f64>,
    pub(crate) max_values: Vec<f64>,
}

pub(crate) fn is_lerc2(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC2)
}

pub(crate) fn inspect(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<BlobInfo> {
    let (info, _) = inspect_with_mask(blob, inherited_mask)?;
    Ok(info)
}

pub(crate) fn inspect_with_mask(
    blob: &[u8],
    inherited_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let (mut info, mut cursor) = parse(blob)?;
    let mask = read_mask(&mut cursor, &info, inherited_mask)?;
    if should_read_depth_ranges(&info) {
        let ranges = read_depth_ranges(&mut cursor, &info)?;
        info.min_values = Some(ranges.min_values);
        info.max_values = Some(ranges.max_values);
    }
    Ok((info, mask))
}

pub(crate) fn decode(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<Decoded> {
    let (mut info, mut cursor) = parse(blob)?;
    let mask = read_mask(&mut cursor, &info, inherited_mask)?;

    let depth_ranges = if should_read_depth_ranges(&info) {
        let ranges = read_depth_ranges(&mut cursor, &info)?;
        info.min_values = Some(ranges.min_values.clone());
        info.max_values = Some(ranges.max_values.clone());
        Some(ranges)
    } else {
        None
    };

    let pixels = decode_pixels(&mut cursor, &info, depth_ranges.as_ref(), mask.as_deref())?;
    Ok(Decoded { info, pixels, mask })
}

pub(crate) fn decode_f64(blob: &[u8], inherited_mask: Option<&[u8]>) -> Result<DecodedF64> {
    let (mut info, mut cursor) = parse(blob)?;
    let mask = read_mask(&mut cursor, &info, inherited_mask)?;

    let depth_ranges = if should_read_depth_ranges(&info) {
        let ranges = read_depth_ranges(&mut cursor, &info)?;
        info.min_values = Some(ranges.min_values.clone());
        info.max_values = Some(ranges.max_values.clone());
        Some(ranges)
    } else {
        None
    };

    let pixels = match decode_pixels_typed::<f64>(
        &mut cursor,
        &info,
        depth_ranges.as_ref(),
        mask.as_deref(),
    )? {
        PixelData::F64(values) => values,
        _ => unreachable!("f64 decode must produce an f64 buffer"),
    };
    Ok(DecodedF64 { info, pixels, mask })
}

pub(crate) fn decode_into<T: Sample, W: BandWriter<T>>(
    blob: &[u8],
    inherited_mask: Option<&[u8]>,
    out: &mut W,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let (mut info, mut cursor) = parse(blob)?;
    let mask = read_mask(&mut cursor, &info, inherited_mask)?;

    let depth_ranges = if should_read_depth_ranges(&info) {
        let ranges = read_depth_ranges(&mut cursor, &info)?;
        info.min_values = Some(ranges.min_values.clone());
        info.max_values = Some(ranges.max_values.clone());
        Some(ranges)
    } else {
        None
    };

    if info.valid_pixel_count != info.pixel_count()? as u32 {
        out.fill_default();
    }
    decode_pixels_into_typed(
        &mut cursor,
        &info,
        depth_ranges.as_ref(),
        mask.as_deref(),
        out,
    )?;
    Ok((info, mask))
}

pub(crate) fn parse(blob: &[u8]) -> Result<(BlobInfo, Cursor<'_>)> {
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
    let max_z_error = cursor.read_f64()?;
    let z_min = cursor.read_f64()?;
    let z_max = cursor.read_f64()?;

    if micro_block_size < 0 {
        return Err(Error::InvalidHeader("negative micro block size"));
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
        let actual = fletcher32(&blob[14..blob_size]);
        if actual != expected {
            return Err(Error::ChecksumMismatch { expected, actual });
        }
    }

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
    };

    Ok((info, cursor))
}

fn should_read_depth_ranges(info: &BlobInfo) -> bool {
    matches!(info.version, Version::Lerc2(version) if version >= 4)
        && info.valid_pixel_count != 0
        && info.z_min != info.z_max
}

fn read_mask(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    inherited_mask: Option<&[u8]>,
) -> Result<Option<Vec<u8>>> {
    let num_pixels = info.pixel_count()?;
    let num_valid = info.valid_pixel_count as usize;
    if num_valid > num_pixels {
        return Err(Error::InvalidBlob(format!(
            "valid pixel count {} exceeds pixel count {}",
            num_valid, num_pixels
        )));
    }

    let num_bytes = cursor.read_u32()? as usize;
    if num_valid == num_pixels {
        if num_bytes != 0 {
            return Err(Error::InvalidBlob(
                "full-valid LERC blob unexpectedly contains mask bytes".into(),
            ));
        }
        return Ok(None);
    }

    if num_valid == 0 {
        cursor.skip(num_bytes)?;
        return Ok(Some(vec![0; num_pixels]));
    }

    if num_bytes == 0 {
        let mask = inherited_mask.ok_or(Error::UnsupportedFeature(
            "external masks are not yet supported",
        ))?;
        if mask.len() != num_pixels {
            return Err(Error::InvalidBlob(
                "inherited mask length does not match the current LERC blob".into(),
            ));
        }
        let inherited_valid = mask.iter().filter(|&&value| value != 0).count();
        if inherited_valid != num_valid {
            return Err(Error::InvalidBlob(
                "inherited mask valid count does not match the current LERC blob".into(),
            ));
        }
        return Ok(Some(mask.to_vec()));
    }

    let bitset = decode_mask_rle(cursor.read_bytes(num_bytes)?, num_pixels.div_ceil(8))?;
    Ok(Some(unpack_mask_bitset(&bitset, num_pixels)))
}

pub(crate) fn decode_mask_rle(encoded: &[u8], bitset_len: usize) -> Result<Vec<u8>> {
    let mut bitset = vec![0u8; bitset_len];
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

pub(crate) fn unpack_mask_bitset(bitset: &[u8], num_pixels: usize) -> Vec<u8> {
    let mut mask = vec![0u8; num_pixels];
    for (i, item) in mask.iter_mut().enumerate() {
        let byte = bitset[i >> 3];
        let bit = 7 - (i & 7);
        *item = (byte >> bit) & 1;
    }
    mask
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
        return Ok(T::into_pixel_data(vec![T::default(); sample_count]));
    }

    if info.z_min == info.z_max {
        return Ok(T::into_pixel_data(constant_pixels::<T>(
            info.data_type,
            num_pixels,
            info.depth as usize,
            mask,
            ConstantValues::Single(info.z_max),
        )));
    }

    if let Some(ranges) = depth_ranges {
        if ranges.min_values == ranges.max_values {
            return Ok(T::into_pixel_data(constant_pixels::<T>(
                info.data_type,
                num_pixels,
                info.depth as usize,
                mask,
                ConstantValues::PerDepth(&ranges.max_values),
            )));
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

    if version > 1 && info.data_type.code() <= 1 && (info.max_z_error - 0.5).abs() < 1e-5 {
        let encode_mode = cursor.read_u8()?;
        if encode_mode > 2 || (version < 4 && encode_mode > 1) {
            return Err(Error::InvalidBlob(format!(
                "invalid Huffman flag {encode_mode}"
            )));
        }
        if encode_mode != 0 {
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

    if version > 1 && info.data_type.code() <= 1 && (info.max_z_error - 0.5).abs() < 1e-5 {
        let encode_mode = cursor.read_u8()?;
        if encode_mode > 2 || (version < 4 && encode_mode > 1) {
            return Err(Error::InvalidBlob(format!(
                "invalid Huffman flag {encode_mode}"
            )));
        }
        if encode_mode != 0 {
            return decode_huffman_into(cursor, info, mask, encode_mode == 1, out);
        }
    }

    decode_tiles_into(cursor, info, depth_ranges, mask, out)
}

enum ConstantValues<'a> {
    Single(f64),
    PerDepth(&'a [f64]),
}

fn constant_pixels<T: Sample>(
    data_type: DataType,
    num_pixels: usize,
    depth: usize,
    mask: Option<&[u8]>,
    values: ConstantValues<'_>,
) -> Vec<T> {
    let mut out = vec![T::default(); num_pixels * depth];
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
    out
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
    let raw = read_values_as::<T>(cursor.read_bytes(sample_len)?, info.data_type)?;

    if num_valid == num_pixels {
        return Ok(T::into_pixel_data(raw));
    }

    let mask = mask.ok_or(Error::InvalidBlob(
        "partial-valid one-sweep block is missing its decoded mask".into(),
    ))?;
    let mut out = vec![T::default(); info.sample_count()?];
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

fn decode_tiles<T: Sample>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
) -> Result<PixelData> {
    let width = info.width as usize;
    let height = info.height as usize;
    let depth = info.depth as usize;
    let num_pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
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

    let mut result = vec![T::default(); info.sample_count()?];
    let mut block_buffer = vec![0.0; micro_block_size * micro_block_size];
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
                        if is_diff_encoding {
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        let prev = result[sample_index(pixel, depth, dim - 1)];
                                        result[sample_index(pixel, depth, dim)] = prev;
                                    }
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
                                    result[sample_index(pixel, depth, dim)] = raw_values[src];
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
                                            (result[sample_index(pixel, depth, dim - 1)].to_f64()
                                                + offset)
                                                .min(z_max)
                                        } else {
                                            offset
                                        };
                                        result[sample_index(pixel, depth, dim)] =
                                            output_value::<T>(value, info.data_type);
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
                                            block_buffer[src]
                                                + result[sample_index(pixel, depth, dim - 1)]
                                                    .to_f64()
                                        } else {
                                            block_buffer[src]
                                        };
                                        result[sample_index(pixel, depth, dim)] =
                                            output_value::<T>(value, info.data_type);
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

    let _ = num_pixels;
    Ok(T::into_pixel_data(result))
}

fn decode_tiles_into<T: Sample, W: BandWriter<T>>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
    out: &mut W,
) -> Result<()> {
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

    let mut block_buffer = vec![0.0; micro_block_size * micro_block_size];
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
                        if is_diff_encoding {
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        out.write(pixel, dim, out.read(pixel, dim - 1));
                                    }
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
