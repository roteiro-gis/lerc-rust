//! Pure-Rust LERC decoder.
//!
//! The public API is decoder-first and mirrors the high-value inspection and
//! decode entry points from Esri's LERC project:
//!
//! - inspect a blob with [`get_blob_info`]
//! - count concatenated blobs with [`get_band_count`]
//! - decode native pixel buffers with [`decode`]
//! - decode promoted `f64` buffers with [`decode_to_f64`]
//! - decode directly into `ndarray::ArrayD` with [`decode_ndarray`]

mod io;

use std::cmp;

use io::Cursor;
use lerc_core::{
    BlobInfo, DataType, Decoded, DecodedF64, Error, NdArrayElement, PixelData, Result, Version,
};
use ndarray::ArrayD;

const MAGIC_LERC1_PREFIX: &[u8; 9] = b"CntZImage";
const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";
const HUFFMAN_LUT_BITS_MAX: u8 = 12;

#[derive(Debug, Clone)]
struct DepthRanges {
    min_values: Vec<f64>,
    max_values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct HuffmanEntry {
    bit_len: u8,
    value: u32,
}

#[derive(Debug, Default)]
struct HuffmanNode {
    value: Option<u32>,
    left: Option<Box<HuffmanNode>>,
    right: Option<Box<HuffmanNode>>,
}

#[derive(Debug)]
struct HuffmanInfo {
    quick_lut: Vec<Option<HuffmanEntry>>,
    quick_bits: u8,
    max_bits: u8,
    tree: HuffmanNode,
    stuffed_data: Vec<u32>,
    src_ptr: usize,
    bit_pos: u8,
}

#[derive(Debug)]
struct HuffmanStream<'a> {
    words: &'a [u32],
    src_ptr: usize,
    bit_pos: u8,
}

#[derive(Debug, Clone)]
struct Lerc1PixelsHeader {
    num_blocks_y: usize,
    num_blocks_x: usize,
    max_value: f32,
}

#[derive(Debug, Clone)]
enum Lerc1Block {
    Zero,
    Constant(f32),
    Raw(Vec<f32>),
    Stuffed {
        offset: f32,
        bits_per_pixel: u8,
        num_valid_pixels: usize,
        stuffed_data: Vec<u32>,
    },
}

#[derive(Debug, Clone)]
struct Lerc1Blob {
    info: BlobInfo,
    mask: Option<Vec<u8>>,
    pixels: Lerc1PixelsHeader,
    blocks: Vec<Lerc1Block>,
    actual_num_blocks_y: usize,
    actual_num_blocks_x: usize,
    base_block_height: usize,
    base_block_width: usize,
    eof_offset: usize,
}

fn is_lerc1(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC1_PREFIX)
}

fn is_lerc2(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC2)
}

pub fn get_blob_info(blob: &[u8]) -> Result<BlobInfo> {
    if is_lerc1(blob) {
        let mut parsed = parse_lerc1(blob, None)?;
        if parsed.info.valid_pixel_count != 0 {
            let pixels = decode_lerc1_pixels(&parsed)?;
            let (z_min, z_max) = scan_range(&pixels, parsed.mask.as_deref())?;
            parsed.info.z_min = z_min;
            parsed.info.z_max = z_max;
        }
        return Ok(parsed.info);
    }

    let (mut info, mut cursor) = parse_lerc2(blob)?;
    let _mask = read_mask(&mut cursor, &info)?;
    if should_read_depth_ranges(&info) {
        let ranges = read_depth_ranges(&mut cursor, &info)?;
        info.min_values = Some(ranges.min_values);
        info.max_values = Some(ranges.max_values);
    }
    Ok(info)
}

pub fn get_band_count(blob: &[u8]) -> Result<usize> {
    let mut offset = 0usize;
    let mut count = 0usize;
    let mut lerc1_mask: Option<Option<Vec<u8>>> = None;

    while offset < blob.len() {
        let slice = &blob[offset..];
        let next_len = if is_lerc1(slice) {
            let parsed = parse_lerc1(slice, lerc1_mask.as_ref())?;
            lerc1_mask = Some(parsed.mask.clone());
            parsed.eof_offset
        } else if is_lerc2(slice) {
            let (info, _) = parse_lerc2(slice)?;
            lerc1_mask = None;
            info.blob_size
        } else {
            return Err(Error::InvalidMagic);
        };

        let next = offset
            .checked_add(next_len)
            .ok_or_else(|| Error::InvalidBlob("band offset overflow".into()))?;
        if next <= offset || next > blob.len() {
            return Err(Error::InvalidBlob(
                "invalid concatenated band blob size".into(),
            ));
        }

        offset = next;
        count += 1;
    }

    Ok(count)
}

pub fn decode(blob: &[u8]) -> Result<Decoded> {
    if is_lerc1(blob) {
        let mut parsed = parse_lerc1(blob, None)?;
        let pixels = decode_lerc1_pixels(&parsed)?;
        if parsed.info.valid_pixel_count != 0 {
            let (z_min, z_max) = scan_range(&pixels, parsed.mask.as_deref())?;
            parsed.info.z_min = z_min;
            parsed.info.z_max = z_max;
        }
        return Ok(Decoded {
            info: parsed.info,
            pixels,
            mask: parsed.mask,
        });
    }

    let (mut info, mut cursor) = parse_lerc2(blob)?;
    let mask = read_mask(&mut cursor, &info)?;

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

fn parse_lerc1(blob: &[u8], shared_mask: Option<&Option<Vec<u8>>>) -> Result<Lerc1Blob> {
    let mut cursor = Cursor::new(blob);
    let magic = cursor.read_bytes(10)?;
    if !magic.starts_with(MAGIC_LERC1_PREFIX) {
        return Err(Error::InvalidMagic);
    }

    let version = cursor.read_i32()?;
    if version < 0 {
        return Err(Error::UnsupportedVersion(version as u32));
    }
    let image_type = cursor.read_i32()?;
    let height = cursor.read_u32()?;
    let width = cursor.read_u32()?;
    let max_z_error = cursor.read_f64()?;

    let mask = if let Some(shared_mask) = shared_mask {
        shared_mask.clone()
    } else {
        read_lerc1_mask(&mut cursor, width, height)?
    };

    let pixels_num_blocks_y = cursor.read_u32()? as usize;
    let pixels_num_blocks_x = cursor.read_u32()? as usize;
    let _pixels_num_bytes = cursor.read_u32()? as usize;
    let pixels_max_value = cursor.read_f32()?;

    if pixels_num_blocks_x == 0 || pixels_num_blocks_y == 0 {
        return Err(Error::InvalidHeader("Lerc1 block grid must be non-zero"));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let num_pixels = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
    let base_block_width = width_usize / pixels_num_blocks_x;
    let base_block_height = height_usize / pixels_num_blocks_y;
    let actual_num_blocks_x =
        pixels_num_blocks_x + usize::from(width_usize % pixels_num_blocks_x != 0);
    let actual_num_blocks_y =
        pixels_num_blocks_y + usize::from(height_usize % pixels_num_blocks_y != 0);

    let valid_pixel_count = match mask.as_deref() {
        Some(mask) => mask.iter().map(|&value| u32::from(value != 0)).sum(),
        None => num_pixels as u32,
    };

    let mut blocks = Vec::with_capacity(actual_num_blocks_x * actual_num_blocks_y);
    for block_y in 0..actual_num_blocks_y {
        let this_block_height =
            if block_y + 1 == actual_num_blocks_y && height_usize % pixels_num_blocks_y != 0 {
                height_usize % pixels_num_blocks_y
            } else {
                base_block_height
            };
        if this_block_height == 0 {
            continue;
        }
        for block_x in 0..actual_num_blocks_x {
            let this_block_width =
                if block_x + 1 == actual_num_blocks_x && width_usize % pixels_num_blocks_x != 0 {
                    width_usize % pixels_num_blocks_x
                } else {
                    base_block_width
                };
            if this_block_width == 0 {
                continue;
            }

            let block_valid_pixels = if let Some(mask) = mask.as_deref() {
                count_valid_in_block(
                    mask,
                    width_usize,
                    block_x * base_block_width,
                    block_y * base_block_height,
                    this_block_width,
                    this_block_height,
                )
            } else {
                this_block_width * this_block_height
            };

            let header_byte = cursor.read_u8()?;
            let encoding = header_byte & 63;
            if encoding > 3 {
                return Err(Error::InvalidBlob(format!(
                    "invalid Lerc1 block encoding {encoding}"
                )));
            }
            if encoding == 2 {
                blocks.push(Lerc1Block::Zero);
                continue;
            }

            let offset = if header_byte != 0 && header_byte != 2 {
                Some(read_lerc1_offset(&mut cursor, header_byte >> 6)?)
            } else {
                None
            };

            if encoding == 3 {
                blocks.push(Lerc1Block::Constant(offset.ok_or(Error::InvalidBlob(
                    "Lerc1 constant block is missing its offset".into(),
                ))?));
                continue;
            }

            if encoding == 0 {
                let byte_len = block_valid_pixels.checked_mul(4).ok_or_else(|| {
                    Error::InvalidBlob("Lerc1 raw block byte count overflows usize".into())
                })?;
                let values = cursor
                    .read_bytes(byte_len)?
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                blocks.push(Lerc1Block::Raw(values));
                continue;
            }

            let packed_header = cursor.read_u8()?;
            let bits_per_pixel = packed_header & 63;
            let num_valid_pixels = match packed_header >> 6 {
                0 => cursor.read_u32()? as usize,
                1 => read_u16(cursor.read_bytes(2)?)? as usize,
                2 => cursor.read_u8()? as usize,
                other => {
                    return Err(Error::InvalidBlob(format!(
                        "invalid Lerc1 valid pixel count type {other}"
                    )))
                }
            };
            let data_bytes = (num_valid_pixels * usize::from(bits_per_pixel)).div_ceil(8);
            let stuffed_data = words_from_padded(cursor.read_bytes(data_bytes)?);
            blocks.push(Lerc1Block::Stuffed {
                offset: offset.ok_or(Error::InvalidBlob(
                    "Lerc1 bit-stuffed block is missing its offset".into(),
                ))?,
                bits_per_pixel,
                num_valid_pixels,
                stuffed_data,
            });
        }
    }

    let eof_offset = cursor.offset();
    let info = BlobInfo {
        version: Version::Lerc1(version as u32),
        data_type: map_lerc1_data_type(image_type),
        width,
        height,
        depth: 1,
        min_values: None,
        max_values: None,
        valid_pixel_count,
        micro_block_size: 0,
        blob_size: eof_offset,
        max_z_error,
        z_min: 0.0,
        z_max: pixels_max_value as f64,
    };

    Ok(Lerc1Blob {
        info,
        mask,
        pixels: Lerc1PixelsHeader {
            num_blocks_y: pixels_num_blocks_y,
            num_blocks_x: pixels_num_blocks_x,
            max_value: pixels_max_value,
        },
        blocks,
        actual_num_blocks_y,
        actual_num_blocks_x,
        base_block_height,
        base_block_width,
        eof_offset,
    })
}

fn map_lerc1_data_type(_image_type: i32) -> DataType {
    DataType::F32
}

fn read_lerc1_offset(cursor: &mut Cursor<'_>, offset_type: u8) -> Result<f32> {
    match offset_type {
        0 => cursor.read_f32(),
        1 => Ok(cursor.read_i16()? as f32),
        2 => Ok(i8::from_le_bytes([cursor.read_u8()?]) as f32),
        _ => Err(Error::InvalidBlob(format!(
            "invalid Lerc1 block offset type {offset_type}"
        ))),
    }
}

fn read_lerc1_mask(cursor: &mut Cursor<'_>, width: u32, height: u32) -> Result<Option<Vec<u8>>> {
    let num_blocks_y = cursor.read_u32()?;
    let num_blocks_x = cursor.read_u32()?;
    let num_bytes = cursor.read_u32()? as usize;
    let max_value = cursor.read_f32()?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
    let bitset_len = num_pixels.div_ceil(8);

    if num_bytes > 0 {
        let bitset = decode_mask_rle(cursor.read_bytes(num_bytes)?, bitset_len)?;
        return Ok(Some(unpack_mask_bitset(&bitset, num_pixels)));
    }

    if num_blocks_y == 0 && num_blocks_x == 0 && max_value == 0.0 {
        return Ok(Some(vec![0; num_pixels]));
    }

    Ok(None)
}

fn decode_lerc1_pixels(parsed: &Lerc1Blob) -> Result<PixelData> {
    let width = parsed.info.width as usize;
    let height = parsed.info.height as usize;
    let mask = parsed.mask.as_deref();
    let mut result = vec![0.0f32; width * height];
    let mut block_buffer = vec![0.0f64; parsed.base_block_width * parsed.base_block_height];
    let mut block_index = 0usize;

    for block_y in 0..parsed.actual_num_blocks_y {
        let this_block_height = if block_y + 1 == parsed.actual_num_blocks_y
            && height % parsed.pixels.num_blocks_y != 0
        {
            height % parsed.pixels.num_blocks_y
        } else {
            parsed.base_block_height
        };
        if this_block_height == 0 {
            continue;
        }

        for block_x in 0..parsed.actual_num_blocks_x {
            let this_block_width = if block_x + 1 == parsed.actual_num_blocks_x
                && width % parsed.pixels.num_blocks_x != 0
            {
                width % parsed.pixels.num_blocks_x
            } else {
                parsed.base_block_width
            };
            if this_block_width == 0 {
                continue;
            }

            let block = &parsed.blocks[block_index];
            block_index += 1;

            let mut stuffed_values: Option<&[f64]> = None;
            let (raw_values, constant_value) = match block {
                Lerc1Block::Zero => (None, Some(0.0f32)),
                Lerc1Block::Constant(value) => (None, Some(*value)),
                Lerc1Block::Raw(values) => (Some(values.as_slice()), None),
                Lerc1Block::Stuffed {
                    offset,
                    bits_per_pixel,
                    num_valid_pixels,
                    stuffed_data,
                } => {
                    if *num_valid_pixels > block_buffer.len() {
                        return Err(Error::InvalidBlob(
                            "Lerc1 stuffed block expands beyond its output buffer".into(),
                        ));
                    }
                    block_buffer[..*num_valid_pixels].fill(0.0);
                    unstuff_v2(
                        stuffed_data,
                        &mut block_buffer[..*num_valid_pixels],
                        *bits_per_pixel,
                        *num_valid_pixels,
                        None,
                        Some(*offset as f64),
                        2.0 * parsed.info.max_z_error,
                        parsed.pixels.max_value as f64,
                    );
                    stuffed_values = Some(&block_buffer[..*num_valid_pixels]);
                    (None, None)
                }
            };

            let mut value_index = 0usize;
            for row in 0..this_block_height {
                let pixel_row = block_y * parsed.base_block_height + row;
                for col in 0..this_block_width {
                    let pixel = pixel_row * width + block_x * parsed.base_block_width + col;
                    if mask.map(|mask| mask[pixel] != 0).unwrap_or(true) {
                        let value = if let Some(value) = constant_value {
                            value
                        } else if let Some(values) = raw_values {
                            let value = values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 raw block payload ended early".into())
                            })?;
                            value
                        } else if let Some(values) = stuffed_values {
                            values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 stuffed block payload ended early".into())
                            })? as f32
                        } else {
                            unreachable!()
                        };
                        result[pixel] = value;
                        value_index += 1;
                    }
                }
            }

            let expected_values = match block {
                Lerc1Block::Zero | Lerc1Block::Constant(_) => 0,
                Lerc1Block::Raw(values) => values.len(),
                Lerc1Block::Stuffed {
                    num_valid_pixels, ..
                } => *num_valid_pixels,
            };
            if expected_values != 0 && value_index != expected_values {
                return Err(Error::InvalidBlob(
                    "Lerc1 block payload does not match the block mask".into(),
                ));
            }
        }
    }

    Ok(PixelData::F32(result))
}

fn parse_lerc2(blob: &[u8]) -> Result<(BlobInfo, Cursor<'_>)> {
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

fn read_mask(cursor: &mut Cursor<'_>, info: &BlobInfo) -> Result<Option<Vec<u8>>> {
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
        return Err(Error::UnsupportedFeature(
            "external masks are not yet supported",
        ));
    }

    let bitset = decode_mask_rle(cursor.read_bytes(num_bytes)?, num_pixels.div_ceil(8))?;
    Ok(Some(unpack_mask_bitset(&bitset, num_pixels)))
}

fn decode_mask_rle(encoded: &[u8], bitset_len: usize) -> Result<Vec<u8>> {
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

fn unpack_mask_bitset(bitset: &[u8], num_pixels: usize) -> Vec<u8> {
    let mut mask = vec![0u8; num_pixels];
    for (i, item) in mask.iter_mut().enumerate() {
        let byte = bitset[i >> 3];
        let bit = 7 - (i & 7);
        *item = ((byte >> bit) & 1) as u8;
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
    let num_pixels = info.pixel_count()?;
    let sample_count = info.sample_count()?;

    if info.valid_pixel_count == 0 {
        return Ok(zeroed_pixels(info.data_type, sample_count));
    }

    if info.z_min == info.z_max {
        return Ok(constant_pixels(
            info.data_type,
            num_pixels,
            info.depth as usize,
            mask,
            ConstantValues::Single(info.z_max),
        ));
    }

    if let Some(ranges) = depth_ranges {
        if ranges.min_values == ranges.max_values {
            return Ok(constant_pixels(
                info.data_type,
                num_pixels,
                info.depth as usize,
                mask,
                ConstantValues::PerDepth(ranges.max_values.clone()),
            ));
        }
    }

    let one_sweep = cursor.read_u8()? != 0;
    if one_sweep {
        return decode_one_sweep(cursor, info, mask);
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
            return decode_huffman(cursor, info, depth_ranges, mask, encode_mode == 1);
        }
    }

    decode_tiles(cursor, info, depth_ranges, mask)
}

fn decode_one_sweep(
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
    let raw = read_typed_values(cursor.read_bytes(sample_len)?, info.data_type)?;

    if num_valid == num_pixels {
        return Ok(cast_pixels(info.data_type, raw));
    }

    let mask = mask.ok_or(Error::InvalidBlob(
        "partial-valid one-sweep block is missing its decoded mask".into(),
    ))?;
    scatter_valid_samples(info.data_type, info.sample_count()?, depth, mask, &raw)
}

fn decode_tiles(
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

    let mut result_bsq = vec![0.0; info.sample_count()?];
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
                let dim_base = dim * num_pixels;
                let prev_dim_base = dim.checked_sub(1).map(|prev| prev * num_pixels);
                let z_max = depth_ranges
                    .and_then(|ranges| ranges.max_values.get(dim))
                    .copied()
                    .unwrap_or(info.z_max);

                match block_encoding {
                    2 => {
                        if is_diff_encoding {
                            let prev_dim_base = prev_dim_base.ok_or(Error::InvalidBlob(
                                "diff encoding requires a previous dimension".into(),
                            ))?;
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        result_bsq[dim_base + pixel] =
                                            result_bsq[prev_dim_base + pixel];
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
                            read_typed_values(cursor.read_bytes(byte_len)?, info.data_type)?;
                        let mut src = 0usize;
                        for row in 0..this_block_height {
                            let pixel_row = block_y * micro_block_size + row;
                            for col in 0..this_block_width {
                                let pixel = pixel_row * width + block_x * micro_block_size + col;
                                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                    result_bsq[dim_base + pixel] = raw_values[src];
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
                        let offset =
                            read_scalar(cursor.read_bytes(offset_type.byte_len())?, offset_type)?;

                        if block_encoding == 3 {
                            for row in 0..this_block_height {
                                let pixel_row = block_y * micro_block_size + row;
                                for col in 0..this_block_width {
                                    let pixel =
                                        pixel_row * width + block_x * micro_block_size + col;
                                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                                        result_bsq[dim_base + pixel] = if is_diff_encoding {
                                            let prev_dim_base =
                                                prev_dim_base.ok_or(Error::InvalidBlob(
                                                    "diff encoding requires a previous dimension"
                                                        .into(),
                                                ))?;
                                            (result_bsq[prev_dim_base + pixel] + offset).min(z_max)
                                        } else {
                                            offset
                                        };
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
                                        let value = block_buffer[src];
                                        result_bsq[dim_base + pixel] = if is_diff_encoding {
                                            let prev_dim_base =
                                                prev_dim_base.ok_or(Error::InvalidBlob(
                                                    "diff encoding requires a previous dimension"
                                                        .into(),
                                                ))?;
                                            value + result_bsq[prev_dim_base + pixel]
                                        } else {
                                            value
                                        };
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

    let values = if depth > 1 {
        swap_bsq_to_bip(&result_bsq, num_pixels, depth)
    } else {
        result_bsq
    };
    Ok(cast_pixels(info.data_type, values))
}

fn decode_huffman(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    depth_ranges: Option<&DepthRanges>,
    mask: Option<&[u8]>,
    delta_encode: bool,
) -> Result<PixelData> {
    let width = info.width as usize;
    let height = info.height as usize;
    let depth = info.depth as usize;
    let num_pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
    let mut result_bsq = vec![0.0; info.sample_count()?];

    let huffman = read_huffman_tree(cursor, info)?;
    let mut stream = HuffmanStream {
        words: &huffman.stuffed_data,
        src_ptr: huffman.src_ptr,
        bit_pos: huffman.bit_pos,
    };
    if stream.bit_pos > 0 {
        stream.src_ptr += 1;
        stream.bit_pos = 0;
    }

    let offset = if info.data_type == DataType::I8 {
        128.0
    } else {
        0.0
    };

    if depth < 2 || delta_encode {
        for dim in 0..depth {
            let dim_base = dim * num_pixels;
            let mut prev_value = 0.0;
            for row in 0..height {
                for col in 0..width {
                    let pixel = row * width + col;
                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                        let value = read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                        if delta_encode {
                            let mut delta = value;
                            if col > 0 && mask.map(|m| m[pixel - 1] != 0).unwrap_or(true) {
                                delta += prev_value;
                            } else if row > 0 && mask.map(|m| m[pixel - width] != 0).unwrap_or(true)
                            {
                                delta += result_bsq[dim_base + pixel - width];
                            } else {
                                delta += prev_value;
                            }
                            let wrapped = ((delta as i64) & 0xFF) as f64;
                            result_bsq[dim_base + pixel] = wrapped;
                            prev_value = wrapped;
                        } else {
                            result_bsq[dim_base + pixel] = value;
                        }
                    }
                }
            }
        }
    } else {
        for row in 0..height {
            for col in 0..width {
                let pixel = row * width + col;
                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                    for dim in 0..depth {
                        let dim_base = dim * num_pixels;
                        result_bsq[dim_base + pixel] =
                            read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                    }
                }
            }
        }
    }

    let consumed_bytes = (stream.src_ptr + 1) * 4 + usize::from(stream.bit_pos > 0) * 4;
    cursor.skip(consumed_bytes)?;

    let values = if depth > 1 {
        swap_bsq_to_bip(&result_bsq, num_pixels, depth)
    } else {
        result_bsq
    };

    let _ = depth_ranges;
    Ok(cast_pixels(info.data_type, values))
}

fn read_huffman_tree(cursor: &mut Cursor<'_>, info: &BlobInfo) -> Result<HuffmanInfo> {
    let version = read_i32(cursor.read_bytes(4)?)?;
    if version < 2 {
        return Err(Error::UnsupportedFeature("Huffman version < 2"));
    }
    let size = read_i32(cursor.read_bytes(4)?)?;
    let i0 = read_i32(cursor.read_bytes(4)?)?;
    let i1 = read_i32(cursor.read_bytes(4)?)?;
    if size <= 0 || i0 < 0 || i1 <= i0 {
        return Err(Error::InvalidBlob("invalid Huffman table header".into()));
    }

    let mut code_lengths = vec![
        0.0;
        usize::try_from(i1 - i0).map_err(|_| {
            Error::InvalidBlob("Huffman code length count does not fit in memory".into())
        })?
    ];
    let _ = decode_bits(
        cursor,
        match info.version {
            Version::Lerc2(version) => version,
            _ => unreachable!("Lerc2 Huffman decode called with a non-Lerc2 blob"),
        },
        &mut code_lengths,
        None,
        info.max_z_error,
        info.z_max,
    )?;

    let size = size as usize;
    let i0 = i0 as usize;
    let i1 = i1 as usize;
    let mut code_table: Vec<Option<(u8, u32)>> = vec![None; size];
    for i in i0..i1 {
        let j = if i < size { i } else { i - size };
        code_table[j] = Some((code_lengths[i - i0] as u8, 0));
    }

    let stuffed_data = words_from_padded(cursor.read_bytes(cursor.remaining())?);
    let mut stream = HuffmanStream {
        words: &stuffed_data,
        src_ptr: 0,
        bit_pos: 0,
    };

    for entry in code_table.iter_mut().flatten() {
        if entry.0 > 0 {
            entry.1 = stream.peek_bits(entry.0 as usize)?;
            stream.advance(entry.0 as usize)?;
        }
    }

    let mut max_bits = 0u8;
    for (bit_len, _) in code_table.iter().flatten() {
        max_bits = max_bits.max(*bit_len);
    }
    let quick_bits = cmp::min(max_bits, HUFFMAN_LUT_BITS_MAX);
    let mut quick_lut = vec![None; 1usize << quick_bits];
    let mut tree = HuffmanNode::default();

    for (symbol, entry) in code_table.into_iter().enumerate() {
        let Some((bit_len, code)) = entry else {
            continue;
        };
        if bit_len == 0 {
            continue;
        }
        if bit_len <= quick_bits {
            let base = code << (quick_bits - bit_len);
            let num_entries = 1usize << (quick_bits - bit_len);
            for extra in 0..num_entries {
                quick_lut[(base as usize) | extra] = Some(HuffmanEntry {
                    bit_len,
                    value: symbol as u32,
                });
            }
        } else {
            insert_huffman_code(&mut tree, code, bit_len, symbol as u32)?;
        }
    }

    let src_ptr = stream.src_ptr;
    let bit_pos = stream.bit_pos;

    Ok(HuffmanInfo {
        quick_lut,
        quick_bits,
        max_bits,
        tree,
        stuffed_data,
        src_ptr,
        bit_pos,
    })
}

fn insert_huffman_code(root: &mut HuffmanNode, code: u32, bit_len: u8, value: u32) -> Result<()> {
    let mut node = root;
    for shift in (0..bit_len).rev() {
        let bit = (code >> shift) & 1;
        if shift == 0 {
            let child = if bit == 0 {
                node.left
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            } else {
                node.right
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            };
            child.value = Some(value);
        } else {
            node = if bit == 0 {
                node.left
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            } else {
                node.right
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            };
        }
    }
    Ok(())
}

fn read_huffman_symbol(stream: &mut HuffmanStream<'_>, info: &HuffmanInfo) -> Result<u32> {
    if let Some(entry) =
        info.quick_lut[stream.peek_bits(info.quick_bits as usize)? as usize].as_ref()
    {
        stream.advance(entry.bit_len as usize)?;
        return Ok(entry.value);
    }

    let value = stream.peek_bits(info.max_bits as usize)?;
    let mut node = &info.tree;
    for used_bits in 0..info.max_bits {
        let bit = (value >> (info.max_bits - used_bits - 1)) & 1;
        node = if bit == 0 {
            node.left.as_deref().ok_or_else(|| {
                Error::InvalidBlob("Huffman decode walked a missing left node".into())
            })?
        } else {
            node.right.as_deref().ok_or_else(|| {
                Error::InvalidBlob("Huffman decode walked a missing right node".into())
            })?
        };

        if node.left.is_none() && node.right.is_none() {
            let symbol = node
                .value
                .ok_or_else(|| Error::InvalidBlob("Huffman leaf is missing a symbol".into()))?;
            stream.advance((used_bits + 1) as usize)?;
            return Ok(symbol);
        }
    }

    Err(Error::InvalidBlob(
        "Huffman symbol exceeded the configured code width".into(),
    ))
}

fn decode_bits(
    cursor: &mut Cursor<'_>,
    version: u32,
    out: &mut [f64],
    offset: Option<f64>,
    max_z_error: f64,
    max_value: f64,
) -> Result<usize> {
    let header_byte = cursor.read_u8()?;
    let bits67 = header_byte >> 6;
    let count_bytes = if bits67 == 0 { 4 } else { 3 - bits67 as usize };
    let do_lut = (header_byte & 32) != 0;
    let num_bits = header_byte & 31;

    let num_elements = match count_bytes {
        1 => cursor.read_u8()? as usize,
        2 => read_u16(cursor.read_bytes(2)?)? as usize,
        4 => cursor.read_u32()? as usize,
        _ => {
            return Err(Error::InvalidBlob(
                "invalid valid pixel count field width".into(),
            ))
        }
    };
    if num_elements > out.len() {
        return Err(Error::InvalidBlob(
            "bit-stuffed block expands beyond its output buffer".into(),
        ));
    }

    if do_lut {
        let offset = offset.ok_or(Error::InvalidBlob(
            "LUT-compressed block is missing its base offset".into(),
        ))?;
        let lut_bytes = cursor.read_u8()? as usize;
        if lut_bytes == 0 {
            return Err(Error::InvalidBlob(
                "LUT-compressed block has zero LUT bytes".into(),
            ));
        }
        let lut_data_bytes = ((lut_bytes - 1) * usize::from(num_bits)).div_ceil(8);
        let lut_words = words_from_padded(cursor.read_bytes(lut_data_bytes)?);
        let lut_bits_per_element = bits_required(lut_bytes - 1);
        let stuffed_data_bytes = (num_elements * usize::from(lut_bits_per_element)).div_ceil(8);
        let stuffed_words = words_from_padded(cursor.read_bytes(stuffed_data_bytes)?);

        let lut_values = if version >= 3 {
            unstuff_lut_v3(
                &lut_words,
                num_bits,
                lut_bytes - 1,
                offset,
                2.0 * max_z_error,
                max_value,
            )
        } else {
            unstuff_lut_v2(
                &lut_words,
                num_bits,
                lut_bytes - 1,
                offset,
                2.0 * max_z_error,
                max_value,
            )
        };

        if version >= 3 {
            unstuff_v3(
                &stuffed_words,
                &mut out[..num_elements],
                lut_bits_per_element,
                num_elements,
                Some(&lut_values),
                None,
                0.0,
                max_value,
            );
        } else {
            unstuff_v2(
                &stuffed_words,
                &mut out[..num_elements],
                lut_bits_per_element,
                num_elements,
                Some(&lut_values),
                None,
                0.0,
                max_value,
            );
        }
        return Ok(num_elements);
    }

    if num_bits == 0 {
        out[..num_elements].fill(offset.unwrap_or(0.0));
        return Ok(num_elements);
    }

    let stuffed_data_bytes = (num_elements * usize::from(num_bits)).div_ceil(8);
    let stuffed_words = words_from_padded(cursor.read_bytes(stuffed_data_bytes)?);
    match (version >= 3, offset) {
        (true, Some(offset)) => unstuff_v3(
            &stuffed_words,
            &mut out[..num_elements],
            num_bits,
            num_elements,
            None,
            Some(offset),
            2.0 * max_z_error,
            max_value,
        ),
        (true, None) => original_unstuff_v3(&stuffed_words, &mut out[..num_elements], num_bits),
        (false, Some(offset)) => unstuff_v2(
            &stuffed_words,
            &mut out[..num_elements],
            num_bits,
            num_elements,
            None,
            Some(offset),
            2.0 * max_z_error,
            max_value,
        ),
        (false, None) => original_unstuff_v2(&stuffed_words, &mut out[..num_elements], num_bits),
    }
    Ok(num_elements)
}

fn unstuff_v2(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    num_pixels: usize,
    lut_values: Option<&[f64]>,
    offset: Option<f64>,
    scale: f64,
    max_value: f64,
) {
    let bit_mask = if bits_per_pixel == 32 {
        u32::MAX
    } else {
        (1u32 << bits_per_pixel) - 1
    };
    let mut words = src.to_vec();
    let num_invalid_tail_bytes =
        words.len() * 4 - (usize::from(bits_per_pixel) * num_pixels).div_ceil(8);
    if let Some(last) = words.last_mut() {
        *last <<= 8 * num_invalid_tail_bytes;
    }

    let mut index = 0usize;
    let mut bits_left = 0usize;
    let mut buffer = 0u32;
    let nmax = offset.map(|offset| quantized_nmax(offset, scale, max_value));

    for item in dest.iter_mut().take(num_pixels) {
        if bits_left == 0 {
            buffer = words[index];
            index += 1;
            bits_left = 32;
        }
        let n = if bits_left >= bits_per_pixel as usize {
            let n = (buffer >> (bits_left - bits_per_pixel as usize)) & bit_mask;
            bits_left -= bits_per_pixel as usize;
            n
        } else {
            let missing_bits = bits_per_pixel as usize - bits_left;
            let mut n = ((buffer & bit_mask) << missing_bits) & bit_mask;
            buffer = words[index];
            index += 1;
            bits_left = 32 - missing_bits;
            n += buffer >> bits_left;
            n
        };
        *item = match (lut_values, offset) {
            (Some(lut_values), _) => lut_values[n as usize],
            (None, Some(offset)) => quantized_value(offset, scale, max_value, n, nmax.unwrap()),
            (None, None) => n as f64,
        };
    }
}

fn unstuff_v3(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    num_pixels: usize,
    lut_values: Option<&[f64]>,
    offset: Option<f64>,
    scale: f64,
    max_value: f64,
) {
    let bit_mask = if bits_per_pixel == 32 {
        u32::MAX
    } else {
        (1u32 << bits_per_pixel) - 1
    };
    let mut index = 0usize;
    let mut bits_left = 0usize;
    let mut bit_pos = 0usize;
    let mut buffer = 0u32;
    let nmax = offset.map(|offset| quantized_nmax(offset, scale, max_value));

    for item in dest.iter_mut().take(num_pixels) {
        if bits_left == 0 {
            buffer = src[index];
            index += 1;
            bits_left = 32;
            bit_pos = 0;
        }
        let n = if bits_left >= bits_per_pixel as usize {
            let n = (buffer >> bit_pos) & bit_mask;
            bits_left -= bits_per_pixel as usize;
            bit_pos += bits_per_pixel as usize;
            n
        } else {
            let missing_bits = bits_per_pixel as usize - bits_left;
            let mut n = (buffer >> bit_pos) & bit_mask;
            buffer = src[index];
            index += 1;
            bits_left = 32 - missing_bits;
            n |=
                (buffer & ((1u32 << missing_bits) - 1)) << (bits_per_pixel as usize - missing_bits);
            bit_pos = missing_bits;
            n
        };
        *item = match (lut_values, offset) {
            (Some(lut_values), _) => lut_values[n as usize],
            (None, Some(offset)) => quantized_value(offset, scale, max_value, n, nmax.unwrap()),
            (None, None) => n as f64,
        };
    }
}

fn original_unstuff_v2(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) {
    unstuff_v2(src, dest, bits_per_pixel, dest.len(), None, None, 0.0, 0.0);
}

fn original_unstuff_v3(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) {
    unstuff_v3(src, dest, bits_per_pixel, dest.len(), None, None, 0.0, 0.0);
}

fn unstuff_lut_v2(
    src: &[u32],
    bits_per_pixel: u8,
    num_pixels: usize,
    offset: f64,
    scale: f64,
    max_value: f64,
) -> Vec<f64> {
    let mut values = vec![0.0; num_pixels];
    unstuff_v2(
        src,
        &mut values,
        bits_per_pixel,
        num_pixels,
        None,
        Some(offset),
        scale,
        max_value,
    );
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(offset);
    out.extend(values);
    out
}

fn unstuff_lut_v3(
    src: &[u32],
    bits_per_pixel: u8,
    num_pixels: usize,
    offset: f64,
    scale: f64,
    max_value: f64,
) -> Vec<f64> {
    let mut values = vec![0.0; num_pixels];
    unstuff_v3(
        src,
        &mut values,
        bits_per_pixel,
        num_pixels,
        None,
        Some(offset),
        scale,
        max_value,
    );
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(offset);
    out.extend(values);
    out
}

fn quantized_nmax(offset: f64, scale: f64, max_value: f64) -> f64 {
    if scale == 0.0 {
        f64::INFINITY
    } else {
        ((max_value - offset) / scale).ceil()
    }
}

fn quantized_value(offset: f64, scale: f64, max_value: f64, n: u32, nmax: f64) -> f64 {
    if (n as f64) < nmax {
        offset + (n as f64) * scale
    } else {
        max_value
    }
}

fn data_type_used(data_type: DataType, tc: u8) -> Result<DataType> {
    let code = match data_type.code() {
        2 | 4 => i32::from(data_type.code()) - i32::from(tc),
        3 | 5 => i32::from(data_type.code()) - 2 * i32::from(tc),
        6 => {
            if tc == 0 {
                6
            } else if tc == 1 {
                2
            } else {
                1
            }
        }
        7 => {
            if tc == 0 {
                7
            } else {
                7 - 2 * i32::from(tc) + 1
            }
        }
        _ => i32::from(data_type.code()),
    };
    DataType::from_code(code)
}

fn bits_required(max_index: usize) -> u8 {
    let mut bits = 0u8;
    let mut value = max_index;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

fn words_from_padded(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut padded = vec![0u8; bytes.len().div_ceil(4) * 4];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

impl<'a> HuffmanStream<'a> {
    fn peek_bits(&self, num_bits: usize) -> Result<u32> {
        if num_bits == 0 {
            return Ok(0);
        }
        if self.src_ptr >= self.words.len() {
            return Err(Error::Truncated {
                offset: self.src_ptr * 4,
                needed: 4,
                available: 0,
            });
        }
        let word = self.words[self.src_ptr];
        let mut value = word.wrapping_shl(self.bit_pos as u32) >> (32 - num_bits as u32);
        if 32 - (self.bit_pos as usize) < num_bits {
            let next = self.words.get(self.src_ptr + 1).copied().unwrap_or(0);
            value |= next >> (64 - self.bit_pos as usize - num_bits);
        }
        Ok(value)
    }

    fn advance(&mut self, bits: usize) -> Result<()> {
        let total = self.bit_pos as usize + bits;
        self.src_ptr = self
            .src_ptr
            .checked_add(total / 32)
            .ok_or_else(|| Error::InvalidBlob("Huffman bitstream pointer overflow".into()))?;
        self.bit_pos = (total % 32) as u8;
        Ok(())
    }
}

fn count_valid_in_block(
    mask: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
    block_width: usize,
    block_height: usize,
) -> usize {
    let mut count = 0usize;
    for row in 0..block_height {
        let row_offset = (y0 + row) * width + x0;
        for col in 0..block_width {
            count += usize::from(mask[row_offset + col] != 0);
        }
    }
    count
}

fn read_typed_values(bytes: &[u8], data_type: DataType) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(bytes.len() / data_type.byte_len());
    let mut cursor = Cursor::new(bytes);
    while cursor.offset() < bytes.len() {
        out.push(read_scalar(
            cursor.read_bytes(data_type.byte_len())?,
            data_type,
        )?);
    }
    Ok(out)
}

fn read_scalar(bytes: &[u8], data_type: DataType) -> Result<f64> {
    Ok(match data_type {
        DataType::I8 => i8::from_le_bytes([bytes[0]]) as f64,
        DataType::U8 => bytes[0] as f64,
        DataType::I16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::U16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
    })
}

fn read_u16(bytes: &[u8]) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_i32(bytes: &[u8]) -> Result<i32> {
    Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
}

fn swap_bsq_to_bip(values: &[f64], num_pixels: usize, depth: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    let mut dest = 0usize;
    for pixel in 0..num_pixels {
        let mut src = pixel;
        for _ in 0..depth {
            out[dest] = values[src];
            dest += 1;
            src += num_pixels;
        }
    }
    out
}

enum ConstantValues {
    Single(f64),
    PerDepth(Vec<f64>),
}

fn constant_pixels(
    data_type: DataType,
    num_pixels: usize,
    depth: usize,
    mask: Option<&[u8]>,
    values: ConstantValues,
) -> PixelData {
    let total = num_pixels * depth;
    let mut out = vec![0.0; total];
    match (depth, values) {
        (_, ConstantValues::Single(value)) => {
            if let Some(mask) = mask {
                for pixel in 0..num_pixels {
                    if mask[pixel] != 0 {
                        let base = pixel * depth;
                        out[base..base + depth].fill(value);
                    }
                }
            } else {
                out.fill(value);
            }
        }
        (depth, ConstantValues::PerDepth(values)) => {
            if let Some(mask) = mask {
                for pixel in 0..num_pixels {
                    if mask[pixel] != 0 {
                        let base = pixel * depth;
                        out[base..base + depth].copy_from_slice(&values);
                    }
                }
            } else {
                for pixel in 0..num_pixels {
                    let base = pixel * depth;
                    out[base..base + depth].copy_from_slice(&values);
                }
            }
        }
    }
    cast_pixels(data_type, out)
}

fn scatter_valid_samples(
    data_type: DataType,
    total_samples: usize,
    depth: usize,
    mask: &[u8],
    valid_samples: &[f64],
) -> Result<PixelData> {
    let mut out = vec![0.0; total_samples];
    let mut src = 0usize;
    for (pixel, &is_valid) in mask.iter().enumerate() {
        if is_valid == 0 {
            continue;
        }
        let base = pixel * depth;
        let end = base + depth;
        let src_end = src + depth;
        if src_end > valid_samples.len() {
            return Err(Error::InvalidBlob(
                "one-sweep valid sample payload is too short".into(),
            ));
        }
        out[base..end].copy_from_slice(&valid_samples[src..src_end]);
        src = src_end;
    }
    if src != valid_samples.len() {
        return Err(Error::InvalidBlob(
            "one-sweep valid sample payload contains trailing values".into(),
        ));
    }
    Ok(cast_pixels(data_type, out))
}

fn zeroed_pixels(data_type: DataType, total_samples: usize) -> PixelData {
    match data_type {
        DataType::I8 => PixelData::I8(vec![0; total_samples]),
        DataType::U8 => PixelData::U8(vec![0; total_samples]),
        DataType::I16 => PixelData::I16(vec![0; total_samples]),
        DataType::U16 => PixelData::U16(vec![0; total_samples]),
        DataType::I32 => PixelData::I32(vec![0; total_samples]),
        DataType::U32 => PixelData::U32(vec![0; total_samples]),
        DataType::F32 => PixelData::F32(vec![0.0; total_samples]),
        DataType::F64 => PixelData::F64(vec![0.0; total_samples]),
    }
}

fn cast_pixels(data_type: DataType, values: Vec<f64>) -> PixelData {
    match data_type {
        DataType::I8 => PixelData::I8(values.into_iter().map(|x| x as i8).collect()),
        DataType::U8 => PixelData::U8(values.into_iter().map(|x| x as u8).collect()),
        DataType::I16 => PixelData::I16(values.into_iter().map(|x| x as i16).collect()),
        DataType::U16 => PixelData::U16(values.into_iter().map(|x| x as u16).collect()),
        DataType::I32 => PixelData::I32(values.into_iter().map(|x| x as i32).collect()),
        DataType::U32 => PixelData::U32(values.into_iter().map(|x| x as u32).collect()),
        DataType::F32 => PixelData::F32(values.into_iter().map(|x| x as f32).collect()),
        DataType::F64 => PixelData::F64(values),
    }
}

fn scan_range(pixels: &PixelData, mask: Option<&[u8]>) -> Result<(f64, f64)> {
    let values = pixels.to_f64();
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for (index, value) in values.into_iter().enumerate() {
        if mask.map(|mask| mask[index] == 0).unwrap_or(false) {
            continue;
        }
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    if !min_value.is_finite() || !max_value.is_finite() {
        return Err(Error::InvalidBlob(
            "cannot compute a value range for an empty LERC pixel buffer".into(),
        ));
    }

    Ok((min_value, max_value))
}

fn fletcher32(bytes: &[u8]) -> u32 {
    let mut sum1 = 0xffffu32;
    let mut sum2 = 0xffffu32;
    let mut words = bytes.len() / 2;
    let mut index = 0usize;

    while words > 0 {
        let chunk = words.min(359);
        words -= chunk;
        for _ in 0..chunk {
            sum1 += (bytes[index] as u32) << 8;
            index += 1;
            sum2 += sum1 + bytes[index] as u32;
            sum1 += bytes[index] as u32;
            index += 1;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if bytes.len() & 1 != 0 {
        sum1 += (bytes[index] as u32) << 8;
        sum2 += sum1;
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum2 << 16) | (sum1 & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn build_header_v2(
        width: u32,
        height: u32,
        valid_pixel_count: u32,
        image_type: i32,
        max_z_error: f64,
        z_min: f64,
        z_max: f64,
        payload_len: usize,
    ) -> Vec<u8> {
        let blob_size = 58 + 4 + payload_len;
        let mut bytes = Vec::with_capacity(blob_size);
        bytes.extend_from_slice(MAGIC_LERC2);
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
        bytes.extend_from_slice(&image_type.to_le_bytes());
        bytes.extend_from_slice(&max_z_error.to_le_bytes());
        bytes.extend_from_slice(&z_min.to_le_bytes());
        bytes.extend_from_slice(&z_max.to_le_bytes());
        bytes
    }

    fn finalize_v4_with_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
        let blob_size = bytes.len() as i32;
        bytes[34..38].copy_from_slice(&blob_size.to_le_bytes());
        let checksum = fletcher32(&bytes[14..blob_size as usize]);
        bytes[10..14].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    fn encode_mask_rle(mask: &[u8]) -> Vec<u8> {
        let bitset_len = mask.len().div_ceil(8);
        let mut bitset = vec![0u8; bitset_len];
        for (index, &value) in mask.iter().enumerate() {
            if value != 0 {
                bitset[index >> 3] |= 1 << (7 - (index & 7));
            }
        }

        let mut encoded = Vec::with_capacity(bitset_len + 4);
        encoded.extend_from_slice(&(bitset_len as i16).to_le_bytes());
        encoded.extend_from_slice(&bitset);
        encoded.extend_from_slice(&i16::MIN.to_le_bytes());
        encoded
    }

    fn pack_msb_bits(values: &[u32], bits_per_pixel: u8) -> Vec<u8> {
        let total_bits = values.len() * usize::from(bits_per_pixel);
        let mut bytes = vec![0u8; total_bits.div_ceil(8)];
        let mut bit_offset = 0usize;
        for &value in values {
            for bit in (0..bits_per_pixel).rev() {
                if ((value >> bit) & 1) != 0 {
                    let byte_index = bit_offset / 8;
                    let bit_index = 7 - (bit_offset % 8);
                    bytes[byte_index] |= 1 << bit_index;
                }
                bit_offset += 1;
            }
        }
        bytes
    }

    fn build_lerc1_blob(
        include_mask_header: bool,
        mask: Option<&[u8]>,
        values: &[f32],
        max_z_error: f64,
        max_value: f32,
    ) -> Vec<u8> {
        let width = 2u32;
        let height = 2u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"CntZImage ");
        bytes.extend_from_slice(&11i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&max_z_error.to_le_bytes());

        if include_mask_header {
            if let Some(mask) = mask {
                let encoded_mask = encode_mask_rle(mask);
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&(encoded_mask.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
                bytes.extend_from_slice(&encoded_mask);
            } else {
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&0u32.to_le_bytes());
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
            }
        }

        let valid_count = mask
            .map(|mask| mask.iter().filter(|&&value| value != 0).count())
            .unwrap_or(values.len());
        let offset = values.iter().copied().reduce(f32::min).unwrap_or(0.0);
        let quantized: Vec<u32> = values
            .iter()
            .map(|&value| ((value - offset) / (2.0 * max_z_error as f32)).round() as u32)
            .collect();
        let max_quantized = quantized.iter().copied().max().unwrap_or(0);
        let bits_per_pixel = bits_required(max_quantized as usize).max(1);
        let payload = pack_msb_bits(&quantized, bits_per_pixel);
        let pixel_section_len = 1 + 4 + 1 + 1 + payload.len();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(pixel_section_len as u32).to_le_bytes());
        bytes.extend_from_slice(&max_value.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.push((bits_per_pixel & 63) | (2 << 6));
        bytes.push(valid_count as u8);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn reads_blob_info_for_constant_lerc2() {
        let mut blob = build_header_v2(3, 2, 6, 3, 0.0, 7.0, 7.0, 0);
        blob.extend_from_slice(&0u32.to_le_bytes());

        let info = get_blob_info(&blob).unwrap();
        assert_eq!(info.width, 3);
        assert_eq!(info.height, 2);
        assert_eq!(info.depth, 1);
        assert_eq!(info.data_type, DataType::U16);
        assert_eq!(info.blob_size, blob.len());
    }

    #[test]
    fn decodes_constant_surface() {
        let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 9.0, 9.0, 0);
        blob.extend_from_slice(&0u32.to_le_bytes());

        let decoded = decode(&blob).unwrap();
        assert_eq!(decoded.mask, None);
        assert_eq!(decoded.pixels, PixelData::U8(vec![9, 9, 9, 9]));
    }

    #[test]
    fn decodes_one_sweep_all_valid() {
        let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 1.0, 4.0, 1 + 4);
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.push(1);
        blob.extend_from_slice(&[1, 2, 3, 4]);

        let decoded = decode(&blob).unwrap();
        assert_eq!(decoded.pixels, PixelData::U8(vec![1, 2, 3, 4]));
    }

    #[test]
    fn decodes_constant_surface_with_per_depth_ranges() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_LERC2);
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&6i32.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&10.0f64.to_le_bytes());
        bytes.extend_from_slice(&20.0f64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&10.0f32.to_le_bytes());
        bytes.extend_from_slice(&20.0f32.to_le_bytes());
        bytes.extend_from_slice(&10.0f32.to_le_bytes());
        bytes.extend_from_slice(&20.0f32.to_le_bytes());

        let blob = finalize_v4_with_checksum(bytes);
        let decoded = decode(&blob).unwrap();
        assert_eq!(decoded.pixels, PixelData::F32(vec![10.0, 20.0, 10.0, 20.0]));
    }

    #[test]
    fn counts_concatenated_bands() {
        let mut blob1 = build_header_v2(1, 1, 1, 1, 0.0, 3.0, 3.0, 0);
        blob1.extend_from_slice(&0u32.to_le_bytes());
        let mut blob2 = build_header_v2(1, 1, 1, 1, 0.0, 4.0, 4.0, 0);
        blob2.extend_from_slice(&0u32.to_le_bytes());

        let mut merged = blob1;
        merged.extend_from_slice(&blob2);
        assert_eq!(get_band_count(&merged).unwrap(), 2);
    }

    #[test]
    fn reads_blob_info_for_lerc1() {
        let blob = build_lerc1_blob(true, None, &[1.0, 2.0, 3.0, 4.0], 0.5, 4.0);

        let info = get_blob_info(&blob).unwrap();
        assert_eq!(info.version, Version::Lerc1(11));
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 2);
        assert_eq!(info.depth, 1);
        assert_eq!(info.data_type, DataType::F32);
    }

    #[test]
    fn decodes_lerc1_masked_bitstuffed_block() {
        let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);

        let decoded = decode(&blob).unwrap();
        assert_eq!(decoded.mask, Some(vec![1, 0, 1, 1]));
        assert_eq!(decoded.pixels, PixelData::F32(vec![1.0, 0.0, 3.0, 4.0]));
        assert_eq!(decoded.info.z_min, 1.0);
        assert_eq!(decoded.info.z_max, 4.0);
    }

    #[test]
    fn counts_concatenated_lerc1_bands_with_shared_mask() {
        let blob1 = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);
        let blob2 = build_lerc1_blob(false, None, &[1.0, 3.0, 4.0], 0.5, 4.0);
        let mut merged = blob1;
        merged.extend_from_slice(&blob2);

        assert_eq!(get_band_count(&merged).unwrap(), 2);
    }

    #[test]
    fn decodes_lerc2_to_ndarray() {
        let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 1.0, 4.0, 1 + 4);
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.push(1);
        blob.extend_from_slice(&[1, 2, 3, 4]);

        let array = decode_ndarray::<u8>(&blob).unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array[IxDyn(&[0, 0])], 1);
        assert_eq!(array[IxDyn(&[0, 1])], 2);
        assert_eq!(array[IxDyn(&[1, 0])], 3);
        assert_eq!(array[IxDyn(&[1, 1])], 4);
    }

    #[test]
    fn decodes_multidimensional_lerc2_to_f64_ndarray() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_LERC2);
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&8i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&6i32.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&10.0f64.to_le_bytes());
        bytes.extend_from_slice(&20.0f64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&10.0f32.to_le_bytes());
        bytes.extend_from_slice(&20.0f32.to_le_bytes());
        bytes.extend_from_slice(&10.0f32.to_le_bytes());
        bytes.extend_from_slice(&20.0f32.to_le_bytes());

        let blob = finalize_v4_with_checksum(bytes);
        let array = decode_ndarray_f64(&blob).unwrap();
        assert_eq!(array.shape(), &[1, 2, 2]);
        assert_eq!(array[IxDyn(&[0, 0, 0])], 10.0);
        assert_eq!(array[IxDyn(&[0, 0, 1])], 20.0);
        assert_eq!(array[IxDyn(&[0, 1, 0])], 10.0);
        assert_eq!(array[IxDyn(&[0, 1, 1])], 20.0);
    }

    #[test]
    fn decodes_lerc1_mask_to_ndarray() {
        let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);

        let mask = decode_mask_ndarray(&blob).unwrap().unwrap();
        assert_eq!(mask.shape(), &[2, 2]);
        assert_eq!(mask[IxDyn(&[0, 0])], 1);
        assert_eq!(mask[IxDyn(&[0, 1])], 0);
        assert_eq!(mask[IxDyn(&[1, 0])], 1);
        assert_eq!(mask[IxDyn(&[1, 1])], 1);
    }
}
