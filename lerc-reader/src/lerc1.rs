use lerc_core::{BlobInfo, DataType, Decoded, DecodedF64, Error, PixelData, Result, Version};

use crate::bitstuff::{unstuff_v2, UnstuffOptions};
use crate::io::Cursor;
use crate::pixel::{count_valid_in_block, words_from_padded, Sample};
use lerc_band_materialize::BandWriter;

const MAGIC_LERC1_PREFIX: &[u8; 9] = b"CntZImage";

#[derive(Debug, Clone)]
struct Lerc1PixelsHeader {
    num_blocks_y: usize,
    num_blocks_x: usize,
    max_value: f32,
}

#[derive(Debug, Clone)]
enum Lerc1BlockEncoding {
    Zero,
    Constant(f32),
    Raw(Vec<f32>),
    Stuffed {
        offset: f32,
        bits_per_pixel: u8,
        stuffed_data: Vec<u32>,
    },
}

#[derive(Debug, Clone)]
struct Lerc1Block {
    encoding: Lerc1BlockEncoding,
    valid_pixel_count: usize,
}

type ValueRange = Option<(f64, f64)>;
type TypedPixels<T> = (Vec<T>, ValueRange);

#[derive(Debug, Clone)]
pub(crate) struct Lerc1Blob {
    pub(crate) info: BlobInfo,
    pub(crate) mask: Option<Vec<u8>>,
    pixels: Lerc1PixelsHeader,
    blocks: Vec<Lerc1Block>,
    actual_num_blocks_y: usize,
    actual_num_blocks_x: usize,
    base_block_height: usize,
    base_block_width: usize,
}

pub(crate) fn is_lerc1(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC1_PREFIX)
}

pub(crate) fn inspect(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<BlobInfo> {
    let (info, _) = inspect_with_mask(blob, shared_mask)?;
    Ok(info)
}

pub(crate) fn inspect_with_mask(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let mut parsed = parse(blob, shared_mask)?;
    if parsed.info.valid_pixel_count != 0 {
        let (z_min, z_max) = scan_range(&parsed)?;
        parsed.info.z_min = z_min;
        parsed.info.z_max = z_max;
    }
    Ok((parsed.info, parsed.mask))
}

pub(crate) fn inspect_mask(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let parsed = parse(blob, shared_mask)?;
    Ok((parsed.info, parsed.mask))
}

pub(crate) fn decode(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<Decoded> {
    let mut parsed = parse(blob, shared_mask)?;
    let (pixels, z_range) = decode_pixels::<f32>(&parsed)?;
    if parsed.info.valid_pixel_count != 0 {
        let (z_min, z_max) = z_range.ok_or_else(|| {
            Error::InvalidBlob("Lerc1 decode produced pixels but not a value range".into())
        })?;
        parsed.info.z_min = z_min;
        parsed.info.z_max = z_max;
    }
    Ok(Decoded {
        info: parsed.info,
        pixels: PixelData::F32(pixels),
        mask: parsed.mask,
    })
}

pub(crate) fn decode_f64(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<DecodedF64> {
    let mut parsed = parse(blob, shared_mask)?;
    let (pixels, z_range) = decode_pixels::<f64>(&parsed)?;
    if parsed.info.valid_pixel_count != 0 {
        let (z_min, z_max) = z_range.ok_or_else(|| {
            Error::InvalidBlob("Lerc1 decode produced pixels but not a value range".into())
        })?;
        parsed.info.z_min = z_min;
        parsed.info.z_max = z_max;
    }
    Ok(DecodedF64 {
        info: parsed.info,
        pixels,
        mask: parsed.mask,
    })
}

pub(crate) fn decode_into<T: Sample, W: BandWriter<T>>(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
    out: &mut W,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let mut parsed = parse(blob, shared_mask)?;
    if parsed.info.valid_pixel_count != parsed.info.pixel_count()? as u32 {
        out.fill_default();
    }
    let z_range = decode_pixels_into(&parsed, out)?;
    if parsed.info.valid_pixel_count != 0 {
        let (z_min, z_max) = z_range.ok_or_else(|| {
            Error::InvalidBlob("Lerc1 decode produced pixels but not a value range".into())
        })?;
        parsed.info.z_min = z_min;
        parsed.info.z_max = z_max;
    }
    Ok((parsed.info, parsed.mask))
}

pub(crate) fn parse(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<Lerc1Blob> {
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
        Some(shared_mask.to_vec())
    } else {
        read_mask(&mut cursor, width, height)?
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
        let this_block_height = block_height(
            block_y,
            actual_num_blocks_y,
            height_usize,
            pixels_num_blocks_y,
            base_block_height,
        );
        if this_block_height == 0 {
            continue;
        }
        for block_x in 0..actual_num_blocks_x {
            let this_block_width = block_width(
                block_x,
                actual_num_blocks_x,
                width_usize,
                pixels_num_blocks_x,
                base_block_width,
            );
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

            let block_encoding = match encoding {
                2 => Lerc1BlockEncoding::Zero,
                3 => Lerc1BlockEncoding::Constant(
                    read_offset_if_present(&mut cursor, header_byte)?.ok_or(Error::InvalidBlob(
                        "Lerc1 constant block is missing its offset".into(),
                    ))?,
                ),
                0 => {
                    let byte_len = block_valid_pixels.checked_mul(4).ok_or_else(|| {
                        Error::InvalidBlob("Lerc1 raw block byte count overflows usize".into())
                    })?;
                    let values = cursor
                        .read_bytes(byte_len)?
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Lerc1BlockEncoding::Raw(values)
                }
                1 => {
                    let offset = read_offset_if_present(&mut cursor, header_byte)?.ok_or(
                        Error::InvalidBlob("Lerc1 bit-stuffed block is missing its offset".into()),
                    )?;
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
                    if num_valid_pixels != block_valid_pixels {
                        return Err(Error::InvalidBlob(
                            "Lerc1 stuffed block valid count does not match its mask".into(),
                        ));
                    }
                    let data_bytes = (num_valid_pixels * usize::from(bits_per_pixel)).div_ceil(8);
                    let stuffed_data = words_from_padded(cursor.read_bytes(data_bytes)?);
                    Lerc1BlockEncoding::Stuffed {
                        offset,
                        bits_per_pixel,
                        stuffed_data,
                    }
                }
                _ => unreachable!(),
            };

            blocks.push(Lerc1Block {
                encoding: block_encoding,
                valid_pixel_count: block_valid_pixels,
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
    })
}

fn map_lerc1_data_type(_image_type: i32) -> DataType {
    DataType::F32
}

fn read_offset_if_present(cursor: &mut Cursor<'_>, header_byte: u8) -> Result<Option<f32>> {
    if header_byte == 0 || header_byte == 2 {
        return Ok(None);
    }
    Ok(Some(read_offset(cursor, header_byte >> 6)?))
}

fn read_offset(cursor: &mut Cursor<'_>, offset_type: u8) -> Result<f32> {
    match offset_type {
        0 => cursor.read_f32(),
        1 => Ok(cursor.read_i16()? as f32),
        2 => Ok(i8::from_le_bytes([cursor.read_u8()?]) as f32),
        _ => Err(Error::InvalidBlob(format!(
            "invalid Lerc1 block offset type {offset_type}"
        ))),
    }
}

fn read_mask(cursor: &mut Cursor<'_>, width: u32, height: u32) -> Result<Option<Vec<u8>>> {
    let num_blocks_y = cursor.read_u32()?;
    let num_blocks_x = cursor.read_u32()?;
    let num_bytes = cursor.read_u32()? as usize;
    let max_value = cursor.read_f32()?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))?;
    let bitset_len = num_pixels.div_ceil(8);

    if num_bytes > 0 {
        let bitset = crate::lerc2::decode_mask_rle(cursor.read_bytes(num_bytes)?, bitset_len)?;
        return Ok(Some(crate::lerc2::unpack_mask_bitset(&bitset, num_pixels)));
    }

    if num_blocks_y == 0 && num_blocks_x == 0 && max_value == 0.0 {
        return Ok(Some(vec![0; num_pixels]));
    }

    Ok(None)
}

fn decode_pixels<T: Sample>(parsed: &Lerc1Blob) -> Result<TypedPixels<T>> {
    let width = parsed.info.width as usize;
    let height = parsed.info.height as usize;
    let mask = parsed.mask.as_deref();
    let mut result = vec![T::default(); width * height];
    let mut block_buffer = vec![0.0f64; parsed.base_block_width * parsed.base_block_height];
    let mut block_index = 0usize;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for block_y in 0..parsed.actual_num_blocks_y {
        let this_block_height = block_height(
            block_y,
            parsed.actual_num_blocks_y,
            height,
            parsed.pixels.num_blocks_y,
            parsed.base_block_height,
        );
        if this_block_height == 0 {
            continue;
        }

        for block_x in 0..parsed.actual_num_blocks_x {
            let this_block_width = block_width(
                block_x,
                parsed.actual_num_blocks_x,
                width,
                parsed.pixels.num_blocks_x,
                parsed.base_block_width,
            );
            if this_block_width == 0 {
                continue;
            }

            let block = &parsed.blocks[block_index];
            block_index += 1;

            let mut stuffed_values: Option<&[f64]> = None;
            let (raw_values, constant_value) = match &block.encoding {
                Lerc1BlockEncoding::Zero => (None, Some(0.0f32)),
                Lerc1BlockEncoding::Constant(value) => (None, Some(*value)),
                Lerc1BlockEncoding::Raw(values) => (Some(values.as_slice()), None),
                Lerc1BlockEncoding::Stuffed {
                    offset,
                    bits_per_pixel,
                    stuffed_data,
                } => {
                    if block.valid_pixel_count > block_buffer.len() {
                        return Err(Error::InvalidBlob(
                            "Lerc1 stuffed block expands beyond its output buffer".into(),
                        ));
                    }
                    block_buffer[..block.valid_pixel_count].fill(0.0);
                    unstuff_v2(
                        stuffed_data,
                        &mut block_buffer[..block.valid_pixel_count],
                        *bits_per_pixel,
                        UnstuffOptions {
                            num_pixels: block.valid_pixel_count,
                            lut_values: None,
                            offset: Some(*offset as f64),
                            scale: 2.0 * parsed.info.max_z_error,
                            max_value: parsed.pixels.max_value as f64,
                        },
                    );
                    stuffed_values = Some(&block_buffer[..block.valid_pixel_count]);
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
                            values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 raw block payload ended early".into())
                            })?
                        } else if let Some(values) = stuffed_values {
                            values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 stuffed block payload ended early".into())
                            })? as f32
                        } else {
                            unreachable!()
                        };
                        let value_f64 = f64::from(value);
                        result[pixel] = T::from_f64(value_f64);
                        min_value = min_value.min(value_f64);
                        max_value = max_value.max(value_f64);
                        value_index += 1;
                    }
                }
            }

            if block.valid_pixel_count != value_index
                && !matches!(
                    block.encoding,
                    Lerc1BlockEncoding::Zero | Lerc1BlockEncoding::Constant(_)
                )
            {
                return Err(Error::InvalidBlob(
                    "Lerc1 block payload does not match the block mask".into(),
                ));
            }
        }
    }

    let z_range = if min_value.is_finite() && max_value.is_finite() {
        Some((min_value, max_value))
    } else {
        None
    };

    Ok((result, z_range))
}

fn decode_pixels_into<T: Sample, W: BandWriter<T>>(
    parsed: &Lerc1Blob,
    out: &mut W,
) -> Result<ValueRange> {
    let width = parsed.info.width as usize;
    let height = parsed.info.height as usize;
    let mask = parsed.mask.as_deref();
    let mut block_buffer = vec![0.0f64; parsed.base_block_width * parsed.base_block_height];
    let mut block_index = 0usize;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for block_y in 0..parsed.actual_num_blocks_y {
        let this_block_height = block_height(
            block_y,
            parsed.actual_num_blocks_y,
            height,
            parsed.pixels.num_blocks_y,
            parsed.base_block_height,
        );
        if this_block_height == 0 {
            continue;
        }

        for block_x in 0..parsed.actual_num_blocks_x {
            let this_block_width = block_width(
                block_x,
                parsed.actual_num_blocks_x,
                width,
                parsed.pixels.num_blocks_x,
                parsed.base_block_width,
            );
            if this_block_width == 0 {
                continue;
            }

            let block = &parsed.blocks[block_index];
            block_index += 1;

            let mut stuffed_values: Option<&[f64]> = None;
            let (raw_values, constant_value) = match &block.encoding {
                Lerc1BlockEncoding::Zero => (None, Some(0.0f32)),
                Lerc1BlockEncoding::Constant(value) => (None, Some(*value)),
                Lerc1BlockEncoding::Raw(values) => (Some(values.as_slice()), None),
                Lerc1BlockEncoding::Stuffed {
                    offset,
                    bits_per_pixel,
                    stuffed_data,
                } => {
                    if block.valid_pixel_count > block_buffer.len() {
                        return Err(Error::InvalidBlob(
                            "Lerc1 stuffed block expands beyond its output buffer".into(),
                        ));
                    }
                    block_buffer[..block.valid_pixel_count].fill(0.0);
                    unstuff_v2(
                        stuffed_data,
                        &mut block_buffer[..block.valid_pixel_count],
                        *bits_per_pixel,
                        UnstuffOptions {
                            num_pixels: block.valid_pixel_count,
                            lut_values: None,
                            offset: Some(*offset as f64),
                            scale: 2.0 * parsed.info.max_z_error,
                            max_value: parsed.pixels.max_value as f64,
                        },
                    );
                    stuffed_values = Some(&block_buffer[..block.valid_pixel_count]);
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
                            values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 raw block payload ended early".into())
                            })?
                        } else if let Some(values) = stuffed_values {
                            values.get(value_index).copied().ok_or_else(|| {
                                Error::InvalidBlob("Lerc1 stuffed block payload ended early".into())
                            })? as f32
                        } else {
                            unreachable!()
                        };
                        let value_f64 = f64::from(value);
                        out.write(pixel, 0, T::from_f64(value_f64));
                        min_value = min_value.min(value_f64);
                        max_value = max_value.max(value_f64);
                        value_index += 1;
                    }
                }
            }

            if block.valid_pixel_count != value_index
                && !matches!(
                    block.encoding,
                    Lerc1BlockEncoding::Zero | Lerc1BlockEncoding::Constant(_)
                )
            {
                return Err(Error::InvalidBlob(
                    "Lerc1 block payload does not match the block mask".into(),
                ));
            }
        }
    }

    if min_value.is_finite() && max_value.is_finite() {
        Ok(Some((min_value, max_value)))
    } else {
        Ok(None)
    }
}

fn scan_range(parsed: &Lerc1Blob) -> Result<(f64, f64)> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut block_buffer = vec![0.0f64; parsed.base_block_width * parsed.base_block_height];

    for block in &parsed.blocks {
        if block.valid_pixel_count == 0 {
            continue;
        }

        match &block.encoding {
            Lerc1BlockEncoding::Zero => {
                min_value = min_value.min(0.0);
                max_value = max_value.max(0.0);
            }
            Lerc1BlockEncoding::Constant(value) => {
                let value = f64::from(*value);
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
            Lerc1BlockEncoding::Raw(values) => {
                for &value in values {
                    let value = f64::from(value);
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
            Lerc1BlockEncoding::Stuffed {
                offset,
                bits_per_pixel,
                stuffed_data,
            } => {
                block_buffer[..block.valid_pixel_count].fill(0.0);
                unstuff_v2(
                    stuffed_data,
                    &mut block_buffer[..block.valid_pixel_count],
                    *bits_per_pixel,
                    UnstuffOptions {
                        num_pixels: block.valid_pixel_count,
                        lut_values: None,
                        offset: Some(*offset as f64),
                        scale: 2.0 * parsed.info.max_z_error,
                        max_value: parsed.pixels.max_value as f64,
                    },
                );
                for &value in &block_buffer[..block.valid_pixel_count] {
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
        }
    }

    if !min_value.is_finite() || !max_value.is_finite() {
        return Err(Error::InvalidBlob(
            "cannot compute a value range for an empty LERC pixel buffer".into(),
        ));
    }

    Ok((min_value, max_value))
}

fn block_width(
    block_x: usize,
    actual_num_blocks_x: usize,
    width: usize,
    num_blocks_x: usize,
    base_block_width: usize,
) -> usize {
    if block_x + 1 == actual_num_blocks_x && width % num_blocks_x != 0 {
        width % num_blocks_x
    } else {
        base_block_width
    }
}

fn block_height(
    block_y: usize,
    actual_num_blocks_y: usize,
    height: usize,
    num_blocks_y: usize,
    base_block_height: usize,
) -> usize {
    if block_y + 1 == actual_num_blocks_y && height % num_blocks_y != 0 {
        height % num_blocks_y
    } else {
        base_block_height
    }
}

fn read_u16(bytes: &[u8]) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}
