use lerc_core::{BlobInfo, DataType, Error, MaskEncoding, PixelData, Result, Version};

use crate::allocation::{checked_mul, default_vec, vec_with_capacity};
use crate::bitstuff::{unstuff_v2, UnstuffOptions};
use crate::io::Cursor;
use crate::materialize::{BandWriter, PixelDataWriter};
use crate::pixel::{
    count_valid_in_block, words_from_padded, AllValid, MaskValidity, Sample, Validity,
};
use crate::{Decoded, DecodedF64};

const MAGIC_LERC1_PREFIX: &[u8; 9] = b"CntZImage";

#[derive(Debug, Clone)]
struct Lerc1PixelsHeader {
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

#[derive(Debug, Clone, Copy)]
struct BlockPos {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Copy)]
struct BlockGrid {
    width: usize,
    height: usize,
    nominal_blocks_x: usize,
    nominal_blocks_y: usize,
    actual_blocks_x: usize,
    actual_blocks_y: usize,
    base_width: usize,
    base_height: usize,
}

impl BlockGrid {
    fn new(width: usize, height: usize, nominal_blocks_x: usize, nominal_blocks_y: usize) -> Self {
        let base_width = width / nominal_blocks_x;
        let base_height = height / nominal_blocks_y;
        Self {
            width,
            height,
            nominal_blocks_x,
            nominal_blocks_y,
            actual_blocks_x: nominal_blocks_x + usize::from(width % nominal_blocks_x != 0),
            actual_blocks_y: nominal_blocks_y + usize::from(height % nominal_blocks_y != 0),
            base_width,
            base_height,
        }
    }

    fn block_count(self) -> Result<usize> {
        checked_mul(
            self.actual_blocks_x,
            self.actual_blocks_y,
            "Lerc1 block count",
        )
    }

    fn max_block_samples(self) -> Result<usize> {
        let remainder_width = self.width % self.nominal_blocks_x;
        let remainder_height = self.height % self.nominal_blocks_y;
        checked_mul(
            self.base_width.max(remainder_width),
            self.base_height.max(remainder_height),
            "Lerc1 maximum block sample count",
        )
    }

    fn for_each_block(self, mut visit: impl FnMut(BlockPos) -> Result<()>) -> Result<()> {
        for block_y in 0..self.actual_blocks_y {
            let height = if block_y + 1 == self.actual_blocks_y
                && self.height % self.nominal_blocks_y != 0
            {
                self.height % self.nominal_blocks_y
            } else {
                self.base_height
            };
            if height == 0 {
                continue;
            }

            for block_x in 0..self.actual_blocks_x {
                let width = if block_x + 1 == self.actual_blocks_x
                    && self.width % self.nominal_blocks_x != 0
                {
                    self.width % self.nominal_blocks_x
                } else {
                    self.base_width
                };
                if width == 0 {
                    continue;
                }

                visit(BlockPos {
                    x: block_x * self.base_width,
                    y: block_y * self.base_height,
                    width,
                    height,
                })?;
            }
        }
        Ok(())
    }
}

type ValueRange = Option<(f64, f64)>;

#[derive(Debug, Clone, Copy)]
enum MaskSource<'a> {
    Inline,
    External(&'a [u8]),
}

#[derive(Debug, Clone)]
pub(crate) struct Lerc1Blob {
    pub(crate) info: BlobInfo,
    pub(crate) mask: Option<Vec<u8>>,
    pixels: Lerc1PixelsHeader,
    blocks: Vec<Lerc1Block>,
    grid: BlockGrid,
}

pub(crate) fn is_lerc1(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC_LERC1_PREFIX)
}

pub(crate) fn inspect_with_mask_options(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
    compute_value_range: bool,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let mut parsed = parse(blob, shared_mask)?;
    if compute_value_range && parsed.info.valid_pixel_count != 0 {
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
    let (info, pixels, mask) = decode_owned::<f32>(blob, shared_mask)?;
    Ok(Decoded {
        info,
        pixels: PixelData::F32(pixels),
        mask,
    })
}

pub(crate) fn decode_f64(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<DecodedF64> {
    let (info, pixels, mask) = decode_owned::<f64>(blob, shared_mask)?;
    Ok(DecodedF64 { info, pixels, mask })
}

pub(crate) fn decode_into<T: Sample, W: BandWriter<T>>(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
    out: &mut W,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    let parsed = parse(blob, shared_mask)?;
    decode_parsed_into(parsed, out)
}

fn decode_owned<T: Sample>(
    blob: &[u8],
    shared_mask: Option<&[u8]>,
) -> Result<(BlobInfo, Vec<T>, Option<Vec<u8>>)> {
    let parsed = parse(blob, shared_mask)?;
    let mut pixels = default_vec(parsed.info.pixel_count()?, "Lerc1 pixel buffer")?;
    let mut writer = PixelDataWriter::new(&mut pixels, 1);
    let (info, mask) = decode_parsed_into(parsed, &mut writer)?;
    Ok((info, pixels, mask))
}

fn decode_parsed_into<T: Sample, W: BandWriter<T>>(
    mut parsed: Lerc1Blob,
    out: &mut W,
) -> Result<(BlobInfo, Option<Vec<u8>>)> {
    if parsed.info.valid_pixel_count as usize != parsed.info.pixel_count()? {
        out.fill_default();
    }
    let z_range = decode_pixels_into(&parsed, out)?;
    if parsed.info.valid_pixel_count != 0 {
        let (z_min, z_max) = z_range.ok_or_else(|| {
            Error::invalid_blob("Lerc1 decode produced pixels but not a value range")
        })?;
        parsed.info.z_min = z_min;
        parsed.info.z_max = z_max;
    }
    Ok((parsed.info, parsed.mask))
}

pub(crate) fn parse(blob: &[u8], shared_mask: Option<&[u8]>) -> Result<Lerc1Blob> {
    let Some(shared_mask) = shared_mask else {
        return parse_with_mask_source(blob, MaskSource::Inline);
    };

    let inline_error = match parse_with_mask_source(blob, MaskSource::Inline) {
        Ok(parsed) => return Ok(parsed),
        Err(error) => error,
    };
    let external_error = match parse_with_mask_source(blob, MaskSource::External(shared_mask)) {
        Ok(parsed) => return Ok(parsed),
        Err(error) => error,
    };
    Err(Error::invalid_blob(format!(
        "failed to parse Lerc1 blob with either an inline mask ({inline_error}) or the supplied shared mask ({external_error})"
    )))
}

fn parse_with_mask_source(blob: &[u8], mask_source: MaskSource<'_>) -> Result<Lerc1Blob> {
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

    if width == 0 || height == 0 {
        return Err(Error::InvalidHeader(
            "width and height must be greater than zero",
        ));
    }
    if !max_z_error.is_finite() || max_z_error < 0.0 {
        return Err(Error::InvalidHeader(
            "max_z_error must be finite and non-negative",
        ));
    }

    let (mask_encoding, mask) = match mask_source {
        MaskSource::Inline => read_mask(&mut cursor, width, height)?,
        MaskSource::External(shared_mask) => {
            validate_shared_mask(shared_mask, width, height)?;
            (MaskEncoding::External, Some(shared_mask.to_vec()))
        }
    };

    let pixels_num_blocks_y = cursor.read_u32()? as usize;
    let pixels_num_blocks_x = cursor.read_u32()? as usize;
    let pixels_num_bytes = cursor.read_u32()? as usize;
    let pixels_max_value = cursor.read_f32()?;

    if pixels_num_blocks_x == 0 || pixels_num_blocks_y == 0 {
        return Err(Error::InvalidHeader("Lerc1 block grid must be non-zero"));
    }
    if !pixels_max_value.is_finite() {
        return Err(Error::InvalidHeader("Lerc1 max pixel value must be finite"));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    if pixels_num_blocks_x > width_usize || pixels_num_blocks_y > height_usize {
        return Err(Error::InvalidHeader(
            "Lerc1 block grid must not exceed raster dimensions",
        ));
    }
    let num_pixels = width_usize
        .checked_mul(height_usize)
        .ok_or(Error::SizeOverflow("Lerc1 pixel count"))?;
    let grid = BlockGrid::new(
        width_usize,
        height_usize,
        pixels_num_blocks_x,
        pixels_num_blocks_y,
    );

    let valid_pixel_count = match mask.as_deref() {
        Some(mask) => mask.iter().map(|&value| u32::from(value != 0)).sum(),
        None => u32::try_from(num_pixels)
            .map_err(|_| Error::SizeOverflow("Lerc1 valid pixel count as u32"))?,
    };

    let pixel_payload = cursor.read_bytes(pixels_num_bytes)?;
    let eof_offset = cursor.offset();
    let mut pixel_cursor = Cursor::new(pixel_payload);
    let mut blocks = vec_with_capacity(grid.block_count()?, "Lerc1 block table")?;
    grid.for_each_block(|block_pos| {
        let block_valid_pixels = if let Some(mask) = mask.as_deref() {
            count_valid_in_block(
                mask,
                width_usize,
                block_pos.x,
                block_pos.y,
                block_pos.width,
                block_pos.height,
            )
        } else {
            checked_mul(
                block_pos.width,
                block_pos.height,
                "Lerc1 block sample count",
            )?
        };

        let header_byte = pixel_cursor.read_u8()?;
        let encoding = header_byte & 63;
        if encoding > 3 {
            return Err(Error::invalid_blob(format!(
                "invalid Lerc1 block encoding {encoding}"
            )));
        }

        let block_encoding = match encoding {
            2 => Lerc1BlockEncoding::Zero,
            3 => Lerc1BlockEncoding::Constant(
                read_offset_if_present(&mut pixel_cursor, header_byte)?.ok_or(
                    Error::invalid_blob("Lerc1 constant block is missing its offset"),
                )?,
            ),
            0 => {
                let byte_len = block_valid_pixels
                    .checked_mul(4)
                    .ok_or(Error::SizeOverflow("Lerc1 raw block byte count"))?;
                let values = pixel_cursor
                    .read_bytes(byte_len)?
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Lerc1BlockEncoding::Raw(values)
            }
            1 => {
                let offset = read_offset_if_present(&mut pixel_cursor, header_byte)?.ok_or(
                    Error::invalid_blob("Lerc1 bit-stuffed block is missing its offset"),
                )?;
                let packed_header = pixel_cursor.read_u8()?;
                let bits_per_pixel = packed_header & 63;
                let num_valid_pixels = match packed_header >> 6 {
                    0 => pixel_cursor.read_u32()? as usize,
                    1 => read_u16(pixel_cursor.read_bytes(2)?)? as usize,
                    2 => pixel_cursor.read_u8()? as usize,
                    other => {
                        return Err(Error::invalid_blob(format!(
                            "invalid Lerc1 valid pixel count type {other}"
                        )))
                    }
                };
                if num_valid_pixels != block_valid_pixels {
                    return Err(Error::invalid_blob(
                        "Lerc1 stuffed block valid count does not match its mask",
                    ));
                }
                let data_bytes = num_valid_pixels
                    .checked_mul(usize::from(bits_per_pixel))
                    .ok_or(Error::SizeOverflow("Lerc1 stuffed block bit count"))?
                    .div_ceil(8);
                let stuffed_data = words_from_padded(pixel_cursor.read_bytes(data_bytes)?)?;
                Lerc1BlockEncoding::Stuffed {
                    offset,
                    bits_per_pixel,
                    stuffed_data,
                }
            }
            _ => return Err(Error::Internal("validated Lerc1 block encoding changed")),
        };

        blocks.push(Lerc1Block {
            encoding: block_encoding,
            valid_pixel_count: block_valid_pixels,
        });
        Ok(())
    })?;

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
        remaining_band_count: 0,
        max_z_error,
        z_min: 0.0,
        z_max: pixels_max_value as f64,
        mask_encoding,
        no_data_value: None,
    };

    Ok(Lerc1Blob {
        info,
        mask,
        pixels: Lerc1PixelsHeader {
            max_value: pixels_max_value,
        },
        blocks,
        grid,
    })
}

fn validate_shared_mask(mask: &[u8], width: u32, height: u32) -> Result<()> {
    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::SizeOverflow("Lerc1 shared-mask pixel count"))?;
    if mask.len() != num_pixels {
        return Err(Error::invalid_blob(
            "shared mask length does not match the current Lerc1 blob",
        ));
    }
    Ok(())
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
        1 => Ok(read_u16(cursor.read_bytes(2)?)? as f32),
        2 => Ok(cursor.read_u8()? as f32),
        _ => Err(Error::invalid_blob(format!(
            "invalid Lerc1 block offset type {offset_type}"
        ))),
    }
}

fn read_mask(
    cursor: &mut Cursor<'_>,
    width: u32,
    height: u32,
) -> Result<(MaskEncoding, Option<Vec<u8>>)> {
    let num_blocks_y = cursor.read_u32()?;
    let num_blocks_x = cursor.read_u32()?;
    let num_bytes = cursor.read_u32()? as usize;
    let max_value = cursor.read_f32()?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::SizeOverflow("Lerc1 mask pixel count"))?;
    let bitset_len = num_pixels.div_ceil(8);

    if num_bytes > 0 {
        let bitset = crate::lerc2::decode_mask_rle(cursor.read_bytes(num_bytes)?, bitset_len)?;
        return Ok((
            MaskEncoding::Explicit,
            Some(crate::lerc2::unpack_mask_bitset(&bitset, num_pixels)?),
        ));
    }

    if num_blocks_y == 0 && num_blocks_x == 0 && max_value == 0.0 {
        return Ok((
            MaskEncoding::ImplicitAllInvalid,
            Some(default_vec(num_pixels, "Lerc1 implicit mask")?),
        ));
    }

    Ok((MaskEncoding::None, None))
}

fn decode_pixels_into<T: Sample, W: BandWriter<T>>(
    parsed: &Lerc1Blob,
    out: &mut W,
) -> Result<ValueRange> {
    match parsed.mask.as_deref() {
        Some(mask) => decode_pixels_into_with_validity(parsed, MaskValidity::new(mask), out),
        None => decode_pixels_into_with_validity(parsed, AllValid, out),
    }
}

fn decode_pixels_into_with_validity<T: Sample, W: BandWriter<T>, V: Validity>(
    parsed: &Lerc1Blob,
    validity: V,
    out: &mut W,
) -> Result<ValueRange> {
    let width = parsed.info.width as usize;
    let block_samples = parsed.grid.max_block_samples()?;
    let mut block_buffer = default_vec(block_samples, "Lerc1 base block buffer")?;
    let mut block_index = 0usize;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    parsed.grid.for_each_block(|block_pos| {
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
                    return Err(Error::invalid_blob(
                        "Lerc1 stuffed block expands beyond its output buffer",
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
                )?;
                stuffed_values = Some(&block_buffer[..block.valid_pixel_count]);
                (None, None)
            }
        };

        let mut value_index = 0usize;
        for row in 0..block_pos.height {
            let pixel_row = block_pos.y + row;
            for col in 0..block_pos.width {
                let pixel = pixel_row * width + block_pos.x + col;
                if validity.is_valid(pixel) {
                    let value = if let Some(value) = constant_value {
                        value
                    } else if let Some(values) = raw_values {
                        values.get(value_index).copied().ok_or_else(|| {
                            Error::invalid_blob("Lerc1 raw block payload ended early")
                        })?
                    } else if let Some(values) = stuffed_values {
                        values.get(value_index).copied().ok_or_else(|| {
                            Error::invalid_blob("Lerc1 stuffed block payload ended early")
                        })? as f32
                    } else {
                        return Err(Error::Internal("Lerc1 block has no value source"));
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
            return Err(Error::invalid_blob(
                "Lerc1 block payload does not match the block mask",
            ));
        }
        Ok(())
    })?;

    if min_value.is_finite() && max_value.is_finite() {
        Ok(Some((min_value, max_value)))
    } else {
        Ok(None)
    }
}

fn scan_range(parsed: &Lerc1Blob) -> Result<(f64, f64)> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let block_samples = parsed.grid.max_block_samples()?;
    let mut block_buffer = default_vec(block_samples, "Lerc1 base block buffer")?;

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
                if block.valid_pixel_count > block_buffer.len() {
                    return Err(Error::invalid_blob(
                        "Lerc1 stuffed block expands beyond its output buffer",
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
                )?;
                for &value in &block_buffer[..block.valid_pixel_count] {
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
        }
    }

    if !min_value.is_finite() || !max_value.is_finite() {
        return Err(Error::invalid_blob(
            "cannot compute a value range for an empty LERC pixel buffer",
        ));
    }

    Ok((min_value, max_value))
}

fn read_u16(bytes: &[u8]) -> Result<u16> {
    let bytes = <[u8; 2]>::try_from(bytes)
        .map_err(|_| Error::Internal("Lerc1 u16 field has the wrong width"))?;
    Ok(u16::from_le_bytes(bytes))
}
