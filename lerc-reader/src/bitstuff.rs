use lerc_core::{DataType, Error, Result};

use crate::io::Cursor;
use crate::pixel::{bits_required, read_scalar, words_from_padded};

#[derive(Clone, Copy)]
pub(crate) struct UnstuffOptions<'a> {
    pub(crate) num_pixels: usize,
    pub(crate) lut_values: Option<&'a [f64]>,
    pub(crate) offset: Option<f64>,
    pub(crate) scale: f64,
    pub(crate) max_value: f64,
}

pub(crate) fn decode_bits(
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
                UnstuffOptions {
                    num_pixels: num_elements,
                    lut_values: Some(&lut_values),
                    offset: None,
                    scale: 0.0,
                    max_value,
                },
            );
        } else {
            unstuff_v2(
                &stuffed_words,
                &mut out[..num_elements],
                lut_bits_per_element,
                UnstuffOptions {
                    num_pixels: num_elements,
                    lut_values: Some(&lut_values),
                    offset: None,
                    scale: 0.0,
                    max_value,
                },
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
            UnstuffOptions {
                num_pixels: num_elements,
                lut_values: None,
                offset: Some(offset),
                scale: 2.0 * max_z_error,
                max_value,
            },
        ),
        (true, None) => original_unstuff_v3(&stuffed_words, &mut out[..num_elements], num_bits),
        (false, Some(offset)) => unstuff_v2(
            &stuffed_words,
            &mut out[..num_elements],
            num_bits,
            UnstuffOptions {
                num_pixels: num_elements,
                lut_values: None,
                offset: Some(offset),
                scale: 2.0 * max_z_error,
                max_value,
            },
        ),
        (false, None) => original_unstuff_v2(&stuffed_words, &mut out[..num_elements], num_bits),
    }
    Ok(num_elements)
}

pub(crate) fn unstuff_v2(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    options: UnstuffOptions<'_>,
) {
    let bit_mask = if bits_per_pixel == 32 {
        u32::MAX
    } else {
        (1u32 << bits_per_pixel) - 1
    };
    let mut words = src.to_vec();
    let num_invalid_tail_bytes =
        words.len() * 4 - (usize::from(bits_per_pixel) * options.num_pixels).div_ceil(8);
    if let Some(last) = words.last_mut() {
        *last <<= 8 * num_invalid_tail_bytes;
    }

    let mut index = 0usize;
    let mut bits_left = 0usize;
    let mut buffer = 0u32;
    let nmax = options
        .offset
        .map(|offset| quantized_nmax(offset, options.scale, options.max_value));

    for item in dest.iter_mut().take(options.num_pixels) {
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
        *item = match (options.lut_values, options.offset) {
            (Some(lut_values), _) => lut_values[n as usize],
            (None, Some(offset)) => {
                quantized_value(offset, options.scale, options.max_value, n, nmax.unwrap())
            }
            (None, None) => n as f64,
        };
    }
}

pub(crate) fn unstuff_v3(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    options: UnstuffOptions<'_>,
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
    let nmax = options
        .offset
        .map(|offset| quantized_nmax(offset, options.scale, options.max_value));

    for item in dest.iter_mut().take(options.num_pixels) {
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
        *item = match (options.lut_values, options.offset) {
            (Some(lut_values), _) => lut_values[n as usize],
            (None, Some(offset)) => {
                quantized_value(offset, options.scale, options.max_value, n, nmax.unwrap())
            }
            (None, None) => n as f64,
        };
    }
}

pub(crate) fn original_unstuff_v2(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) {
    unstuff_v2(
        src,
        dest,
        bits_per_pixel,
        UnstuffOptions {
            num_pixels: dest.len(),
            lut_values: None,
            offset: None,
            scale: 0.0,
            max_value: 0.0,
        },
    );
}

pub(crate) fn original_unstuff_v3(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) {
    unstuff_v3(
        src,
        dest,
        bits_per_pixel,
        UnstuffOptions {
            num_pixels: dest.len(),
            lut_values: None,
            offset: None,
            scale: 0.0,
            max_value: 0.0,
        },
    );
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
        UnstuffOptions {
            num_pixels,
            lut_values: None,
            offset: Some(offset),
            scale,
            max_value,
        },
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
        UnstuffOptions {
            num_pixels,
            lut_values: None,
            offset: Some(offset),
            scale,
            max_value,
        },
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

pub(crate) fn data_type_used(data_type: DataType, tc: u8) -> Result<DataType> {
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

pub(crate) fn read_typed_scalar(cursor: &mut Cursor<'_>, data_type: DataType) -> Result<f64> {
    read_scalar(cursor.read_bytes(data_type.byte_len())?, data_type)
}

fn read_u16(bytes: &[u8]) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}
