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
        _ => return Err(Error::invalid_blob("invalid valid pixel count field width")),
    };
    if num_elements > out.len() {
        return Err(Error::invalid_blob(
            "bit-stuffed block expands beyond its output buffer",
        ));
    }

    if do_lut {
        let offset = offset.ok_or(Error::invalid_blob(
            "LUT-compressed block is missing its base offset",
        ))?;
        let lut_bytes = cursor.read_u8()? as usize;
        if lut_bytes == 0 {
            return Err(Error::invalid_blob(
                "LUT-compressed block has zero LUT bytes",
            ));
        }
        let lut_data_bytes = packed_byte_len(num_bits, lut_bytes - 1)?;
        let lut_words = words_from_padded(cursor.read_bytes(lut_data_bytes)?)?;
        let lut_bits_per_element = bits_required(lut_bytes - 1);
        let stuffed_data_bytes = packed_byte_len(lut_bits_per_element, num_elements)?;
        let stuffed_words = words_from_padded(cursor.read_bytes(stuffed_data_bytes)?)?;

        let lut_values = if version >= 3 {
            unstuff_lut_v3(
                &lut_words,
                num_bits,
                lut_bytes - 1,
                offset,
                2.0 * max_z_error,
                max_value,
            )?
        } else {
            unstuff_lut_v2(
                &lut_words,
                num_bits,
                lut_bytes - 1,
                offset,
                2.0 * max_z_error,
                max_value,
            )?
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
            )?;
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
            )?;
        }
        return Ok(num_elements);
    }

    if num_bits == 0 {
        out[..num_elements].fill(offset.unwrap_or(0.0));
        return Ok(num_elements);
    }

    let stuffed_data_bytes = packed_byte_len(num_bits, num_elements)?;
    let stuffed_words = words_from_padded(cursor.read_bytes(stuffed_data_bytes)?)?;
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
    }?;
    Ok(num_elements)
}

pub(crate) fn unstuff_v2(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    options: UnstuffOptions<'_>,
) -> Result<()> {
    let bit_mask = bit_mask(bits_per_pixel)?;
    let expected_bytes = packed_byte_len(bits_per_pixel, options.num_pixels)?;
    let available_bytes = word_bytes(src)?;
    if expected_bytes > available_bytes {
        return Err(Error::Truncated {
            offset: available_bytes,
            needed: expected_bytes - available_bytes,
            available: 0,
        });
    }

    let num_invalid_tail_bytes = available_bytes - expected_bytes;
    if num_invalid_tail_bytes >= 4 {
        return Err(Error::invalid_blob(
            "bit-stuffed payload has extra padded words",
        ));
    }
    let tail_shift = 8 * num_invalid_tail_bytes;

    let mut index = 0usize;
    let mut bits_left = 0usize;
    let mut buffer = 0u32;
    let nmax = options
        .offset
        .map(|offset| quantized_nmax(offset, options.scale, options.max_value));

    for item in dest.iter_mut().take(options.num_pixels) {
        let n = if bits_per_pixel == 0 {
            0
        } else {
            if bits_left == 0 {
                buffer = read_v2_word(src, &mut index, tail_shift)?;
                bits_left = 32;
            }
            if bits_left >= bits_per_pixel as usize {
                let n = (buffer >> (bits_left - bits_per_pixel as usize)) & bit_mask;
                bits_left -= bits_per_pixel as usize;
                n
            } else {
                let missing_bits = bits_per_pixel as usize - bits_left;
                let mut n = ((buffer & bit_mask) << missing_bits) & bit_mask;
                buffer = read_v2_word(src, &mut index, tail_shift)?;
                bits_left = 32 - missing_bits;
                n += buffer >> bits_left;
                n
            }
        };
        *item = unstuffed_value(n, &options, nmax)?;
    }
    Ok(())
}

pub(crate) fn unstuff_v3(
    src: &[u32],
    dest: &mut [f64],
    bits_per_pixel: u8,
    options: UnstuffOptions<'_>,
) -> Result<()> {
    let bit_mask = bit_mask(bits_per_pixel)?;
    let expected_bytes = packed_byte_len(bits_per_pixel, options.num_pixels)?;
    let available_bytes = word_bytes(src)?;
    if expected_bytes > available_bytes {
        return Err(Error::Truncated {
            offset: available_bytes,
            needed: expected_bytes - available_bytes,
            available: 0,
        });
    }

    let mut index = 0usize;
    let mut bits_left = 0usize;
    let mut bit_pos = 0usize;
    let mut buffer = 0u32;
    let nmax = options
        .offset
        .map(|offset| quantized_nmax(offset, options.scale, options.max_value));

    for item in dest.iter_mut().take(options.num_pixels) {
        let n = if bits_per_pixel == 0 {
            0
        } else {
            if bits_left == 0 {
                buffer = read_word(src, &mut index)?;
                bits_left = 32;
                bit_pos = 0;
            }
            if bits_left >= bits_per_pixel as usize {
                let n = (buffer >> bit_pos) & bit_mask;
                bits_left -= bits_per_pixel as usize;
                bit_pos += bits_per_pixel as usize;
                n
            } else {
                let missing_bits = bits_per_pixel as usize - bits_left;
                let mut n = (buffer >> bit_pos) & bit_mask;
                buffer = read_word(src, &mut index)?;
                bits_left = 32 - missing_bits;
                n |= (buffer & ((1u32 << missing_bits) - 1))
                    << (bits_per_pixel as usize - missing_bits);
                bit_pos = missing_bits;
                n
            }
        };
        *item = unstuffed_value(n, &options, nmax)?;
    }
    Ok(())
}

pub(crate) fn original_unstuff_v2(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) -> Result<()> {
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
    )
}

pub(crate) fn original_unstuff_v3(src: &[u32], dest: &mut [f64], bits_per_pixel: u8) -> Result<()> {
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
    )
}

fn unstuff_lut_v2(
    src: &[u32],
    bits_per_pixel: u8,
    num_pixels: usize,
    offset: f64,
    scale: f64,
    max_value: f64,
) -> Result<Vec<f64>> {
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
    )?;
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(offset);
    out.extend(values);
    Ok(out)
}

fn unstuff_lut_v3(
    src: &[u32],
    bits_per_pixel: u8,
    num_pixels: usize,
    offset: f64,
    scale: f64,
    max_value: f64,
) -> Result<Vec<f64>> {
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
    )?;
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(offset);
    out.extend(values);
    Ok(out)
}

fn bit_mask(bits_per_pixel: u8) -> Result<u32> {
    match bits_per_pixel {
        0 => Ok(0),
        1..=31 => Ok((1u32 << bits_per_pixel) - 1),
        32 => Ok(u32::MAX),
        _ => Err(Error::invalid_blob(
            "bit-stuffed block uses more than 32 bits per pixel",
        )),
    }
}

fn packed_byte_len(bits_per_pixel: u8, num_pixels: usize) -> Result<usize> {
    usize::from(bits_per_pixel)
        .checked_mul(num_pixels)
        .map(|bits| bits.div_ceil(8))
        .ok_or(Error::SizeOverflow("bit-stuffed payload bit count"))
}

fn word_bytes(words: &[u32]) -> Result<usize> {
    words
        .len()
        .checked_mul(4)
        .ok_or(Error::SizeOverflow("bit-stuffed word byte count"))
}

fn read_word(words: &[u32], index: &mut usize) -> Result<u32> {
    let word = words.get(*index).copied().ok_or_else(|| {
        let offset = (*index).saturating_mul(4);
        let total = words.len().saturating_mul(4);
        Error::Truncated {
            offset,
            needed: 4,
            available: total.saturating_sub(offset),
        }
    })?;
    *index += 1;
    Ok(word)
}

fn read_v2_word(words: &[u32], index: &mut usize, tail_shift: usize) -> Result<u32> {
    let word_index = *index;
    let word = read_word(words, index)?;
    Ok(if word_index + 1 == words.len() {
        word << tail_shift
    } else {
        word
    })
}

fn unstuffed_value(n: u32, options: &UnstuffOptions<'_>, nmax: Option<f64>) -> Result<f64> {
    match (options.lut_values, options.offset) {
        (Some(lut_values), _) => lut_values
            .get(n as usize)
            .copied()
            .ok_or_else(|| Error::invalid_blob("bit-stuffed LUT index is outside the LUT")),
        (None, Some(offset)) => {
            let nmax = nmax.ok_or(Error::Internal(
                "bit-stuffed quantization maximum is missing",
            ))?;
            Ok(quantized_value(
                offset,
                options.scale,
                options.max_value,
                n,
                nmax,
            ))
        }
        (None, None) => Ok(n as f64),
    }
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
    let bytes = <[u8; 2]>::try_from(bytes)
        .map_err(|_| Error::Internal("bit-stuffed u16 field has the wrong width"))?;
    Ok(u16::from_le_bytes(bytes))
}
