use lerc_core::{bits_required, DataType, Error, Result};

/// Largest quantized block range that is bit-stuffed rather than stored raw,
/// matching libLerc's limit for each data type.
pub(super) fn max_value_to_quantize(data_type: DataType) -> f64 {
    match data_type {
        DataType::I8 | DataType::U8 | DataType::I16 | DataType::U16 => ((1u32 << 15) - 1) as f64,
        DataType::I32 | DataType::U32 | DataType::F32 | DataType::F64 => ((1u32 << 30) - 1) as f64,
    }
}

/// Quantizes block values against `offset` (the block minimum).
///
/// This mirrors libLerc's arithmetic: each value maps to
/// `trunc((value - offset) / (2 * max_z_error) + 0.5)`, and a decoder rebuilds
/// it as `offset + q * 2 * max_z_error`. Returns `false` when a reconstructed
/// value would leave the requested error bound, which can only happen when the
/// sample magnitude exhausts `f64` precision; callers then store the block raw.
pub(super) fn quantize_block(
    values: impl IntoIterator<Item = f64>,
    offset: f64,
    max_z_error: f64,
    quantized: &mut Vec<u32>,
) -> bool {
    let scale = 1.0 / (2.0 * max_z_error);
    let step = 2.0 * max_z_error;
    let tolerance = max_z_error + max_z_error.abs() * 1e-12 + 1e-12;
    quantized.clear();
    for value in values {
        let quantized_value = ((value - offset) * scale + 0.5) as u32;
        if (offset + f64::from(quantized_value) * step - value).abs() > tolerance {
            return false;
        }
        quantized.push(quantized_value);
    }
    true
}

/// Number of quantization levels a block spans, rounded like the reference
/// encoder. Zero means every value lies within `max_z_error` of the block
/// minimum, so the block can be stored as a constant.
pub(super) fn quantized_block_max(min: f64, max: f64, max_z_error: f64) -> f64 {
    (max - min) * (1.0 / (2.0 * max_z_error))
}

/// Encoded size of a bit-stuffed payload in simple mode.
pub(super) fn simple_payload_len(count: usize, max_quantized: u32) -> Result<usize> {
    let (_, count_bytes) = count_field(count)?;
    let bits = bits_required(max_quantized as usize) as usize;
    count
        .checked_mul(bits)
        .map(|bits| 1 + count_bytes + bits.div_ceil(8))
        .ok_or(Error::SizeOverflow("bit-stuffed payload bit count"))
}

/// Encoded size of a bit-stuffed payload in lookup-table mode, or `None` when
/// the block has too many distinct values for a Lerc2 lookup table.
///
/// `sorted` must hold the block's quantized values in ascending order.
pub(super) fn lut_payload_len(sorted: &[u32]) -> Result<Option<usize>> {
    let Some(&max_quantized) = sorted.last() else {
        return Ok(None);
    };
    let lut_len = sorted.windows(2).filter(|pair| pair[0] != pair[1]).count();
    if lut_len == 0 || lut_len >= MAX_LUT_LEN {
        return Ok(None);
    }
    let (_, count_bytes) = count_field(sorted.len())?;
    let value_bits = bits_required(max_quantized as usize) as usize;
    let index_bits = bits_required(lut_len) as usize;
    let index_bytes = sorted
        .len()
        .checked_mul(index_bits)
        .ok_or(Error::SizeOverflow("lookup-table index bit count"))?
        .div_ceil(8);
    Ok(Some(
        1 + count_bytes + 1 + (lut_len * value_bits).div_ceil(8) + index_bytes,
    ))
}

/// Writes a simple-mode bit-stuffed payload for already quantized values.
pub(super) fn encode_simple_payload(quantized: &[u32], payload: &mut Vec<u8>) -> Result<()> {
    let max_quantized = quantized.iter().copied().max().unwrap_or(0);
    let bits = bits_required(max_quantized as usize);
    if bits == 0 || bits > 31 {
        return Err(Error::Internal("bit-stuffed tile has an invalid bit width"));
    }
    let (count_code, count_bytes) = count_field(quantized.len())?;
    payload.clear();
    payload.reserve(1 + count_bytes + (quantized.len() * bits as usize).div_ceil(8));
    payload.push((count_code << 6) | bits);
    append_count(payload, quantized.len(), count_bytes)?;
    pack_lsb_bits_into(quantized, bits, payload)
}

/// Writes a lookup-table bit-stuffed payload: the distinct non-zero quantized
/// values in ascending order, followed by one table index per value.
///
/// `sorted` is scratch space; it is overwritten with the sorted values.
pub(super) fn encode_lut_payload(
    quantized: &[u32],
    sorted: &mut Vec<u32>,
    indexes: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<()> {
    sorted.clear();
    sorted.extend_from_slice(quantized);
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.first() != Some(&0) {
        return Err(Error::Internal(
            "lookup-table tile does not start at its offset",
        ));
    }
    let lut = &sorted[1..];
    let lut_len = lut.len();
    if lut_len == 0 || lut_len >= MAX_LUT_LEN {
        return Err(Error::Internal(
            "lookup-table tile has an invalid table length",
        ));
    }
    let value_bits = bits_required(lut[lut_len - 1] as usize);
    let index_bits = bits_required(lut_len);
    indexes.clear();
    for &value in quantized {
        let index = sorted
            .binary_search(&value)
            .map_err(|_| Error::Internal("lookup-table value is missing"))?;
        indexes.push(index as u32);
    }
    let (count_code, count_bytes) = count_field(quantized.len())?;
    payload.clear();
    payload.push((count_code << 6) | (1 << 5) | value_bits);
    append_count(payload, quantized.len(), count_bytes)?;
    payload.push((lut_len + 1) as u8);
    pack_lsb_bits_into(lut, value_bits, payload)?;
    pack_lsb_bits_into(indexes, index_bits, payload)
}

/// Lerc2 stores the table length, including the implicit zero, in one byte;
/// libLerc caps tables below 255 entries.
const MAX_LUT_LEN: usize = 255;

pub(super) fn write_raw_block(out: &mut Vec<u8>, values: &[u32]) -> Result<()> {
    let max_value = values.iter().copied().max().unwrap_or(0);
    let bits = bits_required(max_value as usize);
    if bits > 31 {
        return Err(Error::Internal(
            "bit-stuffed payload exceeds the Lerc2 bit width limit",
        ));
    }
    let (count_code, count_bytes) = count_field(values.len())?;
    out.push((count_code << 6) | bits);
    append_count(out, values.len(), count_bytes)?;
    if bits != 0 {
        pack_lsb_bits_into(values, bits, out)?;
    }
    Ok(())
}

fn count_field(count: usize) -> Result<(u8, usize)> {
    if count <= u8::MAX as usize {
        Ok((2, 1))
    } else if count <= u16::MAX as usize {
        Ok((1, 2))
    } else if count <= u32::MAX as usize {
        Ok((0, 4))
    } else {
        Err(Error::SizeOverflow("tile valid-value count as u32"))
    }
}

fn append_count(out: &mut Vec<u8>, count: usize, count_bytes: usize) -> Result<()> {
    match count_bytes {
        1 => out.push(
            u8::try_from(count)
                .map_err(|_| Error::Internal("count does not fit planned u8 field"))?,
        ),
        2 => out.extend_from_slice(
            &u16::try_from(count)
                .map_err(|_| Error::Internal("count does not fit planned u16 field"))?
                .to_le_bytes(),
        ),
        4 => out.extend_from_slice(
            &u32::try_from(count)
                .map_err(|_| Error::Internal("count does not fit planned u32 field"))?
                .to_le_bytes(),
        ),
        _ => return Err(Error::Internal("unsupported count field width")),
    }
    Ok(())
}

fn pack_lsb_bits_into(values: &[u32], bits_per_value: u8, out: &mut Vec<u8>) -> Result<()> {
    let total_bits = values
        .len()
        .checked_mul(bits_per_value as usize)
        .ok_or(Error::SizeOverflow("bit-stuffed payload bit count"))?;
    let byte_len = total_bits.div_ceil(8);
    let initial_len = out.len();
    out.reserve(byte_len);
    let mut accumulator = 0u64;
    let mut bits_in_accumulator = 0u8;
    for &value in values {
        accumulator |= u64::from(value) << bits_in_accumulator;
        bits_in_accumulator += bits_per_value;
        while bits_in_accumulator >= 8 {
            out.push(accumulator as u8);
            accumulator >>= 8;
            bits_in_accumulator -= 8;
        }
    }
    if bits_in_accumulator != 0 {
        out.push(accumulator as u8);
    }
    if out.len() - initial_len != byte_len {
        return Err(Error::Internal(
            "bit-stuffed payload length disagrees with its bit count",
        ));
    }
    Ok(())
}

pub(super) fn reduce_data_type(value: f64, data_type: DataType) -> Result<(u8, DataType)> {
    let reduced = match data_type {
        DataType::I8 | DataType::U8 => (0, data_type),
        DataType::I16 => {
            if fits_i8(value) {
                (2, DataType::I8)
            } else if fits_u8(value) {
                (1, DataType::U8)
            } else {
                (0, DataType::I16)
            }
        }
        DataType::U16 => {
            if fits_u8(value) {
                (1, DataType::U8)
            } else {
                (0, DataType::U16)
            }
        }
        DataType::I32 => {
            if fits_u8(value) {
                (3, DataType::U8)
            } else if fits_i16(value) {
                (2, DataType::I16)
            } else if fits_u16(value) {
                (1, DataType::U16)
            } else {
                (0, DataType::I32)
            }
        }
        DataType::U32 => {
            if fits_u8(value) {
                (2, DataType::U8)
            } else if fits_u16(value) {
                (1, DataType::U16)
            } else {
                (0, DataType::U32)
            }
        }
        DataType::F32 => {
            if fits_u8(value) {
                (2, DataType::U8)
            } else if fits_i16(value) {
                (1, DataType::I16)
            } else {
                (0, DataType::F32)
            }
        }
        DataType::F64 => {
            if fits_i16(value) {
                (3, DataType::I16)
            } else if fits_i32(value) {
                (2, DataType::I32)
            } else if fits_f32(value) {
                (1, DataType::F32)
            } else {
                (0, DataType::F64)
            }
        }
    };
    Ok(reduced)
}

fn fits_i8(value: f64) -> bool {
    (i8::MIN as f64..=i8::MAX as f64).contains(&value) && (value as i8) as f64 == value
}

fn fits_u8(value: f64) -> bool {
    (u8::MIN as f64..=u8::MAX as f64).contains(&value) && (value as u8) as f64 == value
}

fn fits_i16(value: f64) -> bool {
    (i16::MIN as f64..=i16::MAX as f64).contains(&value) && (value as i16) as f64 == value
}

fn fits_u16(value: f64) -> bool {
    (u16::MIN as f64..=u16::MAX as f64).contains(&value) && (value as u16) as f64 == value
}

fn fits_i32(value: f64) -> bool {
    (i32::MIN as f64..=i32::MAX as f64).contains(&value) && (value as i32) as f64 == value
}

fn fits_f32(value: f64) -> bool {
    (value as f32) as f64 == value
}
