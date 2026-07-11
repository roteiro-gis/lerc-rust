use lerc_core::{bits_required, DataType, Error, Result};

#[derive(Debug, Clone)]
pub(super) struct BitstuffTile {
    pub(super) offset: f64,
    pub(super) offset_type: DataType,
    pub(super) type_code: u8,
    pub(super) payload_len: usize,
}

pub(super) fn try_tile(
    values: &[f64],
    offset: f64,
    max_value: f64,
    max_z_error: f64,
    base_type: DataType,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<Option<BitstuffTile>> {
    if max_z_error <= 0.0 {
        return Ok(None);
    }
    let scale = 2.0 * max_z_error;
    let nmax_f = ((max_value - offset) / scale).ceil();
    if !nmax_f.is_finite() || !(0.0..=(u32::MAX as f64)).contains(&nmax_f) {
        return Ok(None);
    }
    let nmax = nmax_f as u32;
    if nmax == 0 {
        return Ok(None);
    }

    let epsilon = max_z_error.abs() * 1e-12 + 1e-12;
    quantized.clear();
    quantized.reserve(values.len());
    let mut max_quantized = 0u32;
    for &value in values {
        let quantized_value = ((value - offset) / scale).round().clamp(0.0, nmax as f64) as u32;
        let reconstructed = if quantized_value < nmax {
            offset + quantized_value as f64 * scale
        } else {
            max_value
        };
        if (reconstructed - value).abs() > max_z_error + epsilon {
            return Ok(None);
        }
        max_quantized = max_quantized.max(quantized_value);
        quantized.push(quantized_value);
    }

    let bits = bits_required(max_quantized as usize);
    if bits == 0 || bits > 31 {
        return Ok(None);
    }

    let (count_code, count_bytes) = count_field(values.len())?;
    let (type_code, offset_type) = reduce_data_type(offset, base_type)?;
    payload.clear();
    let bit_bytes = values
        .len()
        .checked_mul(bits as usize)
        .ok_or(Error::SizeOverflow("bit-stuffed payload bit count"))?
        .div_ceil(8);
    payload.reserve(1 + count_bytes + bit_bytes);
    payload.push((count_code << 6) | bits);
    append_count(payload, values.len(), count_bytes)?;
    pack_lsb_bits_into(quantized, bits, payload)?;
    Ok(Some(BitstuffTile {
        offset,
        offset_type,
        type_code,
        payload_len: payload.len(),
    }))
}

pub(super) fn encode_cached_payload(
    values: impl IntoIterator<Item = f64>,
    value_count: usize,
    offset: f64,
    max_z_error: f64,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<()> {
    if max_z_error <= 0.0 {
        return Err(Error::Internal(
            "cached bit-stuffed tile requires positive max_z_error",
        ));
    }

    let scale = 2.0 * max_z_error;
    quantized.clear();
    quantized.reserve(value_count);
    let mut max_quantized = 0u32;
    for value in values {
        let quantized_value = ((value - offset) / scale).round();
        if !quantized_value.is_finite() || !(0.0..=(u32::MAX as f64)).contains(&quantized_value) {
            return Err(Error::Internal(
                "cached bit-stuffed tile quantized value is out of range",
            ));
        }
        let quantized_value = quantized_value as u32;
        max_quantized = max_quantized.max(quantized_value);
        quantized.push(quantized_value);
    }
    if quantized.len() != value_count {
        return Err(Error::Internal(
            "cached bit-stuffed tile value count changed",
        ));
    }

    let bits = bits_required(max_quantized as usize);
    if bits == 0 || bits > 31 {
        return Err(Error::Internal(
            "cached bit-stuffed tile has an invalid bit width",
        ));
    }
    let (count_code, count_bytes) = count_field(value_count)?;
    let bit_bytes = value_count
        .checked_mul(bits as usize)
        .ok_or(Error::SizeOverflow("bit-stuffed payload bit count"))?
        .div_ceil(8);
    payload.clear();
    payload.reserve(1 + count_bytes + bit_bytes);
    payload.push((count_code << 6) | bits);
    append_count(payload, value_count, count_bytes)?;
    pack_lsb_bits_into(quantized, bits, payload)
}

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
