use lerc_core::{
    append_value_as, bits_required, fletcher32, DataType, Error, MaskView, RasterView, Result,
    Sample,
};

const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";
const VERSION: i32 = 4;
const FIXED_HEADER_LEN: usize = 66;
const MASK_COUNT_LEN: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    pub max_z_error: f64,
    pub micro_block_size: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            max_z_error: 0.0,
            micro_block_size: 8,
        }
    }
}

#[derive(Debug, Clone)]
struct EncodePlan {
    data_type: DataType,
    width: u32,
    height: u32,
    depth: u32,
    valid_pixel_count: u32,
    micro_block_size: u32,
    max_z_error: f64,
    z_min: f64,
    z_max: f64,
    mask_bytes: Vec<u8>,
    min_values: Option<Vec<f64>>,
    max_values: Option<Vec<f64>>,
    body: Vec<u8>,
    exact_len: usize,
}

pub fn encoded_len_upper_bound<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<usize> {
    Ok(build_plan(raster, mask, options)?.exact_len)
}

pub fn encode<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    let plan = build_plan(raster, mask, options)?;
    serialize_plan(&plan)
}

pub fn encode_into<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
    out: &mut [u8],
) -> Result<usize> {
    let encoded = encode(raster, mask, options)?;
    if out.len() < encoded.len() {
        return Err(Error::OutputTooSmall {
            needed: encoded.len(),
            available: out.len(),
        });
    }
    out[..encoded.len()].copy_from_slice(&encoded);
    Ok(encoded.len())
}

fn build_plan<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<EncodePlan> {
    validate_options(raster, mask, options)?;

    let width = raster.width();
    let height = raster.height();
    let depth = raster.depth();
    let pixel_count = raster.pixel_count()?;
    let depth_usize = depth as usize;
    let mask_slice = mask.map(MaskView::data);
    let data_type = raster.data_type();

    let mut valid_pixel_count = 0usize;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut min_values = vec![f64::INFINITY; depth_usize];
    let mut max_values = vec![f64::NEG_INFINITY; depth_usize];

    for pixel in 0..pixel_count {
        if !pixel_is_valid(mask_slice, pixel) {
            continue;
        }
        valid_pixel_count += 1;
        for dim in 0..depth_usize {
            let value = raster.sample(pixel, dim).to_f64();
            if !value.is_finite() {
                return Err(Error::InvalidArgument(
                    "valid raster samples must be finite".into(),
                ));
            }
            z_min = z_min.min(value);
            z_max = z_max.max(value);
            min_values[dim] = min_values[dim].min(value);
            max_values[dim] = max_values[dim].max(value);
        }
    }

    let valid_pixel_count = u32::try_from(valid_pixel_count)
        .map_err(|_| Error::InvalidArgument("valid pixel count exceeds u32".into()))?;
    if valid_pixel_count == 0 {
        z_min = 0.0;
        z_max = 0.0;
    }

    let mask_bytes = if valid_pixel_count == 0 || valid_pixel_count == pixel_count as u32 {
        Vec::new()
    } else {
        encode_mask_rle(mask_slice.expect("partial-valid rasters require a mask"))
    };

    let per_depth_ranges = if valid_pixel_count != 0 && z_min != z_max {
        Some((min_values, max_values))
    } else {
        None
    };
    let has_per_depth_constant = per_depth_ranges
        .as_ref()
        .map(|(mins, maxs)| mins == maxs)
        .unwrap_or(false);

    let body = if valid_pixel_count == 0 || z_min == z_max || has_per_depth_constant {
        Vec::new()
    } else {
        encode_tile_body(raster, mask_slice, options)?
    };

    let range_len = per_depth_ranges
        .as_ref()
        .map(|(mins, maxs)| (mins.len() + maxs.len()) * data_type.byte_len())
        .unwrap_or(0);
    let exact_len = FIXED_HEADER_LEN
        .checked_add(MASK_COUNT_LEN)
        .and_then(|len| len.checked_add(mask_bytes.len()))
        .and_then(|len| len.checked_add(range_len))
        .and_then(|len| len.checked_add(body.len()))
        .ok_or_else(|| Error::InvalidArgument("encoded blob size overflows usize".into()))?;
    if exact_len > i32::MAX as usize {
        return Err(Error::InvalidArgument(
            "encoded blob size exceeds the Lerc2 header limit".into(),
        ));
    }

    let (min_values, max_values) = per_depth_ranges
        .map(|(mins, maxs)| (Some(mins), Some(maxs)))
        .unwrap_or((None, None));

    Ok(EncodePlan {
        data_type,
        width,
        height,
        depth,
        valid_pixel_count,
        micro_block_size: options.micro_block_size,
        max_z_error: options.max_z_error,
        z_min,
        z_max,
        mask_bytes,
        min_values,
        max_values,
        body,
        exact_len,
    })
}

fn validate_options<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<()> {
    if !options.max_z_error.is_finite() || options.max_z_error < 0.0 {
        return Err(Error::InvalidArgument(
            "max_z_error must be finite and non-negative".into(),
        ));
    }
    if options.micro_block_size == 0 {
        return Err(Error::InvalidArgument(
            "micro_block_size must be greater than zero".into(),
        ));
    }
    if options.micro_block_size > i32::MAX as u32 {
        return Err(Error::InvalidArgument(
            "micro_block_size exceeds the Lerc2 header limit".into(),
        ));
    }
    if let Some(mask) = mask {
        if mask.width() != raster.width() || mask.height() != raster.height() {
            return Err(Error::InvalidArgument(
                "mask dimensions must match the raster dimensions".into(),
            ));
        }
    }
    Ok(())
}

fn encode_tile_body<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<&[u8]>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let micro = options.micro_block_size as usize;
    let data_type = raster.data_type();

    let num_blocks_x = width.div_ceil(micro);
    let num_blocks_y = height.div_ceil(micro);
    let last_block_width = if width % micro == 0 {
        micro
    } else {
        width % micro
    };
    let last_block_height = if height % micro == 0 {
        micro
    } else {
        height % micro
    };

    let mut out = Vec::new();
    out.push(0);
    if matches!(data_type, DataType::I8 | DataType::U8) && (options.max_z_error - 0.5).abs() < 1e-5
    {
        out.push(0);
    }

    for block_y in 0..num_blocks_y {
        let block_height = if block_y + 1 == num_blocks_y {
            last_block_height
        } else {
            micro
        };
        for block_x in 0..num_blocks_x {
            let block_width = if block_x + 1 == num_blocks_x {
                last_block_width
            } else {
                micro
            };

            for dim in 0..depth {
                let mut typed_values = Vec::with_capacity(block_width * block_height);
                let mut values = Vec::with_capacity(block_width * block_height);
                for row in 0..block_height {
                    let pixel_row = block_y * micro + row;
                    for col in 0..block_width {
                        let pixel = pixel_row * width + block_x * micro + col;
                        if !pixel_is_valid(mask, pixel) {
                            continue;
                        }
                        let value = raster.sample(pixel, dim);
                        typed_values.push(value);
                        values.push(value.to_f64());
                    }
                }

                let check_code = (((block_x * micro) >> 3) as u8) & 15;
                if values.is_empty() {
                    out.push(tile_header(check_code, 2));
                    continue;
                }

                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if min == max {
                    out.push(tile_header(check_code, 3));
                    append_value_as(&mut out, min, data_type);
                    continue;
                }

                let raw_len = 1 + typed_values.len() * data_type.byte_len();
                if let Some(bitstuff) = try_bitstuff_tile(&values, min, max, options.max_z_error)? {
                    if bitstuff.encoded_len(data_type) < raw_len {
                        out.push(tile_header(check_code, 1));
                        append_value_as(&mut out, bitstuff.offset, data_type);
                        out.extend_from_slice(&bitstuff.payload);
                        continue;
                    }
                }

                out.push(tile_header(check_code, 0));
                for value in typed_values {
                    value.append_le_bytes(&mut out);
                }
            }
        }
    }

    Ok(out)
}

#[derive(Debug, Clone)]
struct BitstuffTile {
    offset: f64,
    payload: Vec<u8>,
}

impl BitstuffTile {
    fn encoded_len(&self, data_type: DataType) -> usize {
        1 + data_type.byte_len() + self.payload.len()
    }
}

fn try_bitstuff_tile(
    values: &[f64],
    offset: f64,
    max_value: f64,
    max_z_error: f64,
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
    let mut quantized = Vec::with_capacity(values.len());
    let mut max_quantized = 0u32;
    for &value in values {
        let quantized_value = ((value - offset) / scale).round().clamp(0.0, nmax as f64) as u32;
        let reconstructed = if (quantized_value as f64) < nmax as f64 {
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
    if bits == 0 {
        return Ok(None);
    }

    let (count_code, count_bytes) = count_field(values.len())?;
    let mut payload =
        Vec::with_capacity(1 + count_bytes + (values.len() * bits as usize).div_ceil(8));
    payload.push((count_code << 6) | bits);
    append_count(&mut payload, values.len(), count_bytes)?;
    payload.extend_from_slice(&pack_lsb_bits(&quantized, bits));
    Ok(Some(BitstuffTile { offset, payload }))
}

fn count_field(count: usize) -> Result<(u8, usize)> {
    if count <= u8::MAX as usize {
        Ok((2, 1))
    } else if count <= u16::MAX as usize {
        Ok((1, 2))
    } else if count <= u32::MAX as usize {
        Ok((0, 4))
    } else {
        Err(Error::InvalidArgument(
            "tile valid-value count exceeds u32".into(),
        ))
    }
}

fn append_count(out: &mut Vec<u8>, count: usize, count_bytes: usize) -> Result<()> {
    match count_bytes {
        1 => out.push(
            u8::try_from(count)
                .map_err(|_| Error::InvalidArgument("count does not fit in u8".into()))?,
        ),
        2 => out.extend_from_slice(
            &u16::try_from(count)
                .map_err(|_| Error::InvalidArgument("count does not fit in u16".into()))?
                .to_le_bytes(),
        ),
        4 => out.extend_from_slice(
            &u32::try_from(count)
                .map_err(|_| Error::InvalidArgument("count does not fit in u32".into()))?
                .to_le_bytes(),
        ),
        _ => {
            return Err(Error::InvalidArgument(
                "unsupported count field width".into(),
            ))
        }
    }
    Ok(())
}

fn pack_lsb_bits(values: &[u32], bits_per_value: u8) -> Vec<u8> {
    let total_bits = values.len() * bits_per_value as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_offset = 0usize;
    for &value in values {
        for bit in 0..bits_per_value {
            if ((value >> bit) & 1) != 0 {
                let byte_index = bit_offset / 8;
                let bit_index = bit_offset % 8;
                out[byte_index] |= 1 << bit_index;
            }
            bit_offset += 1;
        }
    }
    out
}

fn encode_mask_rle(mask: &[u8]) -> Vec<u8> {
    let bitset_len = mask.len().div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    for (index, &value) in mask.iter().enumerate() {
        if value != 0 {
            bitset[index >> 3] |= 1 << (7 - (index & 7));
        }
    }

    let mut encoded = Vec::with_capacity(bitset_len + 4 + bitset_len / i16::MAX as usize);
    let mut offset = 0usize;
    while offset < bitset.len() {
        let chunk = (bitset.len() - offset).min(i16::MAX as usize);
        encoded.extend_from_slice(&(chunk as i16).to_le_bytes());
        encoded.extend_from_slice(&bitset[offset..offset + chunk]);
        offset += chunk;
    }
    encoded.extend_from_slice(&i16::MIN.to_le_bytes());
    encoded
}

fn serialize_plan(plan: &EncodePlan) -> Result<Vec<u8>> {
    let blob_size = i32::try_from(plan.exact_len)
        .map_err(|_| Error::InvalidArgument("encoded blob size exceeds i32".into()))?;
    let mut out = Vec::with_capacity(plan.exact_len);
    out.extend_from_slice(MAGIC_LERC2);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&plan.height.to_le_bytes());
    out.extend_from_slice(&plan.width.to_le_bytes());
    out.extend_from_slice(&plan.depth.to_le_bytes());
    out.extend_from_slice(&plan.valid_pixel_count.to_le_bytes());
    out.extend_from_slice(&(plan.micro_block_size as i32).to_le_bytes());
    out.extend_from_slice(&blob_size.to_le_bytes());
    out.extend_from_slice(&(plan.data_type.code() as i32).to_le_bytes());
    out.extend_from_slice(&plan.max_z_error.to_le_bytes());
    out.extend_from_slice(&plan.z_min.to_le_bytes());
    out.extend_from_slice(&plan.z_max.to_le_bytes());
    out.extend_from_slice(&(plan.mask_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&plan.mask_bytes);

    if let (Some(mins), Some(maxs)) = (&plan.min_values, &plan.max_values) {
        for &value in mins {
            append_value_as(&mut out, value, plan.data_type);
        }
        for &value in maxs {
            append_value_as(&mut out, value, plan.data_type);
        }
    }
    out.extend_from_slice(&plan.body);

    let checksum = fletcher32(&out[14..]);
    out[10..14].copy_from_slice(&checksum.to_le_bytes());
    Ok(out)
}

fn tile_header(check_code: u8, encoding: u8) -> u8 {
    ((check_code & 15) << 2) | (encoding & 3)
}

fn pixel_is_valid(mask: Option<&[u8]>, pixel: usize) -> bool {
    mask.map(|mask| mask[pixel] != 0).unwrap_or(true)
}
