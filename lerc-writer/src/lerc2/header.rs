use lerc_core::{DataType, Error, Result, Sample};

use super::{supports_integer_huffman, ByteSink, RasterAnalysis, MAGIC_LERC2, VERSION_6};

const FIXED_HEADER_LEN_V4_V5: usize = 66;
const FIXED_HEADER_LEN_V6: usize = 90;
pub(super) const MASK_COUNT_LEN: usize = 4;

pub(super) fn write_prefix(
    sink: &mut impl ByteSink,
    analysis: &RasterAnalysis,
    version: i32,
    remaining_bands: i32,
) -> Result<()> {
    sink.extend_from_slice(MAGIC_LERC2)?;
    write_i32(sink, version)?;
    write_u32(sink, 0)?;
    write_u32(sink, analysis.height)?;
    write_u32(sink, analysis.width)?;
    write_u32(sink, analysis.depth)?;
    write_u32(sink, analysis.valid_pixel_count)?;
    write_i32(sink, analysis.micro_block_size as i32)?;
    write_i32(sink, 0)?;
    write_i32(sink, analysis.data_type.code() as i32)?;
    if version >= VERSION_6 {
        write_i32(sink, remaining_bands)?;
        sink.push(u8::from(analysis.original_no_data_value.is_some()))?;
        sink.push(0)?;
        sink.push(0)?;
        sink.push(0)?;
    }
    write_f64(sink, analysis.max_z_error)?;
    write_f64(sink, analysis.z_min)?;
    write_f64(sink, analysis.z_max)?;
    if version >= VERSION_6 {
        write_f64(sink, analysis.encoded_no_data_value.unwrap_or(0.0))?;
        write_f64(sink, analysis.original_no_data_value.unwrap_or(0.0))?;
    }
    Ok(())
}

pub(super) fn write_depth_ranges(
    sink: &mut impl ByteSink,
    analysis: &RasterAnalysis,
) -> Result<()> {
    if let (Some(min_values), Some(max_values)) = (&analysis.min_values, &analysis.max_values) {
        for &value in min_values {
            write_value_as(sink, value, analysis.data_type)?;
        }
        for &value in max_values {
            write_value_as(sink, value, analysis.data_type)?;
        }
    }
    Ok(())
}

pub(super) fn depth_range_len(analysis: &RasterAnalysis) -> Result<usize> {
    if analysis.min_values.is_none() {
        return Ok(0);
    }
    (analysis.depth as usize)
        .checked_mul(2)
        .and_then(|len| len.checked_mul(analysis.data_type.byte_len()))
        .ok_or(Error::SizeOverflow("depth-range byte count"))
}

pub(super) fn header_len(version: i32) -> usize {
    if version >= VERSION_6 {
        FIXED_HEADER_LEN_V6
    } else {
        FIXED_HEADER_LEN_V4_V5
    }
}

pub(super) fn body_prefix_len(data_type: DataType, max_z_error: f64, version: i32) -> usize {
    1 + usize::from(needs_encode_mode_flag(data_type, max_z_error, version))
}

pub(super) fn needs_encode_mode_flag(data_type: DataType, max_z_error: f64, version: i32) -> bool {
    supports_integer_huffman(data_type, max_z_error)
        || (version >= VERSION_6
            && matches!(data_type, DataType::F32 | DataType::F64)
            && max_z_error == 0.0)
}

pub(super) fn write_value_as(
    sink: &mut impl ByteSink,
    value: f64,
    data_type: DataType,
) -> Result<()> {
    lerc_core::dispatch_data_type!(data_type, Target => {
        let value = Target::from_f64(value);
        sink.extend_from_slice(&value.to_le_bytes())
    })
}

pub(super) fn write_u32(sink: &mut impl ByteSink, value: u32) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn write_i32(sink: &mut impl ByteSink, value: i32) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn write_f64(sink: &mut impl ByteSink, value: f64) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}
