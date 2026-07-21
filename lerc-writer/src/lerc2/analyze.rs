use lerc_core::{BandSetView, DataType, Error, Result, Sample};

use super::{
    huffman::{HistogramBuilder, HuffmanHistograms},
    mask::{pixel_is_valid, shared_mask_for_band, validate_slice, MaskKind, PreparedMask},
    select_encode_plan,
    tiles::{plan as plan_tiled_body, TilingOptions},
    BandRasterView, EncodeOptions, EncodePlan, RasterSource, RemappedRaster,
};

#[derive(Debug, Clone)]
pub(super) struct RasterAnalysis {
    pub(super) data_type: DataType,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u32,
    pub(super) valid_pixel_count: u32,
    pub(super) max_z_error: f64,
    pub(super) micro_block_size: u32,
    pub(super) z_min: f64,
    pub(super) z_max: f64,
    pub(super) encoded_no_data_value: Option<f64>,
    pub(super) original_no_data_value: Option<f64>,
    pub(super) min_values: Option<Vec<f64>>,
    pub(super) max_values: Option<Vec<f64>>,
    pub(super) huffman_histograms: Option<HuffmanHistograms>,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedBand<'a> {
    pub(super) mask: PreparedMask<'a>,
    pub(super) analysis: RasterAnalysis,
    pub(super) plan: EncodePlan,
}

struct SemanticStatistics {
    valid_pixel_count: usize,
    z_min: f64,
    z_max: f64,
    min_values: Vec<f64>,
    max_values: Vec<f64>,
}

impl SemanticStatistics {
    fn new(depth: usize) -> Self {
        Self {
            valid_pixel_count: 0,
            z_min: f64::INFINITY,
            z_max: f64::NEG_INFINITY,
            min_values: vec![f64::INFINITY; depth],
            max_values: vec![f64::NEG_INFINITY; depth],
        }
    }

    fn observe_value(&mut self, dim: usize, value: f64) {
        self.z_min = self.z_min.min(value);
        self.z_max = self.z_max.max(value);
        self.min_values[dim] = self.min_values[dim].min(value);
        self.max_values[dim] = self.max_values[dim].max(value);
    }
}

struct FinishedStatistics {
    valid_pixel_count: u32,
    z_min: f64,
    z_max: f64,
    min_values: Option<Vec<f64>>,
    max_values: Option<Vec<f64>>,
}

impl FinishedStatistics {
    fn uses_constant_body(&self) -> bool {
        self.valid_pixel_count == 0
            || self.z_min == self.z_max
            || self
                .min_values
                .as_ref()
                .zip(self.max_values.as_ref())
                .is_some_and(|(mins, maxs)| mins == maxs)
    }
}

fn finish_statistics(
    mut statistics: SemanticStatistics,
    encoded_no_data_value: Option<f64>,
    no_data_by_depth: &[bool],
) -> Result<FinishedStatistics> {
    let valid_pixel_count = u32::try_from(statistics.valid_pixel_count)
        .map_err(|_| Error::SizeOverflow("valid pixel count as u32"))?;
    let mut z_min = statistics.z_min;
    let mut z_max = statistics.z_max;
    if let Some(no_data_value) = encoded_no_data_value {
        z_min = z_min.min(no_data_value);
        z_max = z_max.max(no_data_value);
        for (dim, contains_no_data) in no_data_by_depth.iter().copied().enumerate() {
            if contains_no_data {
                statistics.min_values[dim] = statistics.min_values[dim].min(no_data_value);
                statistics.max_values[dim] = statistics.max_values[dim].max(no_data_value);
            }
        }
    }
    if valid_pixel_count == 0 {
        z_min = 0.0;
        z_max = 0.0;
    }

    let (min_values, max_values) = if valid_pixel_count != 0 && z_min != z_max {
        (Some(statistics.min_values), Some(statistics.max_values))
    } else {
        (None, None)
    };
    Ok(FinishedStatistics {
        valid_pixel_count,
        z_min,
        z_max,
        min_values,
        max_values,
    })
}

pub(super) fn prepare_band_set<'a, T: Sample>(
    band_set: BandSetView<'_, T>,
    shared_mask: Option<&'a [u8]>,
    options: EncodeOptions,
) -> Result<Vec<PreparedBand<'a>>> {
    let mut prepared = Vec::with_capacity(band_set.band_count());
    for band_index in 0..band_set.band_count() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        prepared.push(prepare_raster(
            band,
            shared_mask_for_band(shared_mask, band_index),
            options,
        )?);
    }

    if prepared.iter().any(|band| band.mask.is_derived()) {
        let pixel_count = band_set.pixel_count()?;
        for band in &mut prepared {
            band.mask.make_explicit(pixel_count);
        }
    }
    let pixel_count = band_set.pixel_count()?;
    for band in &mut prepared {
        band.mask
            .prepare_payload(pixel_count, band.analysis.valid_pixel_count as usize)?;
    }
    Ok(prepared)
}

pub(super) fn prepare_single_raster<'a, T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'a>,
    options: EncodeOptions,
) -> Result<PreparedBand<'a>> {
    let mut prepared = prepare_raster(raster, mask, options)?;
    prepared.mask.prepare_payload(
        raster.pixel_count()?,
        prepared.analysis.valid_pixel_count as usize,
    )?;
    Ok(prepared)
}

fn prepare_raster<'a, T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'a>,
    options: EncodeOptions,
) -> Result<PreparedBand<'a>> {
    let pixel_count = raster.pixel_count()?;
    validate_slice(mask.data(), pixel_count)?;
    if options.no_data_value.is_some() && raster.depth() <= 1 {
        return Err(Error::InvalidArgument(
            "no_data_value requires depth greater than one",
        ));
    }
    let depth = raster.depth() as usize;
    let data_type = raster.data_type();
    let input_no_data_value = options
        .no_data_value
        .map(|value| validate_and_coerce_no_data::<T>(value, data_type))
        .transpose()?;
    let mut mask = PreparedMask::from_kind(mask);
    let mut statistics = SemanticStatistics::new(depth);
    let mut no_data_by_depth = vec![false; depth];
    let mut has_mixed_no_data = false;
    if let Some(no_data_value) = input_no_data_value {
        let mut pixel_values = Vec::with_capacity(depth);
        for pixel in 0..pixel_count {
            if !pixel_is_valid(mask.data(), pixel) {
                continue;
            }

            pixel_values.clear();
            let mut no_data_count = 0usize;
            for dim in 0..depth {
                let value = raster.sample(pixel, dim).to_f64();
                validate_sample(value)?;
                no_data_count += usize::from(value == no_data_value);
                pixel_values.push(value);
            }

            if no_data_count == depth {
                mask.derive(pixel_count)?[pixel] = 0;
                continue;
            }

            statistics.valid_pixel_count += 1;
            has_mixed_no_data |= no_data_count != 0;
            for (dim, &value) in pixel_values.iter().enumerate() {
                if value == no_data_value {
                    no_data_by_depth[dim] = true;
                } else {
                    statistics.observe_value(dim, value);
                }
            }
        }
    }

    let mut max_z_error = effective_max_z_error(data_type, options.max_z_error);
    let original_no_data_value = has_mixed_no_data.then_some(options.no_data_value).flatten();
    let mut encoded_no_data_value = has_mixed_no_data.then_some(input_no_data_value).flatten();
    if let Some(no_data_value) = encoded_no_data_value.as_mut() {
        (max_z_error, *no_data_value) = resolve_no_data_encoding::<T>(
            data_type,
            *no_data_value,
            statistics.z_min,
            statistics.z_max,
            max_z_error,
        );
    }

    let mut statistics = Some(statistics);
    let mut finished_statistics = if input_no_data_value.is_some() {
        Some(finish_statistics(
            statistics.take().ok_or(Error::Internal(
                "semantic statistics were already finalized",
            ))?,
            encoded_no_data_value,
            &no_data_by_depth,
        )?)
    } else {
        None
    };
    let skip_tiling = finished_statistics
        .as_ref()
        .is_some_and(FinishedStatistics::uses_constant_body);
    let mut huffman_histograms = if !skip_tiling && encoded_no_data_value == input_no_data_value {
        HistogramBuilder::new(data_type, raster.width() as usize, depth)?
    } else {
        None
    };
    let tiling = if skip_tiling {
        None
    } else {
        let remapped_raster =
            RemappedRaster::new(raster, original_no_data_value, encoded_no_data_value);
        Some(plan_tiled_body(
            remapped_raster,
            mask.data(),
            TilingOptions {
                width: raster.width(),
                height: raster.height(),
                depth: raster.depth(),
                data_type,
                max_z_error,
                micro_block_size: options.micro_block_size,
            },
            |pixel, values| {
                if let Some(statistics) = statistics.as_mut() {
                    statistics.valid_pixel_count += 1;
                    for (dim, &value) in values.iter().enumerate() {
                        validate_sample(value)?;
                        statistics.observe_value(dim, value);
                    }
                } else {
                    for &value in values {
                        validate_sample(value)?;
                    }
                }
                if let Some(histograms) = huffman_histograms.as_mut() {
                    histograms.observe(pixel, mask.data(), values)?;
                }
                Ok(())
            },
        )?)
    };
    if finished_statistics.is_none() {
        finished_statistics = Some(finish_statistics(
            statistics
                .take()
                .ok_or(Error::Internal("semantic statistics are missing"))?,
            encoded_no_data_value,
            &no_data_by_depth,
        )?);
    }
    let finished_statistics =
        finished_statistics.ok_or(Error::Internal("semantic statistics were not finalized"))?;
    let huffman_histograms = huffman_histograms.map(HistogramBuilder::finish);

    let analysis = RasterAnalysis {
        data_type,
        width: raster.width(),
        height: raster.height(),
        depth: raster.depth(),
        valid_pixel_count: finished_statistics.valid_pixel_count,
        max_z_error,
        micro_block_size: options.micro_block_size,
        z_min: finished_statistics.z_min,
        z_max: finished_statistics.z_max,
        encoded_no_data_value,
        original_no_data_value,
        min_values: finished_statistics.min_values,
        max_values: finished_statistics.max_values,
        huffman_histograms,
    };
    let plan = select_encode_plan(&analysis, tiling)?;
    Ok(PreparedBand {
        mask,
        analysis,
        plan,
    })
}

fn validate_sample(value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "valid raster samples must be finite",
        ))
    }
}

fn resolve_no_data_encoding<T: Sample>(
    data_type: DataType,
    original: f64,
    valid_min: f64,
    valid_max: f64,
    max_z_error: f64,
) -> (f64, f64) {
    if !data_type.is_integer() && max_z_error == 0.0 {
        return (max_z_error, original);
    }

    let exclusion_distance = if data_type.is_integer() {
        max_z_error.floor()
    } else {
        2.0 * max_z_error
    };
    if original >= valid_min - exclusion_distance && original <= valid_max + exclusion_distance {
        return (lossless_max_z_error(data_type), original);
    }

    if data_type.is_integer() {
        resolve_integer_no_data::<T>(original, valid_min, valid_max, max_z_error)
    } else {
        resolve_float_no_data::<T>(original, valid_min, max_z_error)
    }
}

fn resolve_integer_no_data<T: Sample>(
    original: f64,
    valid_min: f64,
    valid_max: f64,
    max_z_error: f64,
) -> (f64, f64) {
    let (type_min, type_max) = data_type_range(T::DATA_TYPE);
    let candidate = T::from_f64(valid_min - (max_z_error.floor() + 1.0)).to_f64();
    if candidate >= type_min && candidate < valid_min {
        return (max_z_error, candidate);
    }

    let lossless_max_z_error = 0.5;
    let candidate = T::from_f64(valid_min - 1.0).to_f64();
    if candidate >= type_min && candidate < valid_min {
        return (lossless_max_z_error, candidate);
    }

    let candidate = T::from_f64(valid_max + 1.0).to_f64();
    if candidate <= type_max && candidate > valid_max && candidate < original {
        return (lossless_max_z_error, candidate);
    }

    (lossless_max_z_error, original)
}

fn resolve_float_no_data<T: Sample>(original: f64, valid_min: f64, max_z_error: f64) -> (f64, f64) {
    let distances = [
        4.0 * max_z_error,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1_000.0,
        10_000.0,
    ];
    let large_min_candidate = if valid_min > 0.0 {
        valid_min / 2.0
    } else {
        valid_min * 2.0
    };
    let threshold = T::from_f64(valid_min - 2.0 * max_z_error).to_f64();
    let (type_min, _) = data_type_range(T::DATA_TYPE);
    let mut best_candidate = None::<f64>;

    for candidate in distances
        .into_iter()
        .map(|distance| valid_min - distance)
        .chain(std::iter::once(large_min_candidate))
        .map(|candidate| T::from_f64(candidate).to_f64())
    {
        if candidate.is_finite()
            && candidate > type_min
            && candidate < threshold
            && best_candidate.map_or(true, |best| candidate > best)
        {
            best_candidate = Some(candidate);
        }
    }

    match best_candidate {
        Some(candidate) => (max_z_error, candidate),
        None if original >= valid_min => (0.0, original),
        None => (max_z_error, original),
    }
}

fn lossless_max_z_error(data_type: DataType) -> f64 {
    if data_type.is_integer() {
        0.5
    } else {
        0.0
    }
}

fn data_type_range(data_type: DataType) -> (f64, f64) {
    match data_type {
        DataType::I8 => (i8::MIN as f64, i8::MAX as f64),
        DataType::U8 => (u8::MIN as f64, u8::MAX as f64),
        DataType::I16 => (i16::MIN as f64, i16::MAX as f64),
        DataType::U16 => (u16::MIN as f64, u16::MAX as f64),
        DataType::I32 => (i32::MIN as f64, i32::MAX as f64),
        DataType::U32 => (u32::MIN as f64, u32::MAX as f64),
        DataType::F32 => (-(f32::MAX as f64), f32::MAX as f64),
        DataType::F64 => (-f64::MAX, f64::MAX),
    }
}

fn validate_and_coerce_no_data<T: Sample>(value: f64, data_type: DataType) -> Result<f64> {
    let is_in_range = match data_type {
        DataType::I8 => value >= i8::MIN as f64 && value <= i8::MAX as f64,
        DataType::U8 => value >= u8::MIN as f64 && value <= u8::MAX as f64,
        DataType::I16 => value >= i16::MIN as f64 && value <= i16::MAX as f64,
        DataType::U16 => value >= u16::MIN as f64 && value <= u16::MAX as f64,
        DataType::I32 => value >= i32::MIN as f64 && value <= i32::MAX as f64,
        DataType::U32 => value >= u32::MIN as f64 && value <= u32::MAX as f64,
        DataType::F32 => value >= -(f32::MAX as f64) && value <= f32::MAX as f64,
        DataType::F64 => true,
    };
    if !is_in_range {
        return Err(Error::InvalidArgument(
            "no_data_value is outside the raster data type range",
        ));
    }
    Ok(T::from_f64(value).to_f64())
}

fn effective_max_z_error(data_type: DataType, requested: f64) -> f64 {
    if data_type.is_integer() {
        requested.floor().max(0.5)
    } else {
        requested
    }
}
