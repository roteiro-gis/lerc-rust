use lerc_core::{fletcher32, BandSetView, DataType, Error, MaskView, RasterView, Result, Sample};

mod analyze;
mod bitstuff;
mod header;
mod huffman;
mod mask;
mod options;
mod tiles;

use analyze::{prepare_band_set, prepare_single_raster, PreparedBand, RasterAnalysis};
use header::{
    body_prefix_len, depth_range_len, header_len, needs_encode_mode_flag, write_depth_ranges,
    write_prefix as write_header_prefix, write_u32, write_value_as, MASK_COUNT_LEN,
};
use huffman::{
    build_plan as build_huffman_plan, supports as supports_integer_huffman,
    write_body as write_huffman_body, HuffmanPlan,
};
use mask::{
    pixel_is_valid, validate_dimensions as validate_mask_dimensions,
    validate_slice as validate_mask_slice, MaskKind, PreparedMask,
};
use options::validate as validate_encode_options;
pub use options::EncodeOptions;
use tiles::{plan as plan_tiled_body, write as write_tiled_body, TileScratch, TilingPlan};

const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";
const VERSION_4: i32 = 4;
const VERSION_5: i32 = 5;
const VERSION_6: i32 = 6;

#[derive(Debug, Clone)]
struct EncodePlan {
    version: i32,
    body: BodyPlan,
}

#[derive(Debug, Clone)]
enum BodyPlan {
    Constant,
    PerDepthConstant,
    OneSweep,
    Tiled(TilingPlan),
    Huffman(HuffmanPlan),
}

trait ByteSink {
    fn push(&mut self, byte: u8) -> Result<()>;
    fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()>;
    fn len(&self) -> usize;
}

impl ByteSink for Vec<u8> {
    fn push(&mut self, byte: u8) -> Result<()> {
        Vec::push(self, byte);
        Ok(())
    }

    fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()> {
        Vec::extend_from_slice(self, bytes);
        Ok(())
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }
}

struct SliceSink<'a> {
    out: &'a mut [u8],
    len: usize,
}

impl<'a> SliceSink<'a> {
    fn new(out: &'a mut [u8]) -> Self {
        Self { out, len: 0 }
    }
}

impl ByteSink for SliceSink<'_> {
    fn push(&mut self, byte: u8) -> Result<()> {
        if self.len >= self.out.len() {
            return Err(Error::OutputTooSmall {
                needed: self.len + 1,
                available: self.out.len(),
            });
        }
        self.out[self.len] = byte;
        self.len += 1;
        Ok(())
    }

    fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .len
            .checked_add(bytes.len())
            .ok_or(Error::SizeOverflow("encoded blob size"))?;
        if end > self.out.len() {
            return Err(Error::OutputTooSmall {
                needed: end,
                available: self.out.len(),
            });
        }
        self.out[self.len..end].copy_from_slice(bytes);
        self.len = end;
        Ok(())
    }

    fn len(&self) -> usize {
        self.len
    }
}

trait RasterSource<T: Sample>: Copy {
    fn width(self) -> u32;
    fn height(self) -> u32;
    fn depth(self) -> u32;
    fn data_type(self) -> DataType;
    fn pixel_count(self) -> Result<usize>;
    fn sample(self, pixel: usize, dim: usize) -> T;
}

impl<T: Sample> RasterSource<T> for RasterView<'_, T> {
    fn width(self) -> u32 {
        self.width()
    }

    fn height(self) -> u32 {
        self.height()
    }

    fn depth(self) -> u32 {
        self.depth()
    }

    fn data_type(self) -> DataType {
        self.data_type()
    }

    fn pixel_count(self) -> Result<usize> {
        self.pixel_count()
    }

    fn sample(self, pixel: usize, dim: usize) -> T {
        self.sample(pixel, dim)
    }
}

#[derive(Debug, Clone, Copy)]
struct BandRasterView<'a, T: Sample> {
    band_set: BandSetView<'a, T>,
    band_index: usize,
}

impl<T: Sample> RasterSource<T> for BandRasterView<'_, T> {
    fn width(self) -> u32 {
        self.band_set.width()
    }

    fn height(self) -> u32 {
        self.band_set.height()
    }

    fn depth(self) -> u32 {
        self.band_set.depth()
    }

    fn data_type(self) -> DataType {
        self.band_set.data_type()
    }

    fn pixel_count(self) -> Result<usize> {
        self.band_set.pixel_count()
    }

    fn sample(self, pixel: usize, dim: usize) -> T {
        self.band_set.sample(self.band_index, pixel, dim)
    }
}

#[derive(Debug, Clone, Copy)]
struct RemappedRaster<T: Sample, R: RasterSource<T>> {
    raster: R,
    no_data_mapping: Option<(T, T)>,
}

impl<T: Sample, R: RasterSource<T>> RemappedRaster<T, R> {
    fn new(raster: R, analysis: &RasterAnalysis) -> Self {
        let no_data_mapping = analysis
            .original_no_data_value
            .zip(analysis.encoded_no_data_value)
            .map(|(original, encoded)| (T::from_f64(original), T::from_f64(encoded)))
            .filter(|(original, encoded)| original.to_f64() != encoded.to_f64());
        Self {
            raster,
            no_data_mapping,
        }
    }
}

impl<T: Sample, R: RasterSource<T>> RasterSource<T> for RemappedRaster<T, R> {
    fn width(self) -> u32 {
        self.raster.width()
    }

    fn height(self) -> u32 {
        self.raster.height()
    }

    fn depth(self) -> u32 {
        self.raster.depth()
    }

    fn data_type(self) -> DataType {
        self.raster.data_type()
    }

    fn pixel_count(self) -> Result<usize> {
        self.raster.pixel_count()
    }

    fn sample(self, pixel: usize, dim: usize) -> T {
        let value = self.raster.sample(pixel, dim);
        match self.no_data_mapping {
            Some((original, encoded)) if value.to_f64() == original.to_f64() => encoded,
            _ => value,
        }
    }
}

/// Returns a conservative output-buffer size for encoding one raster.
///
/// # Errors
/// Returns an error for invalid options or dimensions, invalid samples or
/// masks, unsupported no-data use, or overflowing size calculations.
pub fn encoded_len_upper_bound<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;

    let mask = mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data()));
    let prepared = prepare_single_raster(raster, mask, options)?;
    encoded_len_upper_bound_from_analysis(raster, &prepared.mask, &prepared.analysis)
}

/// Returns a conservative output-buffer size for a concatenated band set.
///
/// # Errors
/// Returns an error for invalid options, shapes, masks, sample values, or
/// overflowing aggregate sizes.
pub fn encoded_band_set_len_upper_bound<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let prepared = prepare_band_set(band_set, mask.map(MaskView::data), options)?;
    let mut total = 0usize;
    for (band_index, prepared) in prepared.iter().enumerate() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        total = total
            .checked_add(encoded_len_upper_bound_from_analysis(
                band,
                &prepared.mask,
                &prepared.analysis,
            )?)
            .ok_or(Error::SizeOverflow("encoded band set size"))?;
    }
    Ok(total)
}

/// Encodes one raster as a self-contained Lerc2 blob.
///
/// # Errors
/// Returns an error when the inputs or options are invalid, size arithmetic
/// overflows, or an internal encoding invariant is violated.
pub fn encode<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;

    let mask = mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data()));
    let prepared = prepare_single_raster(raster, mask, options)?;
    let upper_bound =
        encoded_len_upper_bound_from_analysis(raster, &prepared.mask, &prepared.analysis)?;
    let mut out = vec![0u8; upper_bound];
    let written = encode_into_with_analysis(raster, &prepared.mask, &prepared.analysis, &mut out)?;
    out.truncate(written);
    Ok(out)
}

/// Encodes a multi-band raster as concatenated Lerc2 blobs.
///
/// A supplied mask is stored by the first band and inherited by later bands
/// unless no-data filtering produces distinct per-band masks.
///
/// # Errors
/// Returns an error when inputs or options are invalid or encoding fails.
pub fn encode_band_set<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let prepared = prepare_band_set(band_set, mask.map(MaskView::data), options)?;
    let mut upper_bound = 0usize;

    for (band_index, prepared) in prepared.iter().enumerate() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        upper_bound = upper_bound
            .checked_add(encoded_len_upper_bound_from_analysis(
                band,
                &prepared.mask,
                &prepared.analysis,
            )?)
            .ok_or(Error::SizeOverflow("encoded band set size"))?;
    }

    let mut out = vec![0u8; upper_bound];
    let written = encode_band_set_into_with_analysis(band_set, &prepared, &mut out)?;
    out.truncate(written);
    Ok(out)
}

/// Encodes one raster directly into a caller-provided byte slice.
///
/// Returns the number of bytes written.
///
/// # Errors
/// In addition to validation and encoding errors, returns
/// [`Error::OutputTooSmall`] when `out` cannot hold the blob.
pub fn encode_into<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
    out: &mut [u8],
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;

    let mask = mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data()));
    let prepared = prepare_single_raster(raster, mask, options)?;
    encode_into_with_analysis(raster, &prepared.mask, &prepared.analysis, out)
}

/// Encodes a multi-band raster directly into a caller-provided byte slice.
///
/// Returns the total number of concatenated bytes written.
///
/// # Errors
/// In addition to validation and encoding errors, returns
/// [`Error::OutputTooSmall`] when `out` is insufficient.
pub fn encode_band_set_into<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
    out: &mut [u8],
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let prepared = prepare_band_set(band_set, mask.map(MaskView::data), options)?;
    encode_band_set_into_with_analysis(band_set, &prepared, out)
}

fn encoded_len_upper_bound_from_analysis<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: &PreparedMask<'_>,
    analysis: &RasterAnalysis,
) -> Result<usize> {
    let pixel_count = raster.pixel_count()?;
    validate_mask_slice(mask.data(), pixel_count)?;
    let valid_pixel_count = analysis.valid_pixel_count as usize;
    let depth = raster.depth() as usize;
    let num_tiles = tile_count(
        raster.width() as usize,
        raster.height() as usize,
        analysis.micro_block_size,
    )?;
    let byte_len = raster.data_type().byte_len();
    let mask_len = mask.payload().len();
    let range_len = depth_range_len(analysis)?;
    let prefix_len = if analysis.valid_pixel_count == 0
        || analysis.z_min == analysis.z_max
        || has_per_depth_constant(analysis)
    {
        0
    } else {
        body_prefix_len(
            raster.data_type(),
            analysis.max_z_error,
            analysis.plan.version,
        )
    };
    let tile_header_len = num_tiles
        .checked_mul(depth)
        .ok_or(Error::SizeOverflow("tile header byte count"))?;
    let raw_data_len = valid_pixel_count
        .checked_mul(depth)
        .and_then(|len| len.checked_mul(byte_len))
        .ok_or(Error::SizeOverflow("raw tile payload byte count"))?;

    header_len(analysis.plan.version)
        .checked_add(MASK_COUNT_LEN)
        .and_then(|len| len.checked_add(mask_len))
        .and_then(|len| len.checked_add(range_len))
        .and_then(|len| len.checked_add(prefix_len))
        .and_then(|len| len.checked_add(tile_header_len))
        .and_then(|len| len.checked_add(raw_data_len))
        .ok_or(Error::SizeOverflow("encoded upper bound"))
}

fn encode_into_with_analysis<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: &PreparedMask<'_>,
    analysis: &RasterAnalysis,
    out: &mut [u8],
) -> Result<usize> {
    let pixel_count = raster.pixel_count()?;
    validate_mask_slice(mask.data(), pixel_count)?;

    let mut sink = SliceSink::new(out);
    let mut scratch = TileScratch::default();
    write_header_prefix(&mut sink, analysis)?;
    write_u32(
        &mut sink,
        u32::try_from(mask.payload().len())
            .map_err(|_| Error::SizeOverflow("mask payload length as u32"))?,
    )?;
    sink.extend_from_slice(mask.payload())?;
    write_depth_ranges(&mut sink, analysis)?;
    write_body(
        &mut sink,
        &mut scratch,
        RemappedRaster::new(raster, analysis),
        mask.data(),
        analysis,
    )?;

    let written = sink.len();
    if written > i32::MAX as usize {
        return Err(Error::SizeOverflow("Lerc2 blob-size header field"));
    }

    out[34..38].copy_from_slice(&(written as i32).to_le_bytes());
    let checksum = fletcher32(&out[14..written]);
    out[10..14].copy_from_slice(&checksum.to_le_bytes());
    Ok(written)
}

fn encode_band_set_into_with_analysis<T: Sample>(
    band_set: BandSetView<'_, T>,
    prepared: &[PreparedBand<'_>],
    out: &mut [u8],
) -> Result<usize> {
    if prepared.len() != band_set.band_count() {
        return Err(Error::Internal(
            "band analysis count does not match band_count",
        ));
    }

    let mut offset = 0usize;
    for (band_index, prepared) in prepared.iter().enumerate() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        let written = encode_into_with_analysis(
            band,
            &prepared.mask,
            &prepared.analysis,
            &mut out[offset..],
        )?;
        offset = offset
            .checked_add(written)
            .ok_or(Error::SizeOverflow("encoded band set size"))?;
    }
    Ok(offset)
}

fn write_body<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    scratch: &mut TileScratch,
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<()> {
    match &analysis.plan.body {
        BodyPlan::Constant | BodyPlan::PerDepthConstant => Ok(()),
        BodyPlan::OneSweep => write_one_sweep_body(sink, raster, mask),
        BodyPlan::Tiled(plan) => write_tiled_body(sink, scratch, raster, mask, analysis, plan),
        BodyPlan::Huffman(plan) => write_huffman_body(sink, raster, mask, plan),
    }
}

fn plan_raster<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<EncodePlan> {
    if analysis.valid_pixel_count == 0 || analysis.z_min == analysis.z_max {
        return Ok(EncodePlan {
            version: version_with_no_data(analysis, VERSION_4),
            body: BodyPlan::Constant,
        });
    }
    if has_per_depth_constant(analysis) {
        return Ok(EncodePlan {
            version: version_with_no_data(analysis, VERSION_4),
            body: BodyPlan::PerDepthConstant,
        });
    }

    let tiling = plan_tiled_body(raster, mask, analysis)?;
    let mut best_version = tiling.version;
    let mut best_non_one_len = tiling
        .data_len
        .checked_add(usize::from(needs_encode_mode_flag(
            analysis.data_type,
            analysis.max_z_error,
            version_with_no_data(analysis, tiling.version),
        )))
        .ok_or(Error::SizeOverflow("tiled body byte count"))?;
    let mut selected_huffman = None;

    if let Some(huffman) = build_huffman_plan(analysis)? {
        let huffman_total_len = huffman
            .data_len
            .checked_add(1)
            .ok_or(Error::SizeOverflow("Huffman payload byte count"))?;
        if huffman_total_len < best_non_one_len {
            best_non_one_len = huffman_total_len;
            best_version = version_with_no_data(analysis, VERSION_4);
            selected_huffman = Some(huffman);
        }
    }

    let one_sweep_len = (analysis.valid_pixel_count as usize)
        .checked_mul(analysis.depth as usize)
        .and_then(|len| len.checked_mul(analysis.data_type.byte_len()))
        .ok_or(Error::SizeOverflow("one-sweep byte count"))?;
    if one_sweep_len <= best_non_one_len {
        return Ok(EncodePlan {
            version: version_with_no_data(analysis, VERSION_4),
            body: BodyPlan::OneSweep,
        });
    }

    let best_body = match selected_huffman {
        Some(huffman) => BodyPlan::Huffman(huffman),
        None => BodyPlan::Tiled(tiling),
    };

    Ok(EncodePlan {
        version: version_with_no_data(analysis, best_version),
        body: best_body,
    })
}

fn write_one_sweep_body<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    raster: R,
    mask: Option<&[u8]>,
) -> Result<()> {
    sink.push(1)?;
    let pixel_count = raster.pixel_count()?;
    let depth = raster.depth() as usize;
    for pixel in 0..pixel_count {
        if !pixel_is_valid(mask, pixel) {
            continue;
        }
        for dim in 0..depth {
            write_value_as(sink, raster.sample(pixel, dim).to_f64(), raster.data_type())?;
        }
    }
    Ok(())
}

fn tile_count(width: usize, height: usize, micro_block_size: u32) -> Result<usize> {
    let micro = micro_block_size as usize;
    let num_blocks_x = width.div_ceil(micro);
    let num_blocks_y = height.div_ceil(micro);
    num_blocks_x
        .checked_mul(num_blocks_y)
        .ok_or(Error::SizeOverflow("tile count"))
}

fn has_per_depth_constant(analysis: &RasterAnalysis) -> bool {
    analysis
        .min_values
        .as_ref()
        .zip(analysis.max_values.as_ref())
        .map(|(mins, maxs)| mins == maxs)
        .unwrap_or(false)
}

fn version_with_no_data(analysis: &RasterAnalysis, version: i32) -> i32 {
    if analysis.original_no_data_value.is_some() {
        VERSION_6
    } else {
        version
    }
}

fn tile_header(check_code: u8, encoding: u8) -> u8 {
    ((check_code & 15) << 2) | (encoding & 3)
}
