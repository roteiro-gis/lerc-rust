use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lerc_core::{
    bits_required, fletcher32, BandSetView, DataType, Error, MaskView, RasterView, Result, Sample,
};

const MAGIC_LERC2: &[u8; 6] = b"Lerc2 ";
const VERSION_4: i32 = 4;
const VERSION_5: i32 = 5;
const FIXED_HEADER_LEN_V4_V5: usize = 66;
const FIXED_HEADER_LEN_V6: usize = 90;
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
struct RasterAnalysis {
    data_type: DataType,
    width: u32,
    height: u32,
    depth: u32,
    valid_pixel_count: u32,
    max_z_error: f64,
    micro_block_size: u32,
    z_min: f64,
    z_max: f64,
    min_values: Option<Vec<f64>>,
    max_values: Option<Vec<f64>>,
    plan: EncodePlan,
}

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
    Tiled,
    Huffman(HuffmanPlan),
}

#[derive(Debug, Clone)]
struct HuffmanPlan {
    mode: HuffmanMode,
    table_bytes: Vec<u8>,
    codes: Vec<Option<HuffmanCode>>,
    data_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HuffmanMode {
    Delta = 1,
    Plain = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuffmanCode {
    bit_len: u8,
    bits: u32,
}

#[derive(Debug, Clone)]
enum HuffmanNodeKind {
    Leaf(u16),
    Branch { left: usize, right: usize },
}

#[derive(Debug, Clone)]
struct HuffmanNode {
    kind: HuffmanNodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct HuffmanHeapEntry {
    freq: u64,
    min_symbol: u16,
    node_index: usize,
}

#[derive(Debug, Clone)]
enum BlockBody {
    ZeroOrEmpty,
    Raw {
        byte_len: usize,
    },
    Constant {
        offset: f64,
        offset_type: DataType,
        type_code: u8,
    },
    Bitstuff {
        offset: f64,
        offset_type: DataType,
        type_code: u8,
        payload_len: usize,
    },
}

#[derive(Debug, Clone)]
struct BlockPlan {
    is_diff: bool,
    body: BlockBody,
}

impl BlockPlan {
    fn encoded_len(&self) -> usize {
        match self.body {
            BlockBody::ZeroOrEmpty => 1,
            BlockBody::Raw { byte_len } => 1 + byte_len,
            BlockBody::Constant { offset_type, .. } => 1 + offset_type.byte_len(),
            BlockBody::Bitstuff {
                offset_type,
                payload_len,
                ..
            } => 1 + offset_type.byte_len() + payload_len,
        }
    }

    fn header_byte(&self, check_code: u8, version: i32) -> u8 {
        let mut header = tile_header(
            check_code,
            match self.body {
                BlockBody::ZeroOrEmpty => 2,
                BlockBody::Raw { .. } => 0,
                BlockBody::Constant { .. } => 3,
                BlockBody::Bitstuff { .. } => 1,
            },
        );
        if self.is_diff && version >= VERSION_5 {
            header |= 4;
        }
        match self.body {
            BlockBody::Constant { type_code, .. } | BlockBody::Bitstuff { type_code, .. } => {
                header |= type_code << 6;
            }
            BlockBody::ZeroOrEmpty | BlockBody::Raw { .. } => {}
        }
        header
    }
}

trait ByteSink {
    fn push(&mut self, byte: u8) -> Result<()>;
    fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()>;
    fn len(&self) -> usize;
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
            .ok_or_else(|| Error::InvalidArgument("encoded blob size overflows usize".into()))?;
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

#[derive(Debug, Default)]
struct TileScratch {
    raw_bytes: Vec<u8>,
    values_f64: Vec<f64>,
    prev_values_f64: Vec<f64>,
    diff_values_f64: Vec<f64>,
    quantized: Vec<u32>,
    bitstuff_payload: Vec<u8>,
}

impl TileScratch {
    fn clear(&mut self) {
        self.raw_bytes.clear();
        self.values_f64.clear();
        self.prev_values_f64.clear();
        self.diff_values_f64.clear();
        self.quantized.clear();
        self.bitstuff_payload.clear();
    }
}

#[derive(Debug, Default)]
struct MsbBitWriter {
    words: Vec<u32>,
    current: u32,
    bits_used: u8,
}

impl MsbBitWriter {
    fn push_bits(&mut self, bits: u32, bit_len: u8) -> Result<()> {
        let mut remaining = bit_len as usize;
        while remaining > 0 {
            let space = 32usize - self.bits_used as usize;
            let take = remaining.min(space);
            let shift_out = remaining - take;
            let chunk_mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            let chunk = (bits >> shift_out) & chunk_mask;
            self.current |= chunk << (space - take);
            self.bits_used += take as u8;
            remaining -= take;
            if self.bits_used == 32 {
                self.words.push(self.current);
                self.current = 0;
                self.bits_used = 0;
            }
        }
        Ok(())
    }

    fn into_aligned_bytes(mut self) -> Vec<u8> {
        if self.bits_used != 0 {
            self.words.push(self.current);
        }
        words_to_le_bytes(&self.words)
    }

    fn into_bytes_with_trailing_word(mut self) -> Vec<u8> {
        if self.bits_used != 0 {
            self.words.push(self.current);
        }
        self.words.push(0);
        words_to_le_bytes(&self.words)
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
enum MaskKind<'a> {
    None,
    Explicit(&'a [u8]),
    External(&'a [u8]),
}

impl<'a> MaskKind<'a> {
    fn data(self) -> Option<&'a [u8]> {
        match self {
            Self::None => None,
            Self::Explicit(mask) | Self::External(mask) => Some(mask),
        }
    }

    fn stored_payload_len(self, pixel_count: usize, valid_pixel_count: usize) -> Result<usize> {
        match self {
            Self::Explicit(mask) => explicit_mask_payload_len(mask, pixel_count, valid_pixel_count),
            Self::None | Self::External(_) => Ok(0),
        }
    }
}

fn shared_mask_for_band(mask: Option<&[u8]>, band_index: usize) -> MaskKind<'_> {
    match mask {
        Some(mask) if band_index == 0 => MaskKind::Explicit(mask),
        Some(mask) => MaskKind::External(mask),
        None => MaskKind::None,
    }
}

fn validate_encode_options(options: EncodeOptions) -> Result<()> {
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
    Ok(())
}

fn validate_mask_dimensions(width: u32, height: u32, mask: Option<MaskView<'_>>) -> Result<()> {
    if let Some(mask) = mask {
        if mask.width() != width || mask.height() != height {
            return Err(Error::InvalidArgument(
                "mask dimensions must match the raster dimensions".into(),
            ));
        }
    }
    Ok(())
}

fn validate_mask_slice(mask: Option<&[u8]>, pixel_count: usize) -> Result<()> {
    if let Some(mask) = mask {
        if mask.len() != pixel_count {
            return Err(Error::InvalidArgument(
                "mask length does not match the raster dimensions".into(),
            ));
        }
    }
    Ok(())
}

pub fn encoded_len_upper_bound<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;
    encoded_len_upper_bound_for_raster(
        raster,
        mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data())),
        options,
    )
}

pub fn encoded_band_set_len_upper_bound<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let shared_mask = mask.map(MaskView::data);
    let mut total = 0usize;
    for band_index in 0..band_set.band_count() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        total = total
            .checked_add(encoded_len_upper_bound_for_raster(
                band,
                shared_mask_for_band(shared_mask, band_index),
                options,
            )?)
            .ok_or_else(|| {
                Error::InvalidArgument("encoded band set size overflows usize".into())
            })?;
    }
    Ok(total)
}

pub fn encode<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;

    let mask = mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data()));
    let analysis = analyze_raster(raster, mask, options)?;
    let upper_bound = encoded_len_upper_bound_from_analysis(raster, mask, options, &analysis)?;
    let mut out = vec![0u8; upper_bound];
    let written = encode_into_with_analysis(raster, mask, options, &analysis, &mut out)?;
    out.truncate(written);
    Ok(out)
}

pub fn encode_band_set<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
) -> Result<Vec<u8>> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let shared_mask = mask.map(MaskView::data);
    let mut analyses = Vec::with_capacity(band_set.band_count());
    let mut upper_bound = 0usize;

    for band_index in 0..band_set.band_count() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        let mask_kind = shared_mask_for_band(shared_mask, band_index);
        let analysis = analyze_raster(band, mask_kind, options)?;
        upper_bound = upper_bound
            .checked_add(encoded_len_upper_bound_from_analysis(
                band, mask_kind, options, &analysis,
            )?)
            .ok_or_else(|| {
                Error::InvalidArgument("encoded band set size overflows usize".into())
            })?;
        analyses.push(analysis);
    }

    let mut out = vec![0u8; upper_bound];
    let written =
        encode_band_set_into_with_analysis(band_set, shared_mask, options, &analyses, &mut out)?;
    out.truncate(written);
    Ok(out)
}

pub fn encode_into<T: Sample>(
    raster: RasterView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
    out: &mut [u8],
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(raster.width(), raster.height(), mask)?;

    let mask = mask.map_or(MaskKind::None, |mask| MaskKind::Explicit(mask.data()));
    let analysis = analyze_raster(raster, mask, options)?;
    encode_into_with_analysis(raster, mask, options, &analysis, out)
}

pub fn encode_band_set_into<T: Sample>(
    band_set: BandSetView<'_, T>,
    mask: Option<MaskView<'_>>,
    options: EncodeOptions,
    out: &mut [u8],
) -> Result<usize> {
    validate_encode_options(options)?;
    validate_mask_dimensions(band_set.width(), band_set.height(), mask)?;

    let shared_mask = mask.map(MaskView::data);
    let mut analyses = Vec::with_capacity(band_set.band_count());
    for band_index in 0..band_set.band_count() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        analyses.push(analyze_raster(
            band,
            shared_mask_for_band(shared_mask, band_index),
            options,
        )?);
    }

    encode_band_set_into_with_analysis(band_set, shared_mask, options, &analyses, out)
}

fn encoded_len_upper_bound_for_raster<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'_>,
    options: EncodeOptions,
) -> Result<usize> {
    let pixel_count = raster.pixel_count()?;
    validate_mask_slice(mask.data(), pixel_count)?;

    let valid_pixel_count = mask
        .data()
        .map(|mask| mask.iter().filter(|&&value| value != 0).count())
        .unwrap_or(pixel_count);
    let depth = raster.depth() as usize;
    let num_tiles = tile_count(raster.width() as usize, raster.height() as usize, options)?;
    let byte_len = raster.data_type().byte_len();
    let mask_len = mask.stored_payload_len(pixel_count, valid_pixel_count)?;
    let range_len = if valid_pixel_count == 0 {
        0
    } else {
        depth
            .checked_mul(2)
            .and_then(|len| len.checked_mul(byte_len))
            .ok_or_else(|| Error::InvalidArgument("range byte count overflows usize".into()))?
    };
    let prefix_len = body_prefix_len(raster.data_type(), options.max_z_error);
    let tile_header_len = num_tiles
        .checked_mul(depth)
        .ok_or_else(|| Error::InvalidArgument("tile header length overflows usize".into()))?;
    let raw_data_len = valid_pixel_count
        .checked_mul(depth)
        .and_then(|len| len.checked_mul(byte_len))
        .ok_or_else(|| Error::InvalidArgument("raw tile payload length overflows usize".into()))?;

    header_len(VERSION_4)
        .checked_add(MASK_COUNT_LEN)
        .and_then(|len| len.checked_add(mask_len))
        .and_then(|len| len.checked_add(range_len))
        .and_then(|len| len.checked_add(prefix_len))
        .and_then(|len| len.checked_add(tile_header_len))
        .and_then(|len| len.checked_add(raw_data_len))
        .ok_or_else(|| Error::InvalidArgument("encoded upper bound overflows usize".into()))
}

fn analyze_raster<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'_>,
    options: EncodeOptions,
) -> Result<RasterAnalysis> {
    let pixel_count = raster.pixel_count()?;
    validate_mask_slice(mask.data(), pixel_count)?;
    let depth = raster.depth() as usize;
    let data_type = raster.data_type();

    let mut valid_pixel_count = 0usize;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut min_values = vec![f64::INFINITY; depth];
    let mut max_values = vec![f64::NEG_INFINITY; depth];

    for pixel in 0..pixel_count {
        if !pixel_is_valid(mask.data(), pixel) {
            continue;
        }
        valid_pixel_count += 1;
        for dim in 0..depth {
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

    let (min_values, max_values) = if valid_pixel_count != 0 && z_min != z_max {
        (Some(min_values), Some(max_values))
    } else {
        (None, None)
    };

    let mut analysis = RasterAnalysis {
        data_type,
        width: raster.width(),
        height: raster.height(),
        depth: raster.depth(),
        valid_pixel_count,
        max_z_error: options.max_z_error,
        micro_block_size: options.micro_block_size,
        z_min,
        z_max,
        min_values,
        max_values,
        plan: EncodePlan {
            version: VERSION_4,
            body: BodyPlan::Constant,
        },
    };
    analysis.plan = plan_raster(raster, mask.data(), &analysis)?;
    Ok(analysis)
}

fn encoded_len_upper_bound_from_analysis<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'_>,
    options: EncodeOptions,
    analysis: &RasterAnalysis,
) -> Result<usize> {
    let pixel_count = raster.pixel_count()?;
    validate_mask_slice(mask.data(), pixel_count)?;
    let valid_pixel_count = analysis.valid_pixel_count as usize;
    let depth = raster.depth() as usize;
    let num_tiles = tile_count(raster.width() as usize, raster.height() as usize, options)?;
    let byte_len = raster.data_type().byte_len();
    let mask_len = mask.stored_payload_len(pixel_count, valid_pixel_count)?;
    let range_len = depth_range_len(analysis)?;
    let prefix_len = if analysis.valid_pixel_count == 0
        || analysis.z_min == analysis.z_max
        || has_per_depth_constant(analysis)
    {
        0
    } else {
        body_prefix_len(raster.data_type(), options.max_z_error)
    };
    let tile_header_len = num_tiles
        .checked_mul(depth)
        .ok_or_else(|| Error::InvalidArgument("tile header length overflows usize".into()))?;
    let raw_data_len = valid_pixel_count
        .checked_mul(depth)
        .and_then(|len| len.checked_mul(byte_len))
        .ok_or_else(|| Error::InvalidArgument("raw tile payload length overflows usize".into()))?;

    header_len(analysis.plan.version)
        .checked_add(MASK_COUNT_LEN)
        .and_then(|len| len.checked_add(mask_len))
        .and_then(|len| len.checked_add(range_len))
        .and_then(|len| len.checked_add(prefix_len))
        .and_then(|len| len.checked_add(tile_header_len))
        .and_then(|len| len.checked_add(raw_data_len))
        .ok_or_else(|| Error::InvalidArgument("encoded upper bound overflows usize".into()))
}

fn encode_into_with_analysis<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: MaskKind<'_>,
    _options: EncodeOptions,
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
        mask.stored_payload_len(pixel_count, analysis.valid_pixel_count as usize)? as u32,
    )?;
    write_mask_rle(
        &mut sink,
        mask,
        pixel_count,
        analysis.valid_pixel_count as usize,
    )?;
    write_depth_ranges(&mut sink, analysis)?;
    write_body(&mut sink, &mut scratch, raster, mask.data(), analysis)?;

    let written = sink.len();
    if written > i32::MAX as usize {
        return Err(Error::InvalidArgument(
            "encoded blob size exceeds the Lerc2 header limit".into(),
        ));
    }

    out[34..38].copy_from_slice(&(written as i32).to_le_bytes());
    let checksum = fletcher32(&out[14..written]);
    out[10..14].copy_from_slice(&checksum.to_le_bytes());
    Ok(written)
}

fn encode_band_set_into_with_analysis<T: Sample>(
    band_set: BandSetView<'_, T>,
    shared_mask: Option<&[u8]>,
    options: EncodeOptions,
    analyses: &[RasterAnalysis],
    out: &mut [u8],
) -> Result<usize> {
    if analyses.len() != band_set.band_count() {
        return Err(Error::InvalidArgument(
            "band analysis count does not match band_count".into(),
        ));
    }

    let mut offset = 0usize;
    for (band_index, analysis) in analyses.iter().enumerate() {
        let band = BandRasterView {
            band_set,
            band_index,
        };
        let written = encode_into_with_analysis(
            band,
            shared_mask_for_band(shared_mask, band_index),
            options,
            analysis,
            &mut out[offset..],
        )?;
        offset = offset.checked_add(written).ok_or_else(|| {
            Error::InvalidArgument("encoded band set size overflows usize".into())
        })?;
    }
    Ok(offset)
}

fn write_header_prefix(sink: &mut impl ByteSink, analysis: &RasterAnalysis) -> Result<()> {
    sink.extend_from_slice(MAGIC_LERC2)?;
    write_i32(sink, analysis.plan.version)?;
    write_u32(sink, 0)?;
    write_u32(sink, analysis.height)?;
    write_u32(sink, analysis.width)?;
    write_u32(sink, analysis.depth)?;
    write_u32(sink, analysis.valid_pixel_count)?;
    write_i32(sink, analysis.micro_block_size as i32)?;
    write_i32(sink, 0)?;
    write_i32(sink, analysis.data_type.code() as i32)?;
    write_f64(sink, analysis.max_z_error)?;
    write_f64(sink, analysis.z_min)?;
    write_f64(sink, analysis.z_max)?;
    Ok(())
}

fn write_depth_ranges(sink: &mut impl ByteSink, analysis: &RasterAnalysis) -> Result<()> {
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
        BodyPlan::Tiled => write_tiled_body(sink, scratch, raster, mask, analysis),
        BodyPlan::Huffman(plan) => write_huffman_body(sink, raster, mask, analysis, plan),
    }
}

#[derive(Debug, Clone, Copy)]
struct TilingPlan {
    version: i32,
    data_len: usize,
}

fn plan_raster<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<EncodePlan> {
    if analysis.valid_pixel_count == 0 || analysis.z_min == analysis.z_max {
        return Ok(EncodePlan {
            version: VERSION_4,
            body: BodyPlan::Constant,
        });
    }
    if has_per_depth_constant(analysis) {
        return Ok(EncodePlan {
            version: VERSION_4,
            body: BodyPlan::PerDepthConstant,
        });
    }

    let tiling = plan_tiled_body(raster, mask, analysis)?;
    let mut best_body = BodyPlan::Tiled;
    let mut best_version = tiling.version;
    let mut best_non_one_len =
        tiling.data_len + usize::from(needs_huffman_flag(analysis.data_type, analysis.max_z_error));

    if let Some(huffman) = build_huffman_plan(raster, mask, analysis)? {
        let huffman_total_len = huffman.data_len.checked_add(1).ok_or_else(|| {
            Error::InvalidArgument("Huffman payload length overflows usize".into())
        })?;
        if huffman_total_len < best_non_one_len {
            best_non_one_len = huffman_total_len;
            best_version = VERSION_4;
            best_body = BodyPlan::Huffman(huffman);
        }
    }

    let one_sweep_len = (analysis.valid_pixel_count as usize)
        .checked_mul(analysis.depth as usize)
        .and_then(|len| len.checked_mul(analysis.data_type.byte_len()))
        .ok_or_else(|| Error::InvalidArgument("one-sweep byte count overflows usize".into()))?;
    if one_sweep_len <= best_non_one_len {
        return Ok(EncodePlan {
            version: VERSION_4,
            body: BodyPlan::OneSweep,
        });
    }

    Ok(EncodePlan {
        version: best_version,
        body: best_body,
    })
}

fn plan_tiled_body<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<TilingPlan> {
    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let micro = analysis.micro_block_size as usize;
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
    let diff_supported =
        supports_diff_tiles(analysis.data_type, analysis.max_z_error, analysis.depth);

    let mut scratch = TileScratch::default();
    let mut version4_len = 0usize;
    let mut version5_len = 0usize;
    let mut used_diff = false;

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
                collect_block_values(
                    &mut scratch,
                    raster,
                    mask,
                    width,
                    micro,
                    block_x,
                    block_y,
                    block_width,
                    block_height,
                    dim,
                    diff_supported && dim > 0,
                );

                let absolute_plan = choose_absolute_block_plan(
                    &scratch.values_f64,
                    scratch.raw_bytes.len(),
                    analysis.data_type,
                    analysis.max_z_error,
                    &mut scratch.quantized,
                    &mut scratch.bitstuff_payload,
                )?;
                version4_len = version4_len
                    .checked_add(absolute_plan.encoded_len())
                    .ok_or_else(|| {
                        Error::InvalidArgument("tile payload length overflows usize".into())
                    })?;
                version5_len = version5_len
                    .checked_add(absolute_plan.encoded_len())
                    .ok_or_else(|| {
                        Error::InvalidArgument("tile payload length overflows usize".into())
                    })?;

                if diff_supported && dim > 0 {
                    if build_diff_values(
                        &scratch.values_f64,
                        &scratch.prev_values_f64,
                        &mut scratch.diff_values_f64,
                    )? {
                        if let Some(diff_plan) = choose_diff_block_plan(
                            &scratch.diff_values_f64,
                            analysis.max_z_error,
                            &mut scratch.quantized,
                            &mut scratch.bitstuff_payload,
                        )? {
                            if diff_plan.encoded_len() < absolute_plan.encoded_len() {
                                version5_len = version5_len
                                    .checked_sub(absolute_plan.encoded_len())
                                    .and_then(|len| len.checked_add(diff_plan.encoded_len()))
                                    .ok_or_else(|| {
                                        Error::InvalidArgument(
                                            "tile payload length overflows usize".into(),
                                        )
                                    })?;
                                used_diff = true;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(if used_diff {
        TilingPlan {
            version: VERSION_5,
            data_len: version5_len,
        }
    } else {
        TilingPlan {
            version: VERSION_4,
            data_len: version4_len,
        }
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

fn write_tiled_body<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    scratch: &mut TileScratch,
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<()> {
    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let micro = analysis.micro_block_size as usize;
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
    let diff_supported = analysis.plan.version >= VERSION_5
        && supports_diff_tiles(analysis.data_type, analysis.max_z_error, analysis.depth);

    sink.push(0)?;
    if needs_huffman_flag(analysis.data_type, analysis.max_z_error) {
        sink.push(0)?;
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
                collect_block_values(
                    scratch,
                    raster,
                    mask,
                    width,
                    micro,
                    block_x,
                    block_y,
                    block_width,
                    block_height,
                    dim,
                    diff_supported && dim > 0,
                );
                let check_code = (((block_x * micro) >> 3) as u8) & 15;
                let absolute_plan = choose_absolute_block_plan(
                    &scratch.values_f64,
                    scratch.raw_bytes.len(),
                    analysis.data_type,
                    analysis.max_z_error,
                    &mut scratch.quantized,
                    &mut scratch.bitstuff_payload,
                )?;

                let mut chosen_plan = absolute_plan.clone();
                let mut chose_diff = false;
                if diff_supported && dim > 0 {
                    if build_diff_values(
                        &scratch.values_f64,
                        &scratch.prev_values_f64,
                        &mut scratch.diff_values_f64,
                    )? {
                        if let Some(diff_plan) = choose_diff_block_plan(
                            &scratch.diff_values_f64,
                            analysis.max_z_error,
                            &mut scratch.quantized,
                            &mut scratch.bitstuff_payload,
                        )? {
                            if diff_plan.encoded_len() < absolute_plan.encoded_len() {
                                chosen_plan = diff_plan;
                                chose_diff = true;
                            }
                        }
                    }
                }

                if !chose_diff {
                    chosen_plan = choose_absolute_block_plan(
                        &scratch.values_f64,
                        scratch.raw_bytes.len(),
                        analysis.data_type,
                        analysis.max_z_error,
                        &mut scratch.quantized,
                        &mut scratch.bitstuff_payload,
                    )?;
                }

                write_block_plan(
                    sink,
                    &chosen_plan,
                    check_code,
                    analysis.plan.version,
                    &scratch.raw_bytes,
                    &scratch.bitstuff_payload,
                )?;
            }
        }
    }

    Ok(())
}

fn write_huffman_body<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
    plan: &HuffmanPlan,
) -> Result<()> {
    let _ = analysis;
    sink.push(0)?;
    sink.push(plan.mode as u8)?;
    sink.extend_from_slice(&plan.table_bytes)?;

    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let data_type = raster.data_type();
    let mut writer = MsbBitWriter::default();

    match plan.mode {
        HuffmanMode::Delta => {
            for dim in 0..depth {
                let mut prev_value = 0i32;
                for row in 0..height {
                    for col in 0..width {
                        let pixel = row * width + col;
                        if !pixel_is_valid(mask, pixel) {
                            continue;
                        }
                        let value =
                            huffman_sample_value(raster.sample(pixel, dim).to_f64(), data_type);
                        let predictor = if col > 0 && pixel_is_valid(mask, pixel - 1) {
                            prev_value
                        } else if row > 0 && pixel_is_valid(mask, pixel - width) {
                            huffman_sample_value(
                                raster.sample(pixel - width, dim).to_f64(),
                                data_type,
                            )
                        } else {
                            prev_value
                        };
                        let symbol = huffman_delta_symbol(value - predictor, data_type);
                        let code = plan.codes[symbol].ok_or_else(|| {
                            Error::InvalidArgument("missing Huffman delta symbol".into())
                        })?;
                        writer.push_bits(code.bits, code.bit_len)?;
                        prev_value = value;
                    }
                }
            }
        }
        HuffmanMode::Plain => {
            let pixel_count = raster.pixel_count()?;
            for pixel in 0..pixel_count {
                if !pixel_is_valid(mask, pixel) {
                    continue;
                }
                for dim in 0..depth {
                    let value = huffman_sample_value(raster.sample(pixel, dim).to_f64(), data_type);
                    let symbol = huffman_symbol(value, data_type);
                    let code = plan.codes[symbol]
                        .ok_or_else(|| Error::InvalidArgument("missing Huffman symbol".into()))?;
                    writer.push_bits(code.bits, code.bit_len)?;
                }
            }
        }
    }

    sink.extend_from_slice(&writer.into_bytes_with_trailing_word())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_block_values<T: Sample, R: RasterSource<T>>(
    scratch: &mut TileScratch,
    raster: R,
    mask: Option<&[u8]>,
    width: usize,
    micro: usize,
    block_x: usize,
    block_y: usize,
    block_width: usize,
    block_height: usize,
    dim: usize,
    include_prev_values: bool,
) {
    scratch.clear();
    for row in 0..block_height {
        let pixel_row = block_y * micro + row;
        for col in 0..block_width {
            let pixel = pixel_row * width + block_x * micro + col;
            if !pixel_is_valid(mask, pixel) {
                continue;
            }
            let value = raster.sample(pixel, dim);
            value.append_le_bytes(&mut scratch.raw_bytes);
            scratch.values_f64.push(value.to_f64());
            if include_prev_values {
                scratch
                    .prev_values_f64
                    .push(raster.sample(pixel, dim - 1).to_f64());
            }
        }
    }
}

fn choose_absolute_block_plan(
    values: &[f64],
    raw_byte_len: usize,
    base_type: DataType,
    max_z_error: f64,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<BlockPlan> {
    let raw_len = raw_byte_len
        .checked_add(1)
        .ok_or_else(|| Error::InvalidArgument("raw block length overflows usize".into()))?;
    choose_block_plan(
        values,
        Some(raw_len),
        base_type,
        max_z_error,
        false,
        quantized,
        payload,
    )?
    .ok_or_else(|| Error::InvalidArgument("absolute tile plan unexpectedly missing".into()))
}

fn choose_diff_block_plan(
    diff_values: &[f64],
    max_z_error: f64,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<Option<BlockPlan>> {
    choose_block_plan(
        diff_values,
        None,
        DataType::I32,
        max_z_error,
        true,
        quantized,
        payload,
    )
}

fn choose_block_plan(
    values: &[f64],
    raw_len: Option<usize>,
    base_type: DataType,
    max_z_error: f64,
    is_diff: bool,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<Option<BlockPlan>> {
    if values.is_empty() {
        return Ok(Some(BlockPlan {
            is_diff,
            body: BlockBody::ZeroOrEmpty,
        }));
    }

    let (min, max) = min_max(values);
    if min == 0.0 && max == 0.0 {
        return Ok(Some(BlockPlan {
            is_diff,
            body: BlockBody::ZeroOrEmpty,
        }));
    }
    if min == max {
        let (type_code, offset_type) = reduce_data_type(min, base_type)?;
        return Ok(Some(BlockPlan {
            is_diff,
            body: BlockBody::Constant {
                offset: min,
                offset_type,
                type_code,
            },
        }));
    }

    if let Some(bitstuff) =
        try_bitstuff_tile(values, min, max, max_z_error, base_type, quantized, payload)?
    {
        let plan = BlockPlan {
            is_diff,
            body: BlockBody::Bitstuff {
                offset: bitstuff.offset,
                offset_type: bitstuff.offset_type,
                type_code: bitstuff.type_code,
                payload_len: bitstuff.payload_len,
            },
        };
        if raw_len.is_none() || plan.encoded_len() < raw_len.unwrap() {
            return Ok(Some(plan));
        }
    }

    Ok(raw_len.map(|raw_len| BlockPlan {
        is_diff: false,
        body: BlockBody::Raw {
            byte_len: raw_len - 1,
        },
    }))
}

fn write_block_plan(
    sink: &mut impl ByteSink,
    plan: &BlockPlan,
    check_code: u8,
    version: i32,
    raw_bytes: &[u8],
    bitstuff_payload: &[u8],
) -> Result<()> {
    sink.push(plan.header_byte(check_code, version))?;
    match plan.body {
        BlockBody::ZeroOrEmpty => Ok(()),
        BlockBody::Raw { .. } => sink.extend_from_slice(raw_bytes),
        BlockBody::Constant {
            offset,
            offset_type,
            ..
        } => write_value_as(sink, offset, offset_type),
        BlockBody::Bitstuff {
            offset,
            offset_type,
            payload_len,
            ..
        } => {
            write_value_as(sink, offset, offset_type)?;
            sink.extend_from_slice(&bitstuff_payload[..payload_len])
        }
    }
}

fn build_diff_values(current: &[f64], previous: &[f64], out: &mut Vec<f64>) -> Result<bool> {
    if current.len() != previous.len() {
        return Err(Error::InvalidArgument(
            "diff input lengths do not match".into(),
        ));
    }

    out.clear();
    out.reserve(current.len());
    for (&value, &prev) in current.iter().zip(previous) {
        let diff = value - prev;
        if diff < i32::MIN as f64 || diff > i32::MAX as f64 {
            out.clear();
            return Ok(false);
        }
        out.push(diff);
    }
    Ok(true)
}

fn min_max(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

fn supports_diff_tiles(data_type: DataType, max_z_error: f64, depth: u32) -> bool {
    depth > 1 && is_integer_lossless(data_type, max_z_error)
}

fn is_integer_lossless(data_type: DataType, max_z_error: f64) -> bool {
    matches!(
        data_type,
        DataType::I8 | DataType::U8 | DataType::I16 | DataType::U16 | DataType::I32 | DataType::U32
    ) && (max_z_error - 0.5).abs() < 1e-5
}

fn build_huffman_plan<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
) -> Result<Option<HuffmanPlan>> {
    if !needs_huffman_flag(analysis.data_type, analysis.max_z_error) {
        return Ok(None);
    }

    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let mut hist = [0u64; 256];
    let mut delta_hist = [0u64; 256];

    for dim in 0..depth {
        let mut prev_value = 0i32;
        for row in 0..height {
            for col in 0..width {
                let pixel = row * width + col;
                if !pixel_is_valid(mask, pixel) {
                    continue;
                }
                let value =
                    huffman_sample_value(raster.sample(pixel, dim).to_f64(), analysis.data_type);
                hist[huffman_symbol(value, analysis.data_type)] += 1;

                let predictor = if col > 0 && pixel_is_valid(mask, pixel - 1) {
                    prev_value
                } else if row > 0 && pixel_is_valid(mask, pixel - width) {
                    huffman_sample_value(
                        raster.sample(pixel - width, dim).to_f64(),
                        analysis.data_type,
                    )
                } else {
                    prev_value
                };
                delta_hist[huffman_delta_symbol(value - predictor, analysis.data_type)] += 1;
                prev_value = value;
            }
        }
    }

    let plain = build_huffman_candidate_from_hist(&hist)?;
    let delta = build_huffman_candidate_from_hist(&delta_hist)?;
    let selected = match (plain, delta) {
        (Some(plain), Some(delta)) => {
            if plain.data_len <= delta.data_len {
                Some((HuffmanMode::Plain, plain))
            } else {
                Some((HuffmanMode::Delta, delta))
            }
        }
        (Some(plain), None) => Some((HuffmanMode::Plain, plain)),
        (None, Some(delta)) => Some((HuffmanMode::Delta, delta)),
        (None, None) => None,
    };

    Ok(selected.map(|(mode, candidate)| HuffmanPlan {
        mode,
        table_bytes: candidate.table_bytes,
        codes: candidate.codes,
        data_len: candidate.data_len,
    }))
}

struct HuffmanCandidate {
    table_bytes: Vec<u8>,
    codes: Vec<Option<HuffmanCode>>,
    data_len: usize,
}

fn build_huffman_candidate_from_hist(hist: &[u64; 256]) -> Result<Option<HuffmanCandidate>> {
    let Some(codes) = build_huffman_codes(hist)? else {
        return Ok(None);
    };
    let table_bytes = build_huffman_table_bytes(&codes)?;
    let payload_bits = hist
        .iter()
        .zip(&codes)
        .try_fold(0usize, |acc, (&count, code)| {
            let Some(code) = code else {
                return Ok(acc);
            };
            acc.checked_add((count as usize) * code.bit_len as usize)
                .ok_or_else(|| {
                    Error::InvalidArgument("Huffman payload bit count overflows usize".into())
                })
        })?;
    let payload_bytes = (((payload_bits.div_ceil(32)) + 1) * 4)
        .checked_add(table_bytes.len())
        .ok_or_else(|| Error::InvalidArgument("Huffman payload length overflows usize".into()))?;

    Ok(Some(HuffmanCandidate {
        table_bytes,
        codes,
        data_len: payload_bytes,
    }))
}

fn build_huffman_codes(hist: &[u64; 256]) -> Result<Option<Vec<Option<HuffmanCode>>>> {
    let mut nodes = Vec::<HuffmanNode>::new();
    let mut heap = BinaryHeap::new();

    for (symbol, &freq) in hist.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            kind: HuffmanNodeKind::Leaf(symbol as u16),
        });
        heap.push(Reverse(HuffmanHeapEntry {
            freq,
            min_symbol: symbol as u16,
            node_index,
        }));
    }

    if heap.is_empty() {
        return Ok(None);
    }

    while heap.len() > 1 {
        let Reverse(left) = heap.pop().unwrap();
        let Reverse(right) = heap.pop().unwrap();
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            kind: HuffmanNodeKind::Branch {
                left: left.node_index,
                right: right.node_index,
            },
        });
        heap.push(Reverse(HuffmanHeapEntry {
            freq: left.freq.checked_add(right.freq).ok_or_else(|| {
                Error::InvalidArgument("Huffman frequency count overflows u64".into())
            })?,
            min_symbol: left.min_symbol.min(right.min_symbol),
            node_index,
        }));
    }

    let root = heap.pop().unwrap().0.node_index;
    let mut codes = vec![None; 256];
    if assign_huffman_codes(&nodes, root, 0, 0, &mut codes).is_err() {
        return Ok(None);
    }
    Ok(Some(codes))
}

fn assign_huffman_codes(
    nodes: &[HuffmanNode],
    node_index: usize,
    bits: u32,
    bit_len: u8,
    codes: &mut [Option<HuffmanCode>],
) -> Result<()> {
    match nodes[node_index].kind {
        HuffmanNodeKind::Leaf(symbol) => {
            let bit_len = bit_len.max(1);
            if bit_len > 31 {
                return Err(Error::InvalidArgument(
                    "Huffman code length exceeds Lerc2 limits".into(),
                ));
            }
            codes[symbol as usize] = Some(HuffmanCode { bit_len, bits });
        }
        HuffmanNodeKind::Branch { left, right } => {
            if bit_len >= 31 {
                return Err(Error::InvalidArgument(
                    "Huffman code length exceeds Lerc2 limits".into(),
                ));
            }
            assign_huffman_codes(nodes, left, bits << 1, bit_len + 1, codes)?;
            assign_huffman_codes(nodes, right, (bits << 1) | 1, bit_len + 1, codes)?;
        }
    }
    Ok(())
}

fn build_huffman_table_bytes(codes: &[Option<HuffmanCode>]) -> Result<Vec<u8>> {
    let Some(first_symbol) = codes.iter().position(Option::is_some) else {
        return Err(Error::InvalidArgument(
            "Huffman code table cannot be empty".into(),
        ));
    };
    let last_symbol = codes.iter().rposition(Option::is_some).unwrap() + 1;
    let code_lengths: Vec<u32> = codes[first_symbol..last_symbol]
        .iter()
        .map(|code| code.map_or(0, |code| code.bit_len as u32))
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&(codes.len() as i32).to_le_bytes());
    out.extend_from_slice(&(first_symbol as i32).to_le_bytes());
    out.extend_from_slice(&(last_symbol as i32).to_le_bytes());
    write_raw_bitstuff_block(&mut out, &code_lengths)?;

    let mut writer = MsbBitWriter::default();
    for code in codes[first_symbol..last_symbol].iter().flatten() {
        writer.push_bits(code.bits, code.bit_len)?;
    }
    out.extend_from_slice(&writer.into_aligned_bytes());
    Ok(out)
}

fn write_raw_bitstuff_block(out: &mut Vec<u8>, values: &[u32]) -> Result<()> {
    let max_value = values.iter().copied().max().unwrap_or(0);
    let bits = bits_required(max_value as usize);
    if bits > 31 {
        return Err(Error::InvalidArgument(
            "bit-stuffed payload exceeds the Lerc2 bit width limit".into(),
        ));
    }
    let (count_code, count_bytes) = count_field(values.len())?;
    out.push((count_code << 6) | (bits as u8));
    append_count(out, values.len(), count_bytes)?;
    if bits != 0 {
        pack_lsb_bits_into(values, bits as u8, out);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct BitstuffTile {
    offset: f64,
    offset_type: DataType,
    type_code: u8,
    payload_len: usize,
}

fn try_bitstuff_tile(
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
    if bits == 0 || bits > 31 {
        return Ok(None);
    }

    let (count_code, count_bytes) = count_field(values.len())?;
    let (type_code, offset_type) = reduce_data_type(offset, base_type)?;
    payload.clear();
    payload.reserve(1 + count_bytes + (values.len() * bits as usize).div_ceil(8));
    payload.push((count_code << 6) | bits as u8);
    append_count(payload, values.len(), count_bytes)?;
    pack_lsb_bits_into(quantized, bits as u8, payload);
    Ok(Some(BitstuffTile {
        offset,
        offset_type,
        type_code,
        payload_len: payload.len(),
    }))
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

fn pack_lsb_bits_into(values: &[u32], bits_per_value: u8, out: &mut Vec<u8>) {
    let total_bits = values.len() * bits_per_value as usize;
    let byte_len = total_bits.div_ceil(8);
    let base = out.len();
    out.resize(base + byte_len, 0);
    let mut bit_offset = 0usize;
    for &value in values {
        for bit in 0..bits_per_value {
            if ((value >> bit) & 1) != 0 {
                let byte_index = bit_offset / 8;
                let bit_index = bit_offset % 8;
                out[base + byte_index] |= 1 << bit_index;
            }
            bit_offset += 1;
        }
    }
}

fn reduce_data_type(value: f64, data_type: DataType) -> Result<(u8, DataType)> {
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
            if fits_i8(value) {
                (3, DataType::I8)
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

fn huffman_sample_value(value: f64, data_type: DataType) -> i32 {
    match data_type {
        DataType::I8 => value as i8 as i32,
        DataType::U8 => value as u8 as i32,
        _ => unreachable!("Huffman only supports 8-bit sample types"),
    }
}

fn huffman_symbol(value: i32, data_type: DataType) -> usize {
    match data_type {
        DataType::I8 => (value as i8 as i16 + 128) as usize,
        DataType::U8 => value as u8 as usize,
        _ => unreachable!("Huffman only supports 8-bit sample types"),
    }
}

fn huffman_delta_symbol(delta: i32, data_type: DataType) -> usize {
    match data_type {
        DataType::I8 => (((delta & 0xFF) as u8) as i8 as i16 + 128) as usize,
        DataType::U8 => ((delta & 0xFF) as u8) as usize,
        _ => unreachable!("Huffman only supports 8-bit sample types"),
    }
}

fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn pack_mask_bitset(mask: &[u8], pixel_count: usize) -> Result<Vec<u8>> {
    if mask.len() != pixel_count {
        return Err(Error::InvalidArgument(
            "mask length does not match the raster dimensions".into(),
        ));
    }

    let bitset_len = pixel_count.div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    for (index, &value) in mask.iter().enumerate() {
        if value != 0 {
            bitset[index >> 3] |= 1 << (7 - (index & 7));
        }
    }
    Ok(bitset)
}

fn emit_mask_literal_chunks<F>(bytes: &[u8], emit_literal: &mut F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    let mut offset = 0usize;
    while offset < bytes.len() {
        let chunk = (bytes.len() - offset).min(i16::MAX as usize);
        emit_literal(&bytes[offset..offset + chunk])?;
        offset += chunk;
    }
    Ok(())
}

enum MaskRleSegment<'a> {
    Literal(&'a [u8]),
    Repeat { value: u8, count: usize },
}

fn emit_mask_rle_segments<F>(bitset: &[u8], mut emit: F) -> Result<()>
where
    F: FnMut(MaskRleSegment<'_>) -> Result<()>,
{
    let mut literal_start = 0usize;
    let mut offset = 0usize;

    while offset < bitset.len() {
        let value = bitset[offset];
        let mut run_end = offset + 1;
        while run_end < bitset.len() && bitset[run_end] == value {
            run_end += 1;
        }

        let run_len = run_end - offset;
        let repeat_threshold = if literal_start == offset && run_end == bitset.len() {
            2
        } else if literal_start == offset || run_end == bitset.len() {
            4
        } else {
            6
        };

        if run_len >= repeat_threshold {
            emit_mask_literal_chunks(&bitset[literal_start..offset], &mut |bytes| {
                emit(MaskRleSegment::Literal(bytes))
            })?;

            let mut emitted = 0usize;
            while emitted < run_len {
                let chunk = (run_len - emitted).min(i16::MAX as usize);
                emit(MaskRleSegment::Repeat {
                    value,
                    count: chunk,
                })?;
                emitted += chunk;
            }

            literal_start = run_end;
        }

        offset = run_end;
    }

    emit_mask_literal_chunks(&bitset[literal_start..], &mut |bytes| {
        emit(MaskRleSegment::Literal(bytes))
    })
}

fn write_mask_rle(
    sink: &mut impl ByteSink,
    mask: MaskKind<'_>,
    pixel_count: usize,
    valid_pixel_count: usize,
) -> Result<()> {
    if valid_pixel_count == 0 || valid_pixel_count == pixel_count {
        return Ok(());
    }

    let MaskKind::Explicit(mask) = mask else {
        return Ok(());
    };
    let bitset = pack_mask_bitset(mask, pixel_count)?;
    emit_mask_rle_segments(&bitset, |segment| match segment {
        MaskRleSegment::Literal(bytes) => {
            write_i16(sink, bytes.len() as i16)?;
            sink.extend_from_slice(bytes)
        }
        MaskRleSegment::Repeat { value, count } => {
            write_i16(sink, -(count as i16))?;
            sink.push(value)
        }
    })?;
    write_i16(sink, i16::MIN)
}

fn explicit_mask_payload_len(
    mask: &[u8],
    pixel_count: usize,
    valid_pixel_count: usize,
) -> Result<usize> {
    if valid_pixel_count == 0 || valid_pixel_count == pixel_count {
        return Ok(0);
    }

    let bitset = pack_mask_bitset(mask, pixel_count)?;
    let mut len = 2usize;
    emit_mask_rle_segments(&bitset, |segment| {
        len = match segment {
            MaskRleSegment::Literal(bytes) => len
                .checked_add(2)
                .and_then(|len| len.checked_add(bytes.len())),
            MaskRleSegment::Repeat { .. } => len.checked_add(3),
        }
        .ok_or_else(|| Error::InvalidArgument("mask payload length overflows usize".into()))?;
        Ok(())
    })?;
    Ok(len)
}

fn depth_range_len(analysis: &RasterAnalysis) -> Result<usize> {
    if analysis.min_values.is_none() {
        return Ok(0);
    }
    (analysis.depth as usize)
        .checked_mul(2)
        .and_then(|len| len.checked_mul(analysis.data_type.byte_len()))
        .ok_or_else(|| Error::InvalidArgument("range byte count overflows usize".into()))
}

fn tile_count(width: usize, height: usize, options: EncodeOptions) -> Result<usize> {
    let micro = options.micro_block_size as usize;
    let num_blocks_x = width.div_ceil(micro);
    let num_blocks_y = height.div_ceil(micro);
    num_blocks_x
        .checked_mul(num_blocks_y)
        .ok_or_else(|| Error::InvalidArgument("tile count overflows usize".into()))
}

fn header_len(version: i32) -> usize {
    if version >= 6 {
        FIXED_HEADER_LEN_V6
    } else {
        FIXED_HEADER_LEN_V4_V5
    }
}

fn body_prefix_len(data_type: DataType, max_z_error: f64) -> usize {
    1 + usize::from(needs_huffman_flag(data_type, max_z_error))
}

fn needs_huffman_flag(data_type: DataType, max_z_error: f64) -> bool {
    matches!(data_type, DataType::I8 | DataType::U8) && (max_z_error - 0.5).abs() < 1e-5
}

fn has_per_depth_constant(analysis: &RasterAnalysis) -> bool {
    analysis
        .min_values
        .as_ref()
        .zip(analysis.max_values.as_ref())
        .map(|(mins, maxs)| mins == maxs)
        .unwrap_or(false)
}

fn write_value_as(sink: &mut impl ByteSink, value: f64, data_type: DataType) -> Result<()> {
    match data_type {
        DataType::I8 => sink.push((value as i8) as u8),
        DataType::U8 => sink.push(value as u8),
        DataType::I16 => sink.extend_from_slice(&(value as i16).to_le_bytes()),
        DataType::U16 => sink.extend_from_slice(&(value as u16).to_le_bytes()),
        DataType::I32 => sink.extend_from_slice(&(value as i32).to_le_bytes()),
        DataType::U32 => sink.extend_from_slice(&(value as u32).to_le_bytes()),
        DataType::F32 => sink.extend_from_slice(&(value as f32).to_le_bytes()),
        DataType::F64 => sink.extend_from_slice(&value.to_le_bytes()),
    }
}

fn write_u32(sink: &mut impl ByteSink, value: u32) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn write_i32(sink: &mut impl ByteSink, value: i32) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn write_i16(sink: &mut impl ByteSink, value: i16) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn write_f64(sink: &mut impl ByteSink, value: f64) -> Result<()> {
    sink.extend_from_slice(&value.to_le_bytes())
}

fn tile_header(check_code: u8, encoding: u8) -> u8 {
    ((check_code & 15) << 2) | (encoding & 3)
}

fn pixel_is_valid(mask: Option<&[u8]>, pixel: usize) -> bool {
    mask.map(|mask| mask[pixel] != 0).unwrap_or(true)
}
