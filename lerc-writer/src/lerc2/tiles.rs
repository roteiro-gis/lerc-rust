use lerc_core::{DataType, Error, Result, Sample};

use super::{
    bitstuff::{encode_cached_payload, reduce_data_type, try_tile},
    needs_encode_mode_flag, pixel_is_valid, tile_header, write_value_as, ByteSink, RasterAnalysis,
    RasterSource, VERSION_4, VERSION_5,
};

#[derive(Debug, Default)]
pub(super) struct TileScratch {
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

#[derive(Debug, Clone)]
pub(super) struct TilingPlan {
    pub(super) version: i32,
    pub(super) data_len: usize,
    blocks: Vec<BlockPlan>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TilingOptions {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u32,
    pub(super) data_type: DataType,
    pub(super) max_z_error: f64,
    pub(super) micro_block_size: u32,
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
        let check_code = if version >= VERSION_5 {
            check_code & 14
        } else {
            check_code & 15
        };
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
                header |= type_code << 6
            }
            BlockBody::ZeroOrEmpty | BlockBody::Raw { .. } => {}
        }
        header
    }
}

#[derive(Debug, Clone, Copy)]
struct TileGrid {
    width: usize,
    height: usize,
    depth: usize,
    micro: usize,
    blocks_x: usize,
    blocks_y: usize,
}

#[derive(Debug, Clone, Copy)]
struct TilePosition {
    origin_x: usize,
    origin_y: usize,
    width: usize,
    height: usize,
    check_code: u8,
}

impl TileGrid {
    fn new(width: u32, height: u32, depth: u32, micro_block_size: u32) -> Self {
        let width = width as usize;
        let height = height as usize;
        let micro = micro_block_size as usize;
        Self {
            width,
            height,
            depth: depth as usize,
            micro,
            blocks_x: width.div_ceil(micro),
            blocks_y: height.div_ceil(micro),
        }
    }

    fn plan_capacity(self) -> Result<usize> {
        self.blocks_x
            .checked_mul(self.blocks_y)
            .and_then(|count| count.checked_mul(self.depth))
            .ok_or(Error::SizeOverflow("tile plan block count"))
    }

    fn strip_slot_count(self) -> Result<usize> {
        self.blocks_x
            .checked_mul(self.depth)
            .ok_or(Error::SizeOverflow("tile planning strip slot count"))
    }

    fn block_capacity(self) -> Result<usize> {
        self.micro
            .checked_mul(self.micro)
            .ok_or(Error::SizeOverflow("tile planning block capacity"))
    }

    fn visit(self, mut visitor: impl FnMut(TilePosition, usize) -> Result<()>) -> Result<()> {
        for block_y in 0..self.blocks_y {
            let origin_y = block_y * self.micro;
            let block_height = (self.height - origin_y).min(self.micro);
            for block_x in 0..self.blocks_x {
                let origin_x = block_x * self.micro;
                let position = TilePosition {
                    origin_x,
                    origin_y,
                    width: (self.width - origin_x).min(self.micro),
                    height: block_height,
                    check_code: ((origin_x >> 3) as u8) & 15,
                };
                for dim in 0..self.depth {
                    visitor(position, dim)?;
                }
            }
        }
        Ok(())
    }
}

pub(super) fn plan<T: Sample, R: RasterSource<T>>(
    raster: R,
    mask: Option<&[u8]>,
    options: TilingOptions,
    mut observe_pixel: impl FnMut(usize, &[f64]) -> Result<()>,
) -> Result<TilingPlan> {
    let grid = TileGrid::new(
        options.width,
        options.height,
        options.depth,
        options.micro_block_size,
    );
    let diff_supported = supports_diff_tiles(options.data_type, options.max_z_error, options.depth);
    let mut scratch = TileScratch::default();
    let mut pixel_values = Vec::with_capacity(grid.depth);
    let mut data_len = 0usize;
    let mut blocks = Vec::with_capacity(grid.plan_capacity()?);
    let block_capacity = grid.block_capacity()?;
    let mut block_values: Vec<Vec<f64>> = (0..grid.strip_slot_count()?)
        .map(|_| Vec::with_capacity(block_capacity))
        .collect();
    let mut used_diff = false;

    for block_y in 0..grid.blocks_y {
        for values in &mut block_values {
            values.clear();
        }

        let origin_y = block_y * grid.micro;
        let block_height = (grid.height - origin_y).min(grid.micro);
        for row in 0..block_height {
            let pixel_row = origin_y + row;
            for block_x in 0..grid.blocks_x {
                let block_base = block_x * grid.depth;
                let origin_x = block_x * grid.micro;
                let block_width = (grid.width - origin_x).min(grid.micro);
                for col in 0..block_width {
                    let pixel = pixel_row * grid.width + origin_x + col;
                    if !pixel_is_valid(mask, pixel) {
                        continue;
                    }

                    pixel_values.clear();
                    for dim in 0..grid.depth {
                        let value = raster.sample(pixel, dim).to_f64();
                        block_values[block_base + dim].push(value);
                        pixel_values.push(value);
                    }
                    observe_pixel(pixel, &pixel_values)?;
                }
            }
        }

        for block_x in 0..grid.blocks_x {
            let block_base = block_x * grid.depth;
            for dim in 0..grid.depth {
                let values = &block_values[block_base + dim];
                let raw_byte_len = values
                    .len()
                    .checked_mul(options.data_type.byte_len())
                    .ok_or(Error::SizeOverflow("raw block byte count"))?;
                let absolute_plan = choose_absolute_block_plan(
                    values,
                    raw_byte_len,
                    options.data_type,
                    options.max_z_error,
                    &mut scratch.quantized,
                    &mut scratch.bitstuff_payload,
                )?;
                let mut selected_plan = absolute_plan;
                if diff_supported
                    && dim > 0
                    && build_diff_values(
                        values,
                        &block_values[block_base + dim - 1],
                        &mut scratch.diff_values_f64,
                    )?
                {
                    if let Some(diff_plan) = choose_diff_block_plan(
                        &scratch.diff_values_f64,
                        options.max_z_error,
                        &mut scratch.quantized,
                        &mut scratch.bitstuff_payload,
                    )? {
                        if diff_plan.encoded_len() < selected_plan.encoded_len() {
                            selected_plan = diff_plan;
                            used_diff = true;
                        }
                    }
                }
                data_len = data_len
                    .checked_add(selected_plan.encoded_len())
                    .ok_or(Error::SizeOverflow("tile payload byte count"))?;
                blocks.push(selected_plan);
            }
        }
    }

    Ok(TilingPlan {
        version: if used_diff { VERSION_5 } else { VERSION_4 },
        data_len,
        blocks,
    })
}

pub(super) fn write<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    scratch: &mut TileScratch,
    raster: R,
    mask: Option<&[u8]>,
    analysis: &RasterAnalysis,
    version: i32,
    plan: &TilingPlan,
) -> Result<()> {
    sink.push(0)?;
    if needs_encode_mode_flag(analysis.data_type, analysis.max_z_error, version) {
        sink.push(0)?;
    }

    let grid = TileGrid::new(
        analysis.width,
        analysis.height,
        analysis.depth,
        analysis.micro_block_size,
    );
    let mut block_plan_index = 0usize;
    grid.visit(|position, dim| {
        let block_plan = plan
            .blocks
            .get(block_plan_index)
            .ok_or(Error::Internal("cached tile plan is missing a block"))?;
        block_plan_index += 1;
        collect_block_values(
            scratch,
            raster,
            mask,
            grid.width,
            position,
            dim,
            block_plan.is_diff,
        );
        prepare_cached_block_payload(
            block_plan,
            &scratch.values_f64,
            &scratch.prev_values_f64,
            analysis.max_z_error,
            &mut scratch.quantized,
            &mut scratch.bitstuff_payload,
        )?;
        write_block_plan(
            sink,
            block_plan,
            position.check_code,
            version,
            &scratch.raw_bytes,
            &scratch.bitstuff_payload,
        )
    })?;

    if block_plan_index != plan.blocks.len() {
        return Err(Error::Internal("cached tile plan contains trailing blocks"));
    }
    Ok(())
}

fn collect_block_values<T: Sample, R: RasterSource<T>>(
    scratch: &mut TileScratch,
    raster: R,
    mask: Option<&[u8]>,
    raster_width: usize,
    position: TilePosition,
    dim: usize,
    include_prev_values: bool,
) {
    scratch.clear();
    for row in 0..position.height {
        let pixel_row = position.origin_y + row;
        for col in 0..position.width {
            let pixel = pixel_row * raster_width + position.origin_x + col;
            if !pixel_is_valid(mask, pixel) {
                continue;
            }
            let value = raster.sample(pixel, dim);
            value.write_le(&mut scratch.raw_bytes);
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
        .ok_or(Error::SizeOverflow("raw block byte count"))?;
    choose_block_plan(
        values,
        Some(raw_len),
        base_type,
        max_z_error,
        false,
        quantized,
        payload,
    )?
    .ok_or(Error::Internal("absolute tile plan unexpectedly missing"))
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

#[allow(clippy::too_many_arguments)]
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

    if let Some(bitstuff) = try_tile(values, min, max, max_z_error, base_type, quantized, payload)?
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
        if match raw_len {
            Some(raw_len) => plan.encoded_len() < raw_len,
            None => true,
        } {
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
        BlockBody::Raw { byte_len } => {
            if raw_bytes.len() != byte_len {
                return Err(Error::Internal(
                    "raw tile payload length changed after planning",
                ));
            }
            sink.extend_from_slice(raw_bytes)
        }
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
            let payload = bitstuff_payload.get(..payload_len).ok_or(Error::Internal(
                "cached bit-stuffed tile payload is shorter than planned",
            ))?;
            sink.extend_from_slice(payload)
        }
    }
}

fn prepare_cached_block_payload(
    plan: &BlockPlan,
    values: &[f64],
    prev_values: &[f64],
    max_z_error: f64,
    quantized: &mut Vec<u32>,
    payload: &mut Vec<u8>,
) -> Result<()> {
    payload.clear();
    if let BlockBody::Bitstuff {
        offset,
        payload_len,
        ..
    } = &plan.body
    {
        if plan.is_diff {
            if values.len() != prev_values.len() {
                return Err(Error::Internal("diff input lengths do not match"));
            }
            encode_cached_payload(
                values
                    .iter()
                    .zip(prev_values)
                    .map(|(&value, &previous)| value - previous),
                values.len(),
                *offset,
                max_z_error,
                quantized,
                payload,
            )?;
        } else {
            encode_cached_payload(
                values.iter().copied(),
                values.len(),
                *offset,
                max_z_error,
                quantized,
                payload,
            )?;
        }
        if payload.len() != *payload_len {
            return Err(Error::Internal(
                "cached bit-stuffed tile payload length changed",
            ));
        }
    }
    Ok(())
}

fn build_diff_values(current: &[f64], previous: &[f64], out: &mut Vec<f64>) -> Result<bool> {
    if current.len() != previous.len() {
        return Err(Error::Internal("diff input lengths do not match"));
    }
    out.clear();
    out.reserve(current.len());
    for (&value, &previous) in current.iter().zip(previous) {
        let diff = value - previous;
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
    depth > 1 && data_type.is_integer() && (max_z_error - 0.5).abs() < 1e-5
}
