#![allow(missing_docs)]

//! Size and error-bound regressions for lossy tiled encoding on synthetic
//! surfaces. Every surface is deterministic, so the pinned sizes are exact.

use lerc_core::{MaskView, PixelData, RasterView};
use lerc_writer::{encode, EncodeOptions};

#[path = "../../test-support/surfaces.rs"]
mod surfaces;

use surfaces::Surface;

fn encode_surface(surface: &Surface) -> Vec<u8> {
    let raster = RasterView::new(
        surface.width as u32,
        surface.height as u32,
        1,
        &surface.pixels,
    )
    .unwrap();
    let mask = surface
        .mask
        .as_deref()
        .map(|mask| MaskView::new(surface.width as u32, surface.height as u32, mask).unwrap());
    encode(
        raster,
        mask,
        EncodeOptions::new().with_max_z_error(surface.max_z_error),
    )
    .unwrap()
}

/// Checks the decoded values against the source.
///
/// The decoder rebuilds values in `f64` and rounds them to `f32`, which can
/// add half an `f32` ulp of the sample magnitude on top of `max_z_error`.
fn assert_within_bound(surface: &Surface, blob: &[u8]) {
    let decoded = lerc_reader::decode(blob).unwrap();
    assert_eq!(decoded.mask.as_deref(), surface.mask.as_deref());
    let PixelData::F32(values) = decoded.pixels else {
        panic!("{}: unexpected pixel type", surface.name);
    };
    let magnitude = surface
        .pixels
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let rounding = f64::from(magnitude) * f64::from(f32::EPSILON);
    let error = surface
        .pixels
        .iter()
        .zip(&values)
        .enumerate()
        .filter(|(pixel, _)| surface.mask.as_ref().map_or(true, |mask| mask[*pixel] != 0))
        .map(|(_, (&source, &decoded))| (f64::from(decoded) - f64::from(source)).abs())
        .fold(0.0, f64::max);
    assert!(
        error <= surface.max_z_error + rounding,
        "{}: max error {error} exceeds {} (+{rounding} f32 rounding)",
        surface.name,
        surface.max_z_error
    );
}

#[test]
fn every_surface_round_trips_within_max_z_error() {
    for surface in surfaces::all() {
        assert_within_bound(&surface, &encode_surface(&surface));
    }
}

#[test]
fn blocks_within_max_z_error_of_their_minimum_are_constants() {
    // Treating these as bit-stuffed blocks with zero bits per value used to
    // fall back to raw storage, four bytes per pixel.
    let surface = surfaces::near_constant_blocks();
    let blob = encode_surface(&surface);
    // Header, mask count, ranges, flag and one 5-byte constant per 8x8 block.
    assert!(
        blob.len() <= 66 + 4 + 8 + 1 + 64 * 5,
        "{} bytes",
        blob.len()
    );
}

#[test]
fn blocks_with_few_distinct_levels_use_lookup_tables() {
    // Simple bit-stuffing needs 9 bits per value; the lookup table needs 2.
    let surface = surfaces::few_levels();
    let blob = encode_surface(&surface);
    let simple_bitstuffed_body = 16 * (1 + 1 + 1 + 1 + 64 * 9 / 8);
    assert!(
        blob.len() < 66 + 4 + 8 + 1 + simple_bitstuffed_body / 2,
        "{} bytes",
        blob.len()
    );
}

#[test]
fn values_on_a_coarser_decimal_grid_raise_the_header_error() {
    // Centimetre values need no steps finer than the centimetre grid, so the
    // stored bound rises to 0.005 while values stay within the requested 0.001.
    let surface = surfaces::centimetre_grid();
    let blob = encode_surface(&surface);
    assert_eq!(
        lerc_reader::get_blob_info(&blob).unwrap().max_z_error,
        0.005
    );
    assert_within_bound(&surface, &blob);
}

#[test]
fn values_off_any_coarser_grid_keep_the_requested_error() {
    let surface = surfaces::smooth();
    let blob = encode_surface(&surface);
    assert_eq!(
        lerc_reader::get_blob_info(&blob).unwrap().max_z_error,
        0.001
    );
}

#[test]
fn low_bit_rate_surfaces_use_larger_micro_blocks() {
    let surface = surfaces::terraces();
    let blob = encode_surface(&surface);
    assert_eq!(
        lerc_reader::get_blob_info(&blob).unwrap().micro_block_size,
        16
    );
}

#[test]
fn synthetic_surface_sizes_do_not_regress() {
    // Sizes at the time the block planner adopted libLerc's decisions; each is
    // at or below libLerc 4.1.1's output for the same surface.
    for (surface, expected) in surfaces::all().iter().zip([
        SIZE_NEAR_CONSTANT,
        SIZE_FEW_LEVELS,
        SIZE_CENTIMETRE_GRID,
        SIZE_SMOOTH,
        SIZE_TERRACES,
        SIZE_MASKED_DEPTH,
    ]) {
        let len = encode_surface(surface).len();
        assert!(
            len <= expected,
            "{}: {len} bytes exceeds {expected}",
            surface.name
        );
    }
}

const SIZE_NEAR_CONSTANT: usize = 399;
const SIZE_FEW_LEVELS: usize = 511;
const SIZE_CENTIMETRE_GRID: usize = 2_749;
const SIZE_SMOOTH: usize = 3_403;
const SIZE_TERRACES: usize = 399;
const SIZE_MASKED_DEPTH: usize = 34_523;
