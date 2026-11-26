#![allow(missing_docs)]

use lerc_core::{MaskView, RasterView};
use lerc_writer::{encode, EncodeOptions, EncoderProfile};

#[test]
fn raised_grid_preserves_requested_f64_error_bound() {
    // Multiplication by 10 rounds these values onto apparent grid points.
    // The raised 0.05 bound must not replace the requested 0.001 limit when
    // checking reconstruction. Before the fix, the error reached one f64 ULP.
    for profile in [EncoderProfile::Compact, EncoderProfile::Gdal] {
        for exponent in [43, 47] {
            for sign in [-1.0, 1.0] {
                for masked in [false, true] {
                    let pixels: Vec<f64> = (0..256)
                        .map(|i| sign * (2f64.powi(exponent) + (i % 31) as f64 * 0.1))
                        .collect();
                    let validity: Vec<u8> = (0..256).map(|i| u8::from(i % 7 != 0)).collect();
                    let mask = masked.then(|| MaskView::new(16, 16, &validity).unwrap());
                    let blob = encode(
                        RasterView::new(16, 16, 1, &pixels).unwrap(),
                        mask,
                        EncodeOptions::new()
                            .with_max_z_error(0.001)
                            .with_profile(profile),
                    )
                    .unwrap();
                    assert_eq!(lerc_reader::get_blob_info(&blob).unwrap().max_z_error, 0.05);
                    let decoded = lerc_reader::decode_to_f64(&blob).unwrap();
                    for (i, (&source, &actual)) in pixels.iter().zip(&decoded.pixels).enumerate() {
                        if !masked || validity[i] != 0 {
                            assert!(
                                (source - actual).abs() <= 0.001,
                                "{profile:?}, exponent {exponent}, sign {sign}, masked {masked}, sample {i}: {source} -> {actual}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn raised_grid_keeps_safe_blocks_compressed_alongside_raw_fallbacks() {
    for profile in [EncoderProfile::Compact, EncoderProfile::Gdal] {
        // Alternate columns of 8x8 blocks needing raw fallback and blocks
        // whose integer spacing is safely quantized on the raised grid.
        let pixels: Vec<f64> = (0..32 * 16)
            .map(|i| {
                let col = i % 32;
                let delta = if col / 8 % 2 == 0 {
                    (i % 31) as f64 * 0.1
                } else {
                    (i % 31) as f64
                };
                2f64.powi(43) + delta
            })
            .collect();
        let blob = encode(
            RasterView::new(32, 16, 1, &pixels).unwrap(),
            None,
            EncodeOptions::new()
                .with_max_z_error(0.001)
                .with_profile(profile),
        )
        .unwrap();
        assert_eq!(lerc_reader::get_blob_info(&blob).unwrap().max_z_error, 0.05);
        // A whole-raster one-sweep fallback would be larger than the raw samples.
        assert!(blob.len() < pixels.len() * std::mem::size_of::<f64>());
        let decoded = lerc_reader::decode_to_f64(&blob).unwrap();
        for (&source, &actual) in pixels.iter().zip(&decoded.pixels) {
            assert!(
                (source - actual).abs() <= 0.001,
                "{profile:?}: {source} -> {actual}"
            );
        }
    }
}

#[test]
fn constant_blocks_validate_the_original_bound_after_a_grid_raise() {
    // The grid scan accepts both endpoints after multiplication rounds their
    // residuals down. Their actual range is slightly greater than 1e-8, so
    // storing the minimum as a constant would exceed the caller's limit.
    for profile in [EncoderProfile::Compact, EncoderProfile::Gdal] {
        let pixels: Vec<f64> = (0..256)
            .map(|i| {
                if i % 2 == 0 {
                    15.999_999_995
                } else {
                    16.000_000_005
                }
            })
            .collect();
        let blob = encode(
            RasterView::new(16, 16, 1, &pixels).unwrap(),
            None,
            EncodeOptions::new()
                .with_max_z_error(1e-8)
                .with_profile(profile),
        )
        .unwrap();
        assert_eq!(
            lerc_reader::get_blob_info(&blob).unwrap().max_z_error,
            0.005
        );
        let decoded = lerc_reader::decode_to_f64(&blob).unwrap();
        for (&source, &actual) in pixels.iter().zip(&decoded.pixels) {
            assert!(
                (source - actual).abs() <= 1e-8,
                "{profile:?}: {source} -> {actual}"
            );
        }
    }
}

/// One 8 x 8 Float64 block: a far-off value (-9999) next to values on a 0.01
/// grid. libLerc bit-stuffs it, and 100.29 decodes as 100.30000000000109,
/// 1.09e-12 past `max_z_error`; the checked bound stores the block raw.
fn far_off_value_block() -> Vec<f64> {
    let mut values = vec![100.29f64; 64];
    values[0] = -9999.0;
    values[63] = 101.0;
    values
}

fn encode_far_off_block(error_bound: lerc_writer::ErrorBound) -> Vec<u8> {
    lerc_writer::encode(
        lerc_core::RasterView::new(8, 8, 1, &far_off_value_block()).unwrap(),
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.01)
            .with_profile(lerc_writer::EncoderProfile::Gdal)
            .with_error_bound(error_bound),
    )
    .unwrap()
}

fn max_error(blob: &[u8], values: &[f64]) -> f64 {
    let decoded = lerc_reader::decode_to_f64(blob).unwrap();
    decoded
        .pixels
        .iter()
        .zip(values)
        .map(|(decoded, value)| (decoded - value).abs())
        .fold(0.0, f64::max)
}

#[test]
fn unchecked_error_bound_bit_stuffs_blocks_the_checked_bound_stores_raw() {
    let values = far_off_value_block();
    let checked = encode_far_off_block(lerc_writer::ErrorBound::Checked);
    let unchecked = encode_far_off_block(lerc_writer::ErrorBound::Unchecked);
    // Raw storage: 64 doubles behind the block header.
    assert_eq!(checked.len(), 599);
    assert!(max_error(&checked, &values) <= 0.01);
    // libLerc's 114-byte blob, which overshoots by f64 rounding only.
    assert_eq!(unchecked.len(), 114);
    let error = max_error(&unchecked, &values);
    assert!(error > 0.01 && error <= 0.01 + 2e-12, "{error}");
}

#[test]
fn unchecked_error_bound_requires_the_gdal_profile() {
    let values = far_off_value_block();
    let result = lerc_writer::encode(
        lerc_core::RasterView::new(8, 8, 1, &values).unwrap(),
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.01)
            .with_error_bound(lerc_writer::ErrorBound::Unchecked),
    );
    assert!(matches!(result, Err(lerc_core::Error::InvalidArgument(_))));
    assert_eq!(
        lerc_writer::EncodeOptions::new().error_bound,
        lerc_writer::ErrorBound::Checked
    );
}

/// #50's large-magnitude case, `2^47 + k * 0.1` at a requested 0.001: the
/// header bound is raised to the 0.05 grid. Checked, every block that would
/// miss the request is stored raw. Unchecked, the encoder enforces nothing and
/// the samples decode up to one ulp of 2^47 (0.03125) off, 31 times the
/// request, as libLerc decodes its own blob of the same input.
#[test]
fn unchecked_error_bound_gives_up_the_bound_by_an_ulp_of_the_magnitude() {
    let values: Vec<f64> = (0..32 * 16)
        .map(|i| {
            let step = if (i % 32) / 8 % 2 == 0 { 0.1 } else { 1.0 };
            2f64.powi(47) + (i % 31) as f64 * step
        })
        .collect();
    let encode = |error_bound| {
        lerc_writer::encode(
            lerc_core::RasterView::new(32, 16, 1, &values).unwrap(),
            None,
            lerc_writer::EncodeOptions::new()
                .with_max_z_error(0.001)
                .with_profile(lerc_writer::EncoderProfile::Gdal)
                .with_error_bound(error_bound),
        )
        .unwrap()
    };
    let checked = encode(lerc_writer::ErrorBound::Checked);
    let unchecked = encode(lerc_writer::ErrorBound::Unchecked);
    assert!(max_error(&checked, &values) <= 0.001);
    let ulp = 2f64.powi(47 - 52);
    assert_eq!(ulp, 0.03125);
    assert_eq!(max_error(&unchecked, &values), ulp);
    assert!(unchecked.len() < checked.len());
}
