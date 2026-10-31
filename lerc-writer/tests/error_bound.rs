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
