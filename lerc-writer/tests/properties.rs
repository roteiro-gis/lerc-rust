#![allow(missing_docs)]

use lerc_core::{MaskView, RasterView, Sample};
use lerc_writer::{encode, EncodeOptions};
use proptest::prelude::*;

const WIDTH: u32 = 7;
const HEIGHT: u32 = 5;
const PIXEL_COUNT: usize = WIDTH as usize * HEIGHT as usize;

fn check_roundtrip<T: Sample + std::fmt::Debug>(
    depth: usize,
    values: &[T],
    validity: &[bool],
    max_z_error: f64,
) -> Result<(), TestCaseError> {
    let value_count = PIXEL_COUNT * depth;
    let values = &values[..value_count];
    let mask_data: Vec<u8> = validity.iter().map(|&valid| u8::from(valid)).collect();
    let raster = RasterView::new(WIDTH, HEIGHT, depth as u32, values).unwrap();
    let mask = MaskView::new(WIDTH, HEIGHT, &mask_data).unwrap();
    let options = EncodeOptions::new()
        .with_max_z_error(max_z_error)
        .with_micro_block_size(4);
    let blob = encode(raster, Some(mask), options).unwrap();
    let decoded = lerc_reader::decode_to_f64(&blob).unwrap();
    let tolerance = if T::IS_INTEGER { 0.0 } else { max_z_error } + 1e-9;

    prop_assert_eq!(decoded.pixels.len(), value_count);
    for (pixel, &valid) in validity.iter().enumerate().take(PIXEL_COUNT) {
        if !valid {
            continue;
        }
        for dim in 0..depth {
            let index = pixel * depth + dim;
            let expected = values[index].to_f64();
            let actual = decoded.pixels[index];
            prop_assert!(
                (actual - expected).abs() <= tolerance,
                "sample {index}: expected {expected}, got {actual}, tolerance {tolerance}"
            );
        }
    }
    Ok(())
}

macro_rules! roundtrip_property {
    ($name:ident, $sample:ty, $strategy:expr) => {
        proptest! {
            #[test]
            fn $name(
                depth in prop_oneof![Just(1usize), Just(3usize)],
                values in prop::collection::vec($strategy, PIXEL_COUNT * 3),
                validity in prop::collection::vec(any::<bool>(), PIXEL_COUNT),
                max_z_error in prop_oneof![Just(0.0), Just(0.5), Just(0.001)],
            ) {
                check_roundtrip::<$sample>(depth, &values, &validity, max_z_error)?;
            }
        }
    };
}

roundtrip_property!(roundtrips_i8, i8, any::<i8>());
roundtrip_property!(roundtrips_u8, u8, any::<u8>());
roundtrip_property!(roundtrips_i16, i16, any::<i16>());
roundtrip_property!(roundtrips_u16, u16, any::<u16>());
roundtrip_property!(roundtrips_i32, i32, any::<i32>());
roundtrip_property!(roundtrips_u32, u32, any::<u32>());
roundtrip_property!(roundtrips_f32, f32, -10_000.0f32..10_000.0f32);
roundtrip_property!(roundtrips_f64, f64, -10_000.0f64..10_000.0f64);
