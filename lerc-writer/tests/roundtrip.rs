use lerc_core::{Error, MaskView, RasterView};
use lerc_writer::{encode, encode_into, encoded_len_upper_bound, EncodeOptions};

#[test]
fn roundtrips_constant_u16_raster() {
    let pixels = vec![9u16; 6];
    let raster = RasterView::new(3, 2, 1, &pixels).unwrap();
    let blob = encode(raster, None, EncodeOptions::default()).unwrap();

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!(info.version, lerc_core::Version::Lerc2(4));
    assert_eq!(info.blob_size, blob.len());
    assert_eq!(info.width, 3);
    assert_eq!(info.height, 2);
    assert_eq!(info.depth, 1);
    assert_eq!(info.valid_pixel_count, 6);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::U16(pixels));
    assert_eq!(decoded.mask, None);
}

#[test]
fn encoded_len_upper_bound_is_conservative() {
    let pixels = vec![9u16; 6];
    let raster = RasterView::new(3, 2, 1, &pixels).unwrap();
    let upper = encoded_len_upper_bound(raster, None, EncodeOptions::default()).unwrap();
    let blob = encode(raster, None, EncodeOptions::default()).unwrap();

    assert!(upper >= blob.len());
    assert!(upper > blob.len());
}

#[test]
fn roundtrips_bitstuffed_u8_tiles_exactly() {
    let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let raster = RasterView::new(4, 2, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 2,
    };

    let upper_bound = encoded_len_upper_bound(raster, None, options).unwrap();
    let blob = encode(raster, None, options).unwrap();
    assert!(blob.len() <= upper_bound);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::U8(pixels));
}

#[test]
fn roundtrips_masked_f32_raster_with_depth() {
    let pixels = vec![
        10.0f32, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0, 15.0, 25.0,
    ];
    let mask = vec![1u8, 0, 1, 1, 0, 1];
    let raster = RasterView::new(3, 2, 2, &pixels).unwrap();
    let mask = MaskView::new(3, 2, &mask).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.25,
        micro_block_size: 2,
    };

    let blob = encode(raster, Some(mask), options).unwrap();
    let decoded = lerc_reader::decode(&blob).unwrap();
    match decoded.pixels {
        lerc_core::PixelData::F32(values) => {
            assert_eq!(values.len(), pixels.len());
            for (pixel, (&m, chunk)) in [1u8, 0, 1, 1, 0, 1]
                .iter()
                .zip(values.chunks_exact(2))
                .enumerate()
            {
                let expected = &pixels[pixel * 2..pixel * 2 + 2];
                if m != 0 {
                    for (&actual, &expected) in chunk.iter().zip(expected) {
                        assert!((actual - expected).abs() <= options.max_z_error as f32);
                    }
                } else {
                    assert_eq!(chunk, &[0.0, 0.0]);
                }
            }
        }
        other => panic!("expected f32 pixels, got {other:?}"),
    }
    assert_eq!(decoded.mask, Some(vec![1, 0, 1, 1, 0, 1]));
    assert_eq!(decoded.info.min_values.as_deref(), Some(&[10.0, 20.0][..]));
    assert_eq!(decoded.info.max_values.as_deref(), Some(&[15.0, 25.0][..]));
}

#[test]
fn compresses_repeated_mask_bytes() {
    let pixels = vec![7u8; 256];
    let mask_pixels: Vec<u8> = (0..256).map(|index| u8::from(index >= 128)).collect();
    let raster = RasterView::new(256, 1, 1, &pixels).unwrap();
    let mask = MaskView::new(256, 1, &mask_pixels).unwrap();

    let blob = encode(raster, Some(mask), EncodeOptions::default()).unwrap();
    let mask_num_bytes = u32::from_le_bytes(blob[66..70].try_into().unwrap()) as usize;
    let literal_mask_num_bytes = 256usize.div_ceil(8) + 4;

    assert_eq!(mask_num_bytes, 8);
    assert!(mask_num_bytes < literal_mask_num_bytes);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.mask, Some(mask_pixels.clone()));
    assert_eq!(
        decoded.pixels,
        lerc_core::PixelData::U8(
            mask_pixels
                .iter()
                .map(|&value| if value != 0 { 7 } else { 0 })
                .collect()
        )
    );
}

#[test]
fn emits_per_depth_constant_blob_without_tile_payload() {
    let pixels = vec![10u8, 20, 10, 20, 10, 20, 10, 20];
    let raster = RasterView::new(2, 2, 2, &pixels).unwrap();
    let blob = encode(raster, None, EncodeOptions::default()).unwrap();

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::U8(pixels));
    assert_eq!(decoded.info.min_values.as_deref(), Some(&[10.0, 20.0][..]));
    assert_eq!(decoded.info.max_values.as_deref(), Some(&[10.0, 20.0][..]));
}

#[test]
fn encode_into_reports_small_output_buffers() {
    let pixels = vec![1u8, 2, 3, 4];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();
    let blob = encode(raster, None, EncodeOptions::default()).unwrap();
    let mut out = vec![0u8; blob.len() - 1];

    assert!(matches!(
        encode_into(raster, None, EncodeOptions::default(), &mut out),
        Err(Error::OutputTooSmall { .. })
    ));
}

#[test]
fn rejects_non_finite_valid_samples() {
    let pixels = vec![1.0f32, f32::NAN, 3.0, 4.0];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();

    assert!(matches!(
        encode(raster, None, EncodeOptions::default()),
        Err(Error::InvalidArgument(_))
    ));
}
