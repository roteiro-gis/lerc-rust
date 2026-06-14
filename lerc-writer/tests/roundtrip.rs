use lerc_core::{BandLayout, BandSetView, Error, MaskView, RasterView};
use lerc_writer::{
    encode, encode_band_set, encode_band_set_into, encode_into, encoded_band_set_len_upper_bound,
    encoded_len_upper_bound, EncodeOptions,
};

fn body_offset(blob: &[u8], info: &lerc_core::BlobInfo) -> usize {
    let header_len = match info.version {
        lerc_core::Version::Lerc2(version) if version >= 6 => 90,
        lerc_core::Version::Lerc2(_) => 66,
        lerc_core::Version::Lerc1(_) => unreachable!("writer only emits Lerc2"),
    };
    let mask_num_bytes =
        u32::from_le_bytes(blob[header_len..header_len + 4].try_into().unwrap()) as usize;
    let range_len = info
        .min_values
        .as_ref()
        .map(|_| info.depth as usize * 2 * info.data_type.byte_len())
        .unwrap_or(0);
    header_len + 4 + mask_num_bytes + range_len
}

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
fn selects_one_sweep_when_tile_headers_would_dominate() {
    let pixels = vec![5u16, 9, 6, 10];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.0,
        micro_block_size: 1,
        ..EncodeOptions::default()
    };

    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(info.version, lerc_core::Version::Lerc2(4));
    assert_eq!(blob[offset], 1);
    assert_eq!(
        lerc_reader::decode(&blob).unwrap().pixels,
        lerc_core::PixelData::U16(pixels)
    );
}

#[test]
fn selects_huffman_for_repeated_lossless_u8_data() {
    let pixels: Vec<u8> = (0..256)
        .map(|index| if index % 64 < 48 { 7 } else { 9 })
        .collect();
    let raster = RasterView::new(16, 16, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 1,
        ..EncodeOptions::default()
    };

    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(info.version, lerc_core::Version::Lerc2(4));
    assert_eq!(blob[offset], 0);
    assert_ne!(blob[offset + 1], 0);
    assert_eq!(
        lerc_reader::decode(&blob).unwrap().pixels,
        lerc_core::PixelData::U8(pixels)
    );
}

#[test]
fn roundtrips_signed_huffman_i8_data() {
    let pixels: Vec<i8> = (0..256)
        .map(|index| if index % 32 < 24 { -7 } else { 11 })
        .collect();
    let raster = RasterView::new(16, 16, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 1,
        ..EncodeOptions::default()
    };

    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(info.version, lerc_core::Version::Lerc2(4));
    assert_eq!(blob[offset], 0);
    assert_ne!(blob[offset + 1], 0);
    assert_eq!(
        lerc_reader::decode(&blob).unwrap().pixels,
        lerc_core::PixelData::I8(pixels)
    );
}

#[test]
fn selects_v5_diff_tiles_for_lossless_depth_data() {
    let mut pixels = Vec::new();
    for value in 0u16..8 {
        pixels.push(value);
        pixels.push(value);
    }
    let raster = RasterView::new(4, 2, 2, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 8,
        ..EncodeOptions::default()
    };

    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(info.version, lerc_core::Version::Lerc2(5));
    assert_eq!(blob[offset], 0);
    assert_ne!(blob[offset + 8] & 4, 0);
    assert_eq!(
        lerc_reader::decode(&blob).unwrap().pixels,
        lerc_core::PixelData::U16(pixels)
    );
}

#[test]
fn roundtrips_lossless_f64_raster() {
    let pixels = vec![1.25f64, -2.5, 3.75, 4.5, -5.25, 6.0];
    let raster = RasterView::new(3, 2, 1, &pixels).unwrap();
    let blob = encode(
        raster,
        None,
        EncodeOptions {
            max_z_error: 0.0,
            micro_block_size: 1,
            ..EncodeOptions::default()
        },
    )
    .unwrap();

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::F64(pixels));
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
        ..EncodeOptions::default()
    };

    let upper_bound = encoded_len_upper_bound(raster, None, options).unwrap();
    let blob = encode(raster, None, options).unwrap();
    assert!(blob.len() <= upper_bound);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::U8(pixels));
}

#[test]
fn roundtrips_bitstuffed_tile_with_oversized_micro_block() {
    let pixels: Vec<u16> = (0..16).collect();
    let raster = RasterView::new(4, 4, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.5,
        micro_block_size: 1_000_000,
        ..EncodeOptions::default()
    };

    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(info.micro_block_size, options.micro_block_size);
    assert_eq!(blob[offset], 0);
    assert_eq!(blob[offset + 1] & 3, 1);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::U16(pixels));
}

#[test]
fn direct_band_set_apis_materialize_zero_tiles_from_writer_output() {
    let width = 16usize;
    let height = 8usize;
    let mut pixels = vec![0u8; width * height];
    for row in 0..height {
        for col in 8..width {
            pixels[row * width + col] = 7;
        }
    }

    let raster = RasterView::new(width as u32, height as u32, 1, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.0,
        micro_block_size: 8,
        ..EncodeOptions::default()
    };
    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();
    let offset = body_offset(&blob, &info);

    assert_eq!(blob[offset], 0);
    assert_eq!(blob[offset + 1] & 3, 2);
    assert_eq!(blob[offset + 2] & 3, 3);

    let (vec_info, decoded_vec) =
        lerc_reader::decode_band_set_vec::<u8>(&blob, BandLayout::Bsq).unwrap();
    assert_eq!(vec_info.band_count(), 1);
    assert_eq!(decoded_vec, pixels);

    let array = lerc_reader::decode_band_set_ndarray::<u8>(&blob).unwrap();
    assert_eq!(array.shape(), &[height, width]);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), pixels);

    let mut out = vec![99u8; pixels.len()];
    let into_info = lerc_reader::decode_band_set_into(&blob, BandLayout::Bsq, &mut out).unwrap();
    assert_eq!(into_info.band_count(), 1);
    assert_eq!(out, pixels);
}

#[test]
fn emits_v6_no_data_tiled_body_for_depth_rasters() {
    let width = 16usize;
    let height = 8usize;
    let depth = 2usize;
    let no_data = -9999.0f32;
    let mut pixels = Vec::with_capacity(width * height * depth);
    for row in 0..height {
        for col in 0..width {
            if col < 8 {
                pixels.push(row as f32);
                pixels.push(no_data);
            } else {
                pixels.push(7.0 + row as f32);
                pixels.push(9.0);
            }
        }
    }

    let raster = RasterView::new(width as u32, height as u32, depth as u32, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.0,
        micro_block_size: 8,
        no_data_value: Some(f64::from(no_data)),
    };
    let blob = encode(raster, None, options).unwrap();
    let info = lerc_reader::get_blob_info(&blob).unwrap();

    assert_eq!(info.version, lerc_core::Version::Lerc2(6));
    assert_eq!(info.no_data_value, Some(f64::from(no_data)));
    assert!(info.uses_no_data_value());
    assert_eq!(info.z_min, -1.0);
    assert_eq!(info.z_max, -1.0);
    assert_eq!(info.min_values, None);
    assert_eq!(info.max_values, None);
    assert_eq!(blob[46], 1);

    let body_offset = 90 + 4 + depth * 2 * info.data_type.byte_len();
    assert_eq!(blob[body_offset], 0);

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, lerc_core::PixelData::F32(pixels));
}

#[test]
fn upper_bound_covers_v6_no_data_output() {
    let width = 16usize;
    let height = 8usize;
    let depth = 2usize;
    let no_data = -9999.0f32;
    let mut pixels = Vec::with_capacity(width * height * depth);
    for row in 0..height {
        for col in 0..width {
            if col < 8 {
                pixels.push(row as f32);
                pixels.push(no_data);
            } else {
                pixels.push(7.0 + row as f32);
                pixels.push(9.0);
            }
        }
    }

    let raster = RasterView::new(width as u32, height as u32, depth as u32, &pixels).unwrap();
    let options = EncodeOptions {
        max_z_error: 0.0,
        micro_block_size: 8,
        no_data_value: Some(f64::from(no_data)),
    };
    let upper_bound = encoded_len_upper_bound(raster, None, options).unwrap();
    let blob = encode(raster, None, options).unwrap();
    assert!(upper_bound >= blob.len());

    let mut out = vec![0; upper_bound];
    let written = encode_into(raster, None, options, &mut out).unwrap();
    assert_eq!(written, blob.len());
    assert_eq!(&out[..written], blob.as_slice());
}

#[test]
fn band_set_upper_bound_covers_v6_no_data_output() {
    let width = 16usize;
    let height = 8usize;
    let depth = 2usize;
    let band_count = 2usize;
    let no_data = -9999.0f32;
    let mut pixels = Vec::with_capacity(width * height * depth * band_count);
    for band in 0..band_count {
        for row in 0..height {
            for col in 0..width {
                if col < 8 {
                    pixels.push(row as f32 + band as f32);
                    pixels.push(no_data);
                } else {
                    pixels.push(7.0 + row as f32 + band as f32);
                    pixels.push(9.0 + band as f32);
                }
            }
        }
    }

    let band_set = BandSetView::new(
        width as u32,
        height as u32,
        depth as u32,
        band_count,
        BandLayout::Bsq,
        &pixels,
    )
    .unwrap();
    let options = EncodeOptions {
        max_z_error: 0.0,
        micro_block_size: 8,
        no_data_value: Some(f64::from(no_data)),
    };
    let upper_bound = encoded_band_set_len_upper_bound(band_set, None, options).unwrap();
    let blob = encode_band_set(band_set, None, options).unwrap();
    assert!(upper_bound >= blob.len());

    let mut out = vec![0; upper_bound];
    let written = encode_band_set_into(band_set, None, options, &mut out).unwrap();
    assert_eq!(written, blob.len());
    assert_eq!(&out[..written], blob.as_slice());
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
        ..EncodeOptions::default()
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
fn roundtrips_shared_mask_band_set_from_interleaved_input() {
    let pixels = vec![10u8, 50, 0, 0, 11, 51, 12, 52];
    let mask_pixels = vec![1u8, 0, 1, 1];
    let band_set = BandSetView::new(2, 2, 1, 2, BandLayout::Interleaved, &pixels).unwrap();
    let mask = MaskView::new(2, 2, &mask_pixels).unwrap();

    let upper =
        encoded_band_set_len_upper_bound(band_set, Some(mask), EncodeOptions::default()).unwrap();
    let blob = encode_band_set(band_set, Some(mask), EncodeOptions::default()).unwrap();
    assert!(upper >= blob.len());

    let first_info = lerc_reader::inspect_first(&blob).unwrap();
    let second_info = lerc_reader::get_blob_info(&blob[first_info.blob_size..]).unwrap();
    assert_eq!(first_info.mask_encoding, lerc_core::MaskEncoding::Explicit);
    assert_eq!(first_info.mask_count(), 1);
    assert_eq!(second_info.mask_encoding, lerc_core::MaskEncoding::External);
    assert_eq!(second_info.mask_count(), 0);

    let decoded = lerc_reader::decode_band_set(&blob).unwrap();
    assert_eq!(decoded.info.band_count(), 2);
    assert_eq!(decoded.info.mask_count(), 1);
    assert_eq!(
        decoded.band_masks,
        vec![Some(mask_pixels.clone()), Some(mask_pixels.clone())]
    );
    assert_eq!(
        decoded.bands[0],
        lerc_core::PixelData::U8(vec![10, 0, 11, 12])
    );
    assert_eq!(
        decoded.bands[1],
        lerc_core::PixelData::U8(vec![50, 0, 51, 52])
    );
}

#[test]
fn roundtrips_shared_mask_band_set_from_bsq_input() {
    let pixels = vec![10u8, 0, 11, 12, 50, 0, 51, 52];
    let mask_pixels = vec![1u8, 0, 1, 1];
    let band_set = BandSetView::new(2, 2, 1, 2, BandLayout::Bsq, &pixels).unwrap();
    let mask = MaskView::new(2, 2, &mask_pixels).unwrap();

    let blob = encode_band_set(band_set, Some(mask), EncodeOptions::default()).unwrap();
    let decoded = lerc_reader::decode_band_set(&blob).unwrap();
    assert_eq!(decoded.info.band_count(), 2);
    assert_eq!(decoded.info.mask_count(), 1);
    assert_eq!(
        decoded.band_masks,
        vec![Some(mask_pixels.clone()), Some(mask_pixels.clone())]
    );
    assert_eq!(
        decoded.bands[0],
        lerc_core::PixelData::U8(vec![10, 0, 11, 12])
    );
    assert_eq!(
        decoded.bands[1],
        lerc_core::PixelData::U8(vec![50, 0, 51, 52])
    );
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
fn encode_band_set_into_reports_small_output_buffers() {
    let pixels = vec![10u8, 50, 0, 0, 11, 51, 12, 52];
    let mask_pixels = vec![1u8, 0, 1, 1];
    let band_set = BandSetView::new(2, 2, 1, 2, BandLayout::Interleaved, &pixels).unwrap();
    let mask = MaskView::new(2, 2, &mask_pixels).unwrap();
    let blob = encode_band_set(band_set, Some(mask), EncodeOptions::default()).unwrap();
    let mut out = vec![0u8; blob.len() - 1];

    assert!(matches!(
        encode_band_set_into(band_set, Some(mask), EncodeOptions::default(), &mut out),
        Err(Error::OutputTooSmall { .. })
    ));
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
