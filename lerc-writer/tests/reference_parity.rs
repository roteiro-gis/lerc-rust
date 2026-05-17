use ndarray::ArrayD;

#[path = "../../test-support/reference.rs"]
mod reference;

fn json_shape(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn generated_blobs_match_liblerc_decode_hashes() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc parity test because LERC_READER_REFERENCE_HELPER is unset");
        return;
    };

    let u8_pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let u8_blob = lerc_writer::encode(
        lerc_core::RasterView::new(4, 2, 1, &u8_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.5,
            micro_block_size: 2,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let f32_pixels = vec![
        10.0f32, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0, 15.0, 25.0,
    ];
    let f32_mask = vec![1u8, 0, 1, 1, 0, 1];
    let f32_blob = lerc_writer::encode(
        lerc_core::RasterView::new(3, 2, 2, &f32_pixels).unwrap(),
        Some(lerc_core::MaskView::new(3, 2, &f32_mask).unwrap()),
        lerc_writer::EncodeOptions {
            max_z_error: 0.25,
            micro_block_size: 2,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let band_set_pixels = vec![10u8, 50, 0, 0, 11, 51, 12, 52];
    let band_set_mask = vec![1u8, 0, 1, 1];
    let band_set_blob = lerc_writer::encode_band_set(
        lerc_core::BandSetView::new(
            2,
            2,
            1,
            2,
            lerc_core::BandLayout::Interleaved,
            &band_set_pixels,
        )
        .unwrap(),
        Some(lerc_core::MaskView::new(2, 2, &band_set_mask).unwrap()),
        lerc_writer::EncodeOptions::default(),
    )
    .unwrap();

    let one_sweep_pixels = vec![5u16, 9, 6, 10];
    let one_sweep_blob = lerc_writer::encode(
        lerc_core::RasterView::new(2, 2, 1, &one_sweep_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.0,
            micro_block_size: 1,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let huffman_pixels: Vec<u8> = (0..256)
        .map(|index| if index % 64 < 48 { 7 } else { 9 })
        .collect();
    let huffman_blob = lerc_writer::encode(
        lerc_core::RasterView::new(16, 16, 1, &huffman_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.5,
            micro_block_size: 1,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let mut diff_pixels = Vec::new();
    for value in 0u16..8 {
        diff_pixels.push(value);
        diff_pixels.push(value);
    }
    let diff_blob = lerc_writer::encode(
        lerc_core::RasterView::new(4, 2, 2, &diff_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.5,
            micro_block_size: 8,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let i8_huffman_pixels: Vec<i8> = (0..256)
        .map(|index| if index % 32 < 24 { -7 } else { 11 })
        .collect();
    let i8_huffman_blob = lerc_writer::encode(
        lerc_core::RasterView::new(16, 16, 1, &i8_huffman_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.5,
            micro_block_size: 1,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let f64_pixels = vec![1.25f64, -2.5, 3.75, 4.5, -5.25, 6.0];
    let f64_blob = lerc_writer::encode(
        lerc_core::RasterView::new(3, 2, 1, &f64_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.0,
            micro_block_size: 1,
            ..lerc_writer::EncodeOptions::default()
        },
    )
    .unwrap();

    let no_data = -9999.0f32;
    let mut no_data_pixels = Vec::new();
    for row in 0..8 {
        for col in 0..16 {
            if col < 8 {
                no_data_pixels.push(row as f32);
                no_data_pixels.push(no_data);
            } else {
                no_data_pixels.push(7.0 + row as f32);
                no_data_pixels.push(9.0);
            }
        }
    }
    let no_data_blob = lerc_writer::encode(
        lerc_core::RasterView::new(16, 8, 2, &no_data_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions {
            max_z_error: 0.0,
            micro_block_size: 8,
            no_data_value: Some(f64::from(no_data)),
        },
    )
    .unwrap();

    for (name, blob, kind) in [
        ("u8-bitstuff", u8_blob, 0u8),
        ("f32-depth-mask", f32_blob, 1u8),
        ("u8-band-set-shared-mask", band_set_blob, 2u8),
        ("u16-one-sweep", one_sweep_blob, 3u8),
        ("u8-huffman", huffman_blob, 4u8),
        ("u16-v5-diff", diff_blob, 5u8),
        ("i8-huffman", i8_huffman_blob, 6u8),
        ("f64-lossless", f64_blob, 7u8),
        ("f32-v6-no-data", no_data_blob, 8u8),
    ] {
        let path = reference::write_temp_bytes(&format!("lerc-writer-{name}"), "lerc2", &blob);
        let reference_json =
            reference::run_reference_json(&helper, &["hash", path.to_str().unwrap()]);
        match kind {
            0 => {
                let raster: ArrayD<u8> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            1 => {
                let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
                let mask = lerc_reader::decode_mask_ndarray(&blob).unwrap().unwrap();
                let (pixel_len, pixel_hash) = reference::array_hash(&raster);
                let (mask_len, mask_hash) = reference::array_hash(&mask);
                assert_eq!(
                    pixel_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(pixel_hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_eq!(
                    mask_len,
                    reference_json["mask_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(mask_hash, reference_json["mask_hash"].as_str().unwrap());
            }
            2 => {
                let raster = lerc_reader::decode_band_set_ndarray_with_layout::<u8>(
                    &blob,
                    lerc_core::BandLayout::Interleaved,
                )
                .unwrap();
                let mask = lerc_reader::decode_band_mask_ndarray(&blob)
                    .unwrap()
                    .unwrap();
                let (pixel_len, pixel_hash) = reference::array_hash(&raster);
                let (mask_len, mask_hash) = reference::array_hash(&mask);
                assert_eq!(
                    pixel_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(pixel_hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_eq!(
                    mask_len,
                    reference_json["mask_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(mask_hash, reference_json["mask_hash"].as_str().unwrap());
            }
            3 => {
                let raster: ArrayD<u16> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            4 => {
                let raster: ArrayD<u8> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            5 => {
                let raster: ArrayD<u16> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            6 => {
                let raster: ArrayD<i8> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            7 => {
                let raster: ArrayD<f64> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
            }
            8 => {
                let metadata_json =
                    reference::run_reference_json(&helper, &["metadata", path.to_str().unwrap()]);
                let info = lerc_reader::get_blob_info(&blob).unwrap();
                assert_eq!(info.version, lerc_core::Version::Lerc2(6));
                assert_eq!(info.width as u64, metadata_json["width"].as_u64().unwrap());
                assert_eq!(
                    info.height as u64,
                    metadata_json["height"].as_u64().unwrap()
                );
                assert_eq!(info.depth as u64, metadata_json["depth"].as_u64().unwrap());
                assert_eq!(
                    info.data_type.code() as u64,
                    metadata_json["data_type"].as_u64().unwrap()
                );
                assert_eq!(
                    info.valid_pixel_count as u64,
                    metadata_json["valid_pixel_count"].as_u64().unwrap()
                );
                assert_eq!(
                    info.mask_count() as u64,
                    metadata_json["mask_count"].as_u64().unwrap()
                );
                assert_eq!(
                    info.uses_no_data_value(),
                    metadata_json["uses_no_data_value"].as_bool().unwrap()
                );
                assert_eq!(info.no_data_value, Some(f64::from(no_data)));

                let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    raster.shape(),
                    &json_shape(&reference_json["pixel_shape"]),
                    "{name}"
                );
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_eq!(reference_json["mask_hash"], serde_json::Value::Null);
            }
            _ => unreachable!(),
        }
        let _ = std::fs::remove_file(path);
    }
}
