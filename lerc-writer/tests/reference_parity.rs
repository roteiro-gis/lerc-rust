use ndarray::ArrayD;

#[path = "../../test-support/reference.rs"]
mod reference;

fn write_temp_blob(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("lerc-writer-{name}-{nanos}.lerc2"));
    std::fs::write(&path, bytes).unwrap();
    path
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
    ] {
        let path = write_temp_blob(name, &blob);
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
                let raster = lerc_reader::decode_ndarray_f64(&blob).unwrap();
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
            _ => unreachable!(),
        }
        let _ = std::fs::remove_file(path);
    }
}
