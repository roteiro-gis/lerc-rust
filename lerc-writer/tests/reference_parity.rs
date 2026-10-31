#![allow(missing_docs)]

use ndarray::ArrayD;

#[path = "../../test-support/reference.rs"]
mod reference;
#[path = "../../test-support/surfaces.rs"]
mod surfaces;

const LIBLERC_HUFFMAN_FIXTURE: &[u8] =
    include_bytes!("../../testdata/interoperability/liblerc-v4-u8-huffman.lerc2");

fn json_shape(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}

fn assert_reference_decode<T>(helper: &std::path::Path, name: &str, blob: &[u8])
where
    T: lerc_reader::NdArrayElement + reference::SampleBytes,
{
    let path = reference::write_temp_bytes(&format!("liblerc-encoded-{name}"), "lerc2", blob);
    let reference_json = reference::run_reference_json(helper, &["hash", path.to_str().unwrap()]);
    let raster: ArrayD<T> = lerc_reader::decode_ndarray(blob)
        .unwrap_or_else(|err| panic!("Rust failed to decode {name}: {err}"));
    let (pixel_len, pixel_hash) = reference::array_hash(&raster);
    assert_eq!(raster.shape(), json_shape(&reference_json["pixel_shape"]));
    assert_eq!(
        pixel_len,
        reference_json["pixel_byte_len"].as_u64().unwrap() as usize
    );
    assert_eq!(pixel_hash, reference_json["pixel_hash"].as_str().unwrap());

    let mask = lerc_reader::decode_mask_ndarray(blob).unwrap();
    match mask {
        Some(mask) => {
            let (mask_len, mask_hash) = reference::array_hash(&mask);
            assert_eq!(mask.shape(), json_shape(&reference_json["mask_shape"]));
            assert_eq!(
                mask_len,
                reference_json["mask_byte_len"].as_u64().unwrap() as usize
            );
            assert_eq!(mask_hash, reference_json["mask_hash"].as_str().unwrap());
        }
        None => assert!(reference_json["mask_shape"].is_null()),
    }
    let _ = std::fs::remove_file(path);
}

fn assert_type_decodes_reference<T>(helper: &std::path::Path, name: &str)
where
    T: lerc_core::Sample + lerc_reader::NdArrayElement + reference::SampleBytes,
{
    const WIDTH: usize = 16;
    const HEIGHT: usize = 8;
    let pixels: Vec<T> = (0..WIDTH * HEIGHT)
        .map(|index| T::from_f64(((index * 17 + index / 3) % 29) as f64))
        .collect();
    let mask: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|index| u8::from(index % 5 != 0 && index % 11 != 0))
        .collect();
    let max_z_errors = if T::IS_INTEGER {
        [0.0, 1.0]
    } else {
        [0.0, 0.25]
    };

    for max_z_error in max_z_errors {
        for mask_data in [None, Some(mask.as_slice())] {
            let reference_blob = reference::encode_with_reference(
                helper,
                &pixels,
                mask_data,
                reference::ReferenceEncodeOptions {
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                    max_z_error,
                    codec_version: 4,
                    no_data_value: None,
                },
            );
            let mask_name = if mask_data.is_some() {
                "masked"
            } else {
                "all-valid"
            };
            if name == "u8" && mask_data.is_none() && max_z_error == 0.0 {
                assert_eq!(
                    reference_blob, LIBLERC_HUFFMAN_FIXTURE,
                    "pinned libLerc no longer reproduces the Huffman fixture"
                );
            }
            assert_reference_decode::<T>(
                helper,
                &format!("{name}-{mask_name}-max-z-error-{max_z_error}"),
                &reference_blob,
            );
        }
    }
}

#[test]
fn decodes_liblerc_encoded_type_matrix() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc encode parity test because the helper is unset");
        return;
    };

    assert_type_decodes_reference::<i8>(&helper, "i8");
    assert_type_decodes_reference::<u8>(&helper, "u8");
    assert_type_decodes_reference::<i16>(&helper, "i16");
    assert_type_decodes_reference::<u16>(&helper, "u16");
    assert_type_decodes_reference::<i32>(&helper, "i32");
    assert_type_decodes_reference::<u32>(&helper, "u32");
    assert_type_decodes_reference::<f32>(&helper, "f32");
    assert_type_decodes_reference::<f64>(&helper, "f64");
}

#[test]
fn decodes_liblerc_encoded_no_data() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc encode parity test because the helper is unset");
        return;
    };

    const WIDTH: usize = 16;
    const HEIGHT: usize = 8;
    const DEPTH: usize = 2;
    const NO_DATA: f32 = -9_999.0;
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT * DEPTH);
    for pixel in 0..WIDTH * HEIGHT {
        pixels.push((pixel % 31) as f32);
        pixels.push(if pixel % 7 == 0 {
            NO_DATA
        } else {
            ((pixel * 3) % 37) as f32
        });
    }
    let mask: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|index| u8::from(index % 13 != 0))
        .collect();
    let reference_blob = reference::encode_with_reference(
        &helper,
        &pixels,
        Some(&mask),
        reference::ReferenceEncodeOptions {
            width: WIDTH,
            height: HEIGHT,
            depth: DEPTH,
            max_z_error: 0.25,
            codec_version: 6,
            no_data_value: Some(f64::from(NO_DATA)),
        },
    );

    assert_reference_decode::<f32>(&helper, "f32-masked-no-data", &reference_blob);
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
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(2),
    )
    .unwrap();

    let f32_pixels = vec![
        10.0f32, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0, 15.0, 25.0,
    ];
    let f32_mask = vec![1u8, 0, 1, 1, 0, 1];
    let f32_blob = lerc_writer::encode(
        lerc_core::RasterView::new(3, 2, 2, &f32_pixels).unwrap(),
        Some(lerc_core::MaskView::new(3, 2, &f32_mask).unwrap()),
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.25)
            .with_micro_block_size(2),
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
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.0)
            .with_micro_block_size(2),
    )
    .unwrap();

    let huffman_pixels: Vec<u8> = (0..256)
        .map(|index| if index % 64 < 48 { 7 } else { 9 })
        .collect();
    let huffman_blob = lerc_writer::encode(
        lerc_core::RasterView::new(16, 16, 1, &huffman_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(2),
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
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(8),
    )
    .unwrap();

    let i8_huffman_pixels: Vec<i8> = (0..256)
        .map(|index| if index % 32 < 24 { -7 } else { 11 })
        .collect();
    let i8_huffman_blob = lerc_writer::encode(
        lerc_core::RasterView::new(16, 16, 1, &i8_huffman_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(2),
    )
    .unwrap();

    let f64_pixels = vec![1.25f64, -2.5, 3.75, 4.5, -5.25, 6.0];
    let f64_blob = lerc_writer::encode(
        lerc_core::RasterView::new(3, 2, 1, &f64_pixels).unwrap(),
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.0)
            .with_micro_block_size(2),
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
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.25)
            .with_micro_block_size(8)
            .with_no_data_value(f64::from(no_data)),
    )
    .unwrap();

    let i32_no_data = -2_000_000_000i32;
    let i32_diff_pixels = vec![
        -117_000_351,
        -91_000_273,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -87_000_261,
        -74_000_222,
        -110_000_330,
        -98_000_294,
        i32_no_data,
        i32_no_data,
    ];
    let i32_diff_mask = vec![0, 1, 1, 1, 1, 1, 1, 1];
    let i32_diff_blob = lerc_writer::encode(
        lerc_core::RasterView::new(4, 2, 2, &i32_diff_pixels).unwrap(),
        Some(lerc_core::MaskView::new(4, 2, &i32_diff_mask).unwrap()),
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(9)
            .with_no_data_value(f64::from(i32_no_data)),
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
        ("i32-v5-positive-diff", i32_diff_blob, 9u8),
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
            9 => {
                let raster: ArrayD<i32> = lerc_reader::decode_ndarray(&blob).unwrap();
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
            _ => unreachable!(),
        }
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn lossy_surfaces_decode_with_liblerc_and_are_no_larger() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc parity test because LERC_READER_REFERENCE_HELPER is unset");
        return;
    };

    for surface in surfaces::all() {
        let raster = lerc_core::RasterView::new(
            surface.width as u32,
            surface.height as u32,
            1,
            &surface.pixels,
        )
        .unwrap();
        let mask = surface.mask.as_deref().map(|mask| {
            lerc_core::MaskView::new(surface.width as u32, surface.height as u32, mask).unwrap()
        });
        let blob = lerc_writer::encode(
            raster,
            mask,
            lerc_writer::EncodeOptions::new().with_max_z_error(surface.max_z_error),
        )
        .unwrap();
        assert_reference_decode::<f32>(&helper, surface.name, &blob);

        let reference_blob = reference::encode_with_reference(
            &helper,
            &surface.pixels,
            surface.mask.as_deref(),
            reference::ReferenceEncodeOptions {
                width: surface.width,
                height: surface.height,
                depth: 1,
                max_z_error: surface.max_z_error,
                codec_version: 4,
                no_data_value: None,
            },
        );
        eprintln!(
            "{}: lerc-writer {} bytes, libLerc {} bytes",
            surface.name,
            blob.len(),
            reference_blob.len()
        );
        assert!(
            blob.len() <= reference_blob.len(),
            "{}: {} bytes, libLerc {}",
            surface.name,
            blob.len(),
            reference_blob.len()
        );
    }
}

fn assert_gdal_profile_matches<T>(
    helper: &std::path::Path,
    name: &str,
    samples: &[T],
    mask: Option<&[u8]>,
    (width, height, depth): (usize, usize, usize),
    max_z_error: f64,
) where
    T: lerc_core::Sample + reference::SampleBytes,
{
    let raster =
        lerc_core::RasterView::new(width as u32, height as u32, depth as u32, samples).unwrap();
    let mask_view =
        mask.map(|mask| lerc_core::MaskView::new(width as u32, height as u32, mask).unwrap());
    let blob = lerc_writer::encode(
        raster,
        mask_view,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(max_z_error)
            .with_profile(lerc_writer::EncoderProfile::Gdal),
    )
    .unwrap();
    let reference_blob = reference::encode_with_reference(
        helper,
        samples,
        mask,
        reference::ReferenceEncodeOptions {
            width,
            height,
            depth,
            max_z_error,
            codec_version: 4,
            no_data_value: None,
        },
    );
    // libLerc Huffman-codes lossless 8-bit data. Its code lengths depend on
    // how the C++ standard library's heap breaks frequency ties, so only the
    // decoded values are compared there.
    let huffman_candidate = T::DATA_TYPE.byte_len() == 1 && max_z_error < 1.0;
    if huffman_candidate {
        let ours = lerc_reader::decode(&blob).unwrap();
        let theirs = lerc_reader::decode(&reference_blob).unwrap();
        assert_eq!(ours.pixels, theirs.pixels, "{name}: decoded values differ");
        assert_eq!(ours.mask, theirs.mask, "{name}: decoded masks differ");
        return;
    }
    if blob != reference_blob {
        let first_difference = blob
            .iter()
            .zip(&reference_blob)
            .enumerate()
            // Skip the Fletcher-32 checksum, which differs whenever anything does.
            .position(|(offset, (left, right))| !(10..14).contains(&offset) && left != right)
            .unwrap_or(blob.len().min(reference_blob.len()));
        panic!(
            "{name} (max_z_error {max_z_error}): {} bytes vs libLerc {} bytes, first difference at byte {first_difference}",
            blob.len(),
            reference_blob.len()
        );
    }
}

fn gdal_profile_type_matrix<T>(helper: &std::path::Path, name: &str, errors: &[f64])
where
    T: lerc_core::Sample + reference::SampleBytes,
{
    const WIDTH: usize = 40;
    const HEIGHT: usize = 24;
    let ramp: Vec<T> = (0..WIDTH * HEIGHT)
        .map(|index| T::from_f64(((index * 17 + index / 3) % 59) as f64))
        .collect();
    // Plateaus with sparse spikes exercise constant, lookup-table and raw blocks.
    let plateaus: Vec<T> = (0..WIDTH * HEIGHT)
        .map(|index| {
            let (col, row) = (index % WIDTH, index / WIDTH);
            let spike = if (col * 7 + row * 3) % 23 == 0 { 90 } else { 0 };
            T::from_f64((10 + (col / 8) * 4 + (row / 8) * 2 + spike) as f64)
        })
        .collect();
    let mask: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|index| u8::from(index % 5 != 0 && index % 11 != 0))
        .collect();
    let depth3: Vec<T> = (0..WIDTH * HEIGHT * 3)
        .map(|index| T::from_f64(((index * 5 + index / 7) % 41) as f64))
        .collect();
    for &max_z_error in errors {
        for (variant, samples) in [("ramp", &ramp), ("plateaus", &plateaus)] {
            assert_gdal_profile_matches(
                helper,
                &format!("{name} {variant}"),
                samples,
                None,
                (WIDTH, HEIGHT, 1),
                max_z_error,
            );
            assert_gdal_profile_matches(
                helper,
                &format!("{name} {variant} masked"),
                samples,
                Some(&mask),
                (WIDTH, HEIGHT, 1),
                max_z_error,
            );
        }
        assert_gdal_profile_matches(
            helper,
            &format!("{name} depth 3"),
            &depth3,
            Some(&mask),
            (WIDTH, HEIGHT, 3),
            max_z_error,
        );
    }
}

#[test]
fn gdal_profile_matches_liblerc_bytes() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc parity test because LERC_READER_REFERENCE_HELPER is unset");
        return;
    };

    let integer_errors = [0.0, 0.5, 1.0, 3.0];
    gdal_profile_type_matrix::<i8>(&helper, "i8", &integer_errors);
    gdal_profile_type_matrix::<u8>(&helper, "u8", &integer_errors);
    gdal_profile_type_matrix::<i16>(&helper, "i16", &integer_errors);
    gdal_profile_type_matrix::<u16>(&helper, "u16", &integer_errors);
    gdal_profile_type_matrix::<i32>(&helper, "i32", &integer_errors);
    gdal_profile_type_matrix::<u32>(&helper, "u32", &integer_errors);
    let float_errors = [0.0, 0.001, 0.01, 0.3, 2.0];
    gdal_profile_type_matrix::<f32>(&helper, "f32", &float_errors);
    gdal_profile_type_matrix::<f64>(&helper, "f64", &float_errors);

    let huffman: Vec<u8> = (0..256)
        .map(|index| if index % 64 < 48 { 7 } else { 9 })
        .collect();
    assert_gdal_profile_matches(&helper, "u8 Huffman", &huffman, None, (16, 16, 1), 0.5);

    // A few hundred scattered noisy pixels: libLerc doubles the micro-block
    // size for the mostly empty grid, then stores the values in one sweep
    // while keeping the doubled size in the header.
    const SPARSE: usize = 256;
    let mut state = 0x2545_f491u32;
    let mut sparse_mask = vec![0u8; SPARSE * SPARSE];
    let sparse: Vec<f32> = (0..SPARSE * SPARSE)
        .map(|pixel| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            sparse_mask[pixel] = u8::from(state % 211 == 0);
            (state % 100_000) as f32 * 0.37
        })
        .collect();
    assert_gdal_profile_matches(
        &helper,
        "sparse noisy",
        &sparse,
        Some(&sparse_mask),
        (SPARSE, SPARSE, 1),
        0.0001,
    );
    let sparse_blob = lerc_writer::encode(
        lerc_core::RasterView::new(SPARSE as u32, SPARSE as u32, 1, &sparse).unwrap(),
        Some(lerc_core::MaskView::new(SPARSE as u32, SPARSE as u32, &sparse_mask).unwrap()),
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.0001)
            .with_profile(lerc_writer::EncoderProfile::Gdal),
    )
    .unwrap();
    assert_eq!(
        lerc_reader::get_blob_info(&sparse_blob)
            .unwrap()
            .micro_block_size,
        16
    );

    for surface in surfaces::all() {
        assert_gdal_profile_matches(
            &helper,
            surface.name,
            &surface.pixels,
            surface.mask.as_deref(),
            (surface.width, surface.height, 1),
            surface.max_z_error,
        );
        let wide: Vec<f64> = surface
            .pixels
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        assert_gdal_profile_matches(
            &helper,
            &format!("{} f64", surface.name),
            &wide,
            surface.mask.as_deref(),
            (surface.width, surface.height, 1),
            surface.max_z_error,
        );
    }
}
