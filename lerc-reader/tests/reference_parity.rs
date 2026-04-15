use ndarray::ArrayD;

#[path = "../../test-support/reference.rs"]
mod reference;

#[derive(Clone, Copy)]
enum FixtureKind {
    U8,
    F32,
    BandSetU8,
}

fn fixture(path: &str) -> std::path::PathBuf {
    reference::fixture(env!("CARGO_MANIFEST_DIR"), path)
}

fn load_blob(path: &std::path::Path) -> Vec<u8> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("csv") {
        std::fs::read_to_string(path)
            .unwrap()
            .trim()
            .split(',')
            .map(|value| value.parse::<u8>().unwrap())
            .collect()
    } else {
        std::fs::read(path).unwrap()
    }
}

#[test]
fn metadata_matches_liblerc_for_interoperability_fixtures() {
    let Some(helper) = reference::helper_path() else {
        eprintln!("skipping libLerc parity test because LERC_READER_REFERENCE_HELPER is unset");
        return;
    };

    let cases = [
        ("world.lerc1", FixtureKind::F32, 1usize),
        ("california_400_400_1_float.lerc2", FixtureKind::F32, 1usize),
        (
            "bluemarble_256_256_3_byte.lerc2",
            FixtureKind::BandSetU8,
            3usize,
        ),
        ("esri_js_sanity_u8_3d.csv", FixtureKind::U8, 1usize),
    ];

    for (relative_path, kind, expected_band_count) in cases {
        let path = fixture(relative_path);
        let blob = load_blob(&path);
        let reference_json =
            reference::run_reference_json(&helper, &["metadata", path.to_str().unwrap()]);

        match kind {
            FixtureKind::BandSetU8 => {
                let band_count = lerc_reader::get_band_count(&blob).unwrap();
                assert_eq!(band_count, expected_band_count, "{relative_path}");
                let decoded = lerc_reader::decode_band_set(&blob).unwrap();
                let first = &decoded.info.bands[0];
                assert_eq!(
                    decoded.info.bands.len(),
                    expected_band_count,
                    "{relative_path}"
                );
                assert_eq!(
                    decoded.info.bands.len() as u64,
                    reference_json["band_count"].as_u64().unwrap()
                );
                assert_eq!(
                    first.width as u64,
                    reference_json["width"].as_u64().unwrap()
                );
                assert_eq!(
                    first.height as u64,
                    reference_json["height"].as_u64().unwrap()
                );
                assert_eq!(
                    first.depth as u64,
                    reference_json["depth"].as_u64().unwrap()
                );
                assert_eq!(
                    first.data_type.code() as u64,
                    reference_json["data_type"].as_u64().unwrap()
                );
                assert_eq!(
                    decoded.info.mask_count() as u64,
                    reference_json["mask_count"].as_u64().unwrap(),
                    "{relative_path}"
                );
                assert_eq!(
                    decoded.info.uses_no_data_value(),
                    reference_json["uses_no_data_value"].as_bool().unwrap(),
                    "{relative_path}"
                );
            }
            FixtureKind::U8 | FixtureKind::F32 => {
                let info = lerc_reader::get_blob_info(&blob).unwrap();
                assert_eq!(info.width as u64, reference_json["width"].as_u64().unwrap());
                assert_eq!(
                    info.height as u64,
                    reference_json["height"].as_u64().unwrap()
                );
                assert_eq!(info.depth as u64, reference_json["depth"].as_u64().unwrap());
                assert_eq!(
                    info.data_type.code() as u64,
                    reference_json["data_type"].as_u64().unwrap()
                );
                assert_eq!(
                    info.valid_pixel_count as u64,
                    reference_json["valid_pixel_count"].as_u64().unwrap()
                );
                assert_eq!(
                    info.blob_size as u64,
                    reference_json["blob_size"].as_u64().unwrap()
                );
                assert_eq!(
                    info.z_min,
                    reference_json["z_min"].as_f64().unwrap(),
                    "{relative_path}"
                );
                assert_eq!(
                    info.z_max,
                    reference_json["z_max"].as_f64().unwrap(),
                    "{relative_path}"
                );
                assert_eq!(
                    info.max_z_error,
                    reference_json["max_z_error"].as_f64().unwrap(),
                    "{relative_path}"
                );
                assert_eq!(
                    info.mask_count() as u64,
                    reference_json["mask_count"].as_u64().unwrap(),
                    "{relative_path}"
                );
                assert_eq!(
                    info.uses_no_data_value(),
                    reference_json["uses_no_data_value"].as_bool().unwrap(),
                    "{relative_path}"
                );
            }
        }
    }
}

#[test]
fn decoded_pixels_and_masks_match_liblerc_hashes() {
    let Some(helper) = reference::helper_path() else {
        eprintln!(
            "skipping libLerc pixel parity test because LERC_READER_REFERENCE_HELPER is unset"
        );
        return;
    };

    let cases = [
        ("world.lerc1", FixtureKind::F32),
        ("california_400_400_1_float.lerc2", FixtureKind::F32),
        ("bluemarble_256_256_3_byte.lerc2", FixtureKind::BandSetU8),
        ("esri_js_sanity_u8_3d.csv", FixtureKind::U8),
    ];

    for (relative_path, kind) in cases {
        let path = fixture(relative_path);
        let blob = load_blob(&path);
        let reference_json =
            reference::run_reference_json(&helper, &["hash", path.to_str().unwrap()]);

        match kind {
            FixtureKind::U8 => {
                let raster: ArrayD<u8> = lerc_reader::decode_ndarray(&blob).unwrap();
                assert_eq!(
                    raster.shape(),
                    &json_shape(&reference_json["pixel_shape"]),
                    "{relative_path}"
                );
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_mask_hash(
                    relative_path,
                    lerc_reader::decode_mask_ndarray(&blob).unwrap(),
                    &reference_json,
                );
            }
            FixtureKind::F32 => {
                let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
                assert_eq!(
                    raster.shape(),
                    &json_shape(&reference_json["pixel_shape"]),
                    "{relative_path}"
                );
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_mask_hash(
                    relative_path,
                    lerc_reader::decode_mask_ndarray(&blob).unwrap(),
                    &reference_json,
                );
            }
            FixtureKind::BandSetU8 => {
                let raster: ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob).unwrap();
                assert_eq!(
                    raster.shape(),
                    &json_shape(&reference_json["pixel_shape"]),
                    "{relative_path}"
                );
                let (byte_len, hash) = reference::array_hash(&raster);
                assert_eq!(
                    byte_len,
                    reference_json["pixel_byte_len"].as_u64().unwrap() as usize
                );
                assert_eq!(hash, reference_json["pixel_hash"].as_str().unwrap());
                assert_mask_hash(
                    relative_path,
                    lerc_reader::decode_band_mask_ndarray(&blob).unwrap(),
                    &reference_json,
                );
            }
        }
    }
}

fn assert_mask_hash(context: &str, mask: Option<ArrayD<u8>>, reference_json: &serde_json::Value) {
    match (mask, reference_json["mask_hash"].as_str()) {
        (Some(mask), Some(expected_hash)) => {
            assert_eq!(
                mask.shape(),
                &json_shape(&reference_json["mask_shape"]),
                "{context}"
            );
            let (byte_len, hash) = reference::array_hash(&mask);
            assert_eq!(
                byte_len,
                reference_json["mask_byte_len"].as_u64().unwrap() as usize
            );
            assert_eq!(hash, expected_hash);
        }
        (None, None) => {}
        (Some(_), None) => panic!("{context}: Rust produced a mask but libLerc did not"),
        (None, Some(_)) => panic!("{context}: libLerc produced a mask but Rust did not"),
    }
}

fn json_shape(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}
