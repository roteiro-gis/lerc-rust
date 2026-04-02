use lerc_core::{DataType, PixelData};
use ndarray::IxDyn;

fn load_esri_js_sanity_fixture() -> Vec<u8> {
    include_str!("data/esri_js_sanity_u8_3d.csv")
        .trim()
        .split(',')
        .map(|value| value.parse::<u8>().unwrap())
        .collect()
}

fn load_binary_fixture(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn decodes_esri_js_sanity_fixture() {
    let blob = load_esri_js_sanity_fixture();

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!(info.width, 30);
    assert_eq!(info.height, 20);
    assert_eq!(info.depth, 3);
    assert_eq!(info.data_type, DataType::U8);
    assert_eq!(info.min_values.as_deref(), Some(&[0.0, 30.0, 60.0][..]));
    assert_eq!(info.max_values.as_deref(), Some(&[29.0, 59.0, 89.0][..]));

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.mask, None);
    let pixels = match decoded.pixels {
        PixelData::U8(pixels) => pixels,
        other => panic!("expected U8 pixels, got {other:?}"),
    };
    assert_eq!(pixels.len(), 30 * 20 * 3);
    assert_eq!(&pixels[..6], &[13, 57, 68, 14, 59, 80]);
}

#[test]
fn decodes_official_world_lerc1_fixture() {
    let blob = load_binary_fixture("tests/data/world.lerc1");

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!(info.version, lerc_core::Version::Lerc1(11));
    assert_eq!(info.width, 257);
    assert_eq!(info.height, 257);
    assert_eq!(info.depth, 1);
    assert_eq!(info.data_type, DataType::F32);
    assert_eq!(info.valid_pixel_count, 65025);

    let raster = lerc_reader::decode_ndarray_f64(&blob).unwrap();
    let mask = lerc_reader::decode_mask_ndarray(&blob).unwrap().unwrap();
    assert_eq!(raster.shape(), &[257, 257]);
    assert_eq!(mask.shape(), &[257, 257]);
    assert_eq!(mask.iter().copied().map(u64::from).sum::<u64>(), 65025);
    assert_eq!(mask[IxDyn(&[0, 0])], 0);
    assert_eq!(mask[IxDyn(&[1, 1])], 1);
    assert_eq!(mask[IxDyn(&[255, 255])], 1);
    assert_eq!(raster[IxDyn(&[1, 1])], 0.0);
    assert_eq!(raster[IxDyn(&[128, 128])], 0.0);
    assert!((raster[IxDyn(&[255, 255])] - 1553.9237060546875).abs() < 1e-9);
}

#[test]
fn decodes_official_california_lerc2_fixture() {
    let blob = load_binary_fixture("tests/data/california_400_400_1_float.lerc2");

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!(info.version, lerc_core::Version::Lerc2(3));
    assert_eq!(info.width, 400);
    assert_eq!(info.height, 400);
    assert_eq!(info.depth, 1);
    assert_eq!(info.data_type, DataType::F32);
    assert_eq!(info.valid_pixel_count, 58515);

    let raster = lerc_reader::decode_ndarray_f64(&blob).unwrap();
    let mask = lerc_reader::decode_mask_ndarray(&blob).unwrap().unwrap();
    assert_eq!(raster.shape(), &[400, 400]);
    assert_eq!(mask.shape(), &[400, 400]);
    assert_eq!(mask.iter().copied().map(u64::from).sum::<u64>(), 58515);
    assert_eq!(mask[IxDyn(&[0, 0])], 0);
    assert_eq!(mask[IxDyn(&[0, 67])], 1);
    assert_eq!(mask[IxDyn(&[399, 276])], 1);
    assert!((raster[IxDyn(&[0, 67])] - 1443.2926025390625).abs() < 1e-9);
    assert!((raster[IxDyn(&[399, 276])] - 148.27206420898438).abs() < 1e-9);
}

#[test]
fn decodes_official_bluemarble_band_set_fixture() {
    let blob = load_binary_fixture("tests/data/bluemarble_256_256_3_byte.lerc2");

    assert_eq!(lerc_reader::get_band_count(&blob).unwrap(), 3);

    let raster = lerc_reader::decode_band_set_ndarray::<u8>(&blob).unwrap();
    let band_mask = lerc_reader::decode_band_mask_ndarray(&blob)
        .unwrap()
        .unwrap();
    assert_eq!(raster.shape(), &[256, 256, 3]);
    assert_eq!(band_mask.shape(), &[256, 256, 3]);

    assert_eq!(raster[IxDyn(&[0, 0, 0])], 1);
    assert_eq!(raster[IxDyn(&[0, 0, 1])], 4);
    assert_eq!(raster[IxDyn(&[0, 0, 2])], 19);
    assert_eq!(raster[IxDyn(&[128, 0, 0])], 2);
    assert_eq!(raster[IxDyn(&[128, 0, 1])], 5);
    assert_eq!(raster[IxDyn(&[128, 0, 2])], 20);
    assert_eq!(raster[IxDyn(&[255, 255, 0])], 0);
    assert_eq!(raster[IxDyn(&[255, 255, 1])], 0);
    assert_eq!(raster[IxDyn(&[255, 255, 2])], 0);

    assert_eq!(band_mask[IxDyn(&[0, 0, 0])], 1);
    assert_eq!(band_mask[IxDyn(&[0, 0, 1])], 1);
    assert_eq!(band_mask[IxDyn(&[0, 0, 2])], 1);
    assert_eq!(band_mask[IxDyn(&[255, 255, 0])], 0);
    assert_eq!(band_mask[IxDyn(&[255, 255, 1])], 0);
    assert_eq!(band_mask[IxDyn(&[255, 255, 2])], 0);
}
