use lerc_core::{DataType, PixelData};

fn load_esri_js_sanity_fixture() -> Vec<u8> {
    include_str!("data/esri_js_sanity_u8_3d.csv")
        .trim()
        .split(',')
        .map(|value| value.parse::<u8>().unwrap())
        .collect()
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
