#![allow(missing_docs)]

use lerc_core::PixelData;

const LERC1_STUFFED_REMAINDER_TILE: &[u8] =
    include_bytes!("../corpus/headers/lerc1_stuffed_remainder_tile");

#[test]
fn preserves_lerc1_stuffed_remainder_tile_regression() {
    assert_eq!(LERC1_STUFFED_REMAINDER_TILE.len(), 89);

    let info = lerc_reader::get_blob_info(LERC1_STUFFED_REMAINDER_TILE).unwrap();
    assert_eq!((info.width, info.height), (5, 5));
    assert_eq!((info.z_min, info.z_max), (0.0, 3.0));

    let decoded = lerc_reader::decode(LERC1_STUFFED_REMAINDER_TILE).unwrap();
    assert_eq!(
        decoded.pixels,
        PixelData::F32(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, // row 0
            0.0, 0.0, 0.0, 0.0, 0.0, // row 1
            0.0, 0.0, 0.0, 0.0, 0.0, // row 2
            0.0, 0.0, 0.0, 0.0, 1.0, // row 3
            0.0, 0.0, 0.0, 2.0, 3.0, // row 4
        ])
    );
}
