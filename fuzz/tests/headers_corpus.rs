#![allow(missing_docs)]

use lerc_core::PixelData;

const LERC1_STUFFED_REMAINDER_TILE: &[u8] =
    include_bytes!("../corpus/headers/lerc1_stuffed_remainder_tile");

#[test]
fn preserves_lerc1_stuffed_remainder_tile_regression() {
    assert_eq!(LERC1_STUFFED_REMAINDER_TILE.len(), 89);
    let mut blob = LERC1_STUFFED_REMAINDER_TILE.to_vec();
    // This original minimized fuzz seed predates strict section-boundary
    // validation. Give its 23-byte pixel payload the canonical declared size
    // before asserting the legal remainder-tile behavior.
    blob[58..62].copy_from_slice(&23u32.to_le_bytes());

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!((info.width, info.height), (5, 5));
    assert_eq!((info.z_min, info.z_max), (0.0, 3.0));

    let decoded = lerc_reader::decode(&blob).unwrap();
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
