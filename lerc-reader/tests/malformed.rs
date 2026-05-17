use lerc_core::Error;

#[path = "../../test-support/lerc_test.rs"]
mod lerc_test;

use lerc_test::{
    build_header_v2, encode_mask_rle, finalize_lerc2_with_checksum, pack_msb_bits, HeaderV2,
};

fn build_lerc1_blob_with_stuffed_count(
    mask: &[u8],
    values: &[f32],
    declared_valid_count: u8,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CntZImage ");
    bytes.extend_from_slice(&11i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0.5f64.to_le_bytes());

    let encoded_mask = encode_mask_rle(mask);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(encoded_mask.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&encoded_mask);

    let offset = values.iter().copied().reduce(f32::min).unwrap_or(0.0);
    let quantized: Vec<u32> = values
        .iter()
        .map(|&value| ((value - offset) / 1.0f32).round() as u32)
        .collect();
    let bits_per_pixel = 1u8;
    let payload = pack_msb_bits(&quantized, bits_per_pixel);
    let pixel_section_len = 1 + 4 + 1 + 1 + payload.len();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(pixel_section_len as u32).to_le_bytes());
    bytes.extend_from_slice(&4.0f32.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.push((bits_per_pixel & 63) | (2 << 6));
    bytes.push(declared_valid_count);
    bytes.extend_from_slice(&payload);
    bytes
}

#[test]
fn strict_single_blob_api_rejects_concatenated_payload() {
    let mut blob1 = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 3.0,
        z_max: 3.0,
        payload_len: 0,
    });
    blob1.extend_from_slice(&0u32.to_le_bytes());
    let mut blob2 = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 4.0,
        z_max: 4.0,
        payload_len: 0,
    });
    blob2.extend_from_slice(&0u32.to_le_bytes());
    let mut merged = blob1;
    merged.extend_from_slice(&blob2);

    assert!(matches!(
        lerc_reader::decode(&merged),
        Err(Error::InvalidBlob(_))
    ));
    assert!(matches!(
        lerc_reader::get_blob_info(&merged),
        Err(Error::InvalidBlob(_))
    ));
}

#[test]
fn rejects_mask_rle_with_trailing_bytes_after_sentinel() {
    let mut mask = encode_mask_rle(&[1, 1, 1, 0]);
    mask.push(0xAA);
    let mut blob = build_header_v2(HeaderV2 {
        width: 2,
        height: 2,
        valid_pixel_count: 3,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 3.0,
        payload_len: mask.len() + 1 + 3,
    });
    blob.extend_from_slice(&(mask.len() as u32).to_le_bytes());
    blob.extend_from_slice(&mask);
    blob.push(1);
    blob.extend_from_slice(&[1, 2, 3]);

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(Error::InvalidBlob(_))
    ));
}

#[test]
fn rejects_zero_depth_lerc2_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);
    assert!(matches!(
        lerc_reader::get_blob_info(&blob),
        Err(Error::InvalidHeader("depth must be greater than zero"))
    ));
}

#[test]
fn rejects_no_data_flag_for_unit_depth_lerc2_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.push(0);
    bytes.push(0);
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&(-1.0f64).to_le_bytes());
    bytes.extend_from_slice(&(-9999.0f64).to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);
    assert!(matches!(
        lerc_reader::get_blob_info(&blob),
        Err(Error::InvalidHeader(
            "no-data values require depth greater than one"
        ))
    ));
}

#[test]
fn rejects_lerc1_stuffed_block_with_mismatched_valid_count() {
    let blob = build_lerc1_blob_with_stuffed_count(&[1, 0, 0, 0], &[1.0], 2);
    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(Error::InvalidBlob(_))
    ));
}

#[test]
fn rejects_invalid_huffman_table_header() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.5,
        z_min: 0.0,
        z_max: 255.0,
        payload_len: 1 + 1 + 16,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(0);
    blob.push(1);
    blob.extend_from_slice(&2i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(Error::InvalidBlob(_))
    ));
}

#[test]
fn rejects_non_lerc_trailing_segment_in_band_count() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 3.0,
        z_max: 3.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(b"junk");

    assert!(matches!(
        lerc_reader::get_band_count(&blob),
        Err(Error::InvalidMagic)
    ));
}

#[test]
fn rejects_lerc2_checksum_range_shorter_than_header_prefix() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&3i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());

    assert!(matches!(
        lerc_reader::get_blob_info(&bytes),
        Err(Error::InvalidHeader(
            "blob size is smaller than checksum range"
        ))
    ));
}

#[test]
fn rejects_truncated_bit_stuffed_lerc2_block_without_panicking() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 2,
        height: 1,
        valid_pixel_count: 2,
        image_type: 1,
        max_z_error: 0.5,
        z_min: 0.0,
        z_max: 4.0,
        payload_len: 4 + 1 + 1,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(17);
    blob.push(0xff);

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(Error::Truncated { .. })
    ));
}
