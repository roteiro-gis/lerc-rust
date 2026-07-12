#![allow(missing_docs)]

use lerc_core::{BandLayout, Error, PixelData};

#[path = "../../test-support/lerc_test.rs"]
mod lerc_test;

use lerc_test::{
    build_header_v2, build_header_v6, encode_mask_rle, finalize_lerc2_with_checksum, pack_msb_bits,
    HeaderV2, HeaderV6,
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

fn build_huffman_blob(table_payload: &[u8]) -> Vec<u8> {
    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.5,
        z_min: 0.0,
        z_max: 255.0,
        payload_len: 1 + 1 + table_payload.len(),
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(0);
    blob.push(1);
    blob.extend_from_slice(table_payload);
    blob
}

fn build_lerc2_v4_no_mask_blob(
    width: u32,
    height: u32,
    depth: u32,
    valid_pixel_count: u32,
    z_min: f64,
    z_max: f64,
    pixel_payload: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&depth.to_le_bytes());
    bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&z_min.to_le_bytes());
    bytes.extend_from_slice(&z_max.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(pixel_payload);
    finalize_lerc2_with_checksum(bytes)
}

fn build_lerc1_blob_with_block_grid(
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CntZImage ");
    bytes.extend_from_slice(&11i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&0.5f64.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&blocks_y.to_le_bytes());
    bytes.extend_from_slice(&blocks_x.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes
}

fn build_lerc1_remainder_tile_blob() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CntZImage ");
    bytes.extend_from_slice(&11i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&5u32.to_le_bytes());
    bytes.extend_from_slice(&5u32.to_le_bytes());
    bytes.extend_from_slice(&0.5f64.to_le_bytes());

    // No mask: all 25 pixels are valid.
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    // A nominal 3x3 grid over 5x5 has four blocks on each axis. The final
    // block is 2x2 even though the base blocks are 1x1.
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&3.0f32.to_le_bytes());

    bytes.extend_from_slice(&[2; 15]); // Zero blocks.

    bytes.push(1); // Stuffed block with an f32 offset.
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.push((2 << 6) | 2); // u8 valid-count followed by two-bit values.
    bytes.push(4);
    bytes.extend_from_slice(&pack_msb_bits(&[0, 1, 2, 3], 2));
    bytes
}

fn assert_metadata_invalid_header(blob: &[u8], expected: &'static str) {
    let result = lerc_reader::get_blob_info(blob);
    assert!(
        matches!(result, Err(Error::InvalidHeader(actual)) if actual == expected),
        "{result:?}"
    );

    let result = lerc_reader::get_band_count(blob);
    assert!(
        matches!(result, Err(Error::InvalidHeader(actual)) if actual == expected),
        "{result:?}"
    );
}

#[test]
fn accepts_lerc1_stuffed_remainder_tile_larger_than_base_tile() {
    let blob = build_lerc1_remainder_tile_blob();
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, // row 0
        0.0, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 0.0, 0.0, 0.0, 0.0, // row 2
        0.0, 0.0, 0.0, 0.0, 1.0, // row 3
        0.0, 0.0, 0.0, 2.0, 3.0, // row 4
    ];

    let info = lerc_reader::get_blob_info(&blob).unwrap();
    assert_eq!((info.width, info.height), (5, 5));
    assert_eq!((info.z_min, info.z_max), (0.0, 3.0));

    let decoded = lerc_reader::decode(&blob).unwrap();
    assert_eq!(decoded.pixels, PixelData::F32(expected.clone()));

    let mut direct = vec![f32::NAN; expected.len()];
    let band_info = lerc_reader::decode_band_set_into(&blob, BandLayout::Bsq, &mut direct).unwrap();
    assert_eq!(band_info.bands, vec![info]);
    assert_eq!(direct, expected);
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
fn rejects_lerc2_trailing_pixel_payload_in_constant_blob() {
    let blob = build_lerc2_v4_no_mask_blob(1, 1, 1, 1, 7.0, 7.0, b"junk");
    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");

    let result = lerc_reader::decode_to_f64(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_lerc2_trailing_pixel_payload_in_all_invalid_blob() {
    let blob = build_lerc2_v4_no_mask_blob(1, 1, 1, 0, 0.0, 1.0, b"junk");
    let result = lerc_reader::decode_first(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");

    let mut out = vec![0u8; 1];
    let result = lerc_reader::decode_band_set_into(&blob, lerc_core::BandLayout::Bsq, &mut out);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_zero_sized_lerc2_band_set_vec_before_payload_decode() {
    let blob = build_lerc2_v4_no_mask_blob(0, 1, 1, 0, 0.0, 0.0, b"junk");
    let result = lerc_reader::decode(&blob);
    assert!(
        matches!(
            result,
            Err(Error::InvalidHeader(
                "width and height must be greater than zero"
            ))
        ),
        "{result:?}"
    );

    let result = lerc_reader::decode_band_set_vec::<u8>(&blob, lerc_core::BandLayout::Bsq);
    assert!(
        matches!(
            result,
            Err(Error::InvalidHeader(
                "width and height must be greater than zero"
            ))
        ),
        "{result:?}"
    );

    let result = lerc_reader::decode_band_set_ndarray::<u8>(&blob);
    assert!(
        matches!(
            result,
            Err(Error::InvalidHeader(
                "width and height must be greater than zero"
            ))
        ),
        "{result:?}"
    );

    let result = lerc_reader::decode_band_set_ndarray_f64(&blob);
    assert!(
        matches!(
            result,
            Err(Error::InvalidHeader(
                "width and height must be greater than zero"
            ))
        ),
        "{result:?}"
    );
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

    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_zero_sized_lerc2_dimensions_in_metadata_paths() {
    for (width, height) in [(0, 1), (1, 0)] {
        let mut blob = build_header_v2(HeaderV2 {
            width,
            height,
            valid_pixel_count: 0,
            image_type: 1,
            max_z_error: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            payload_len: 0,
        });
        blob.extend_from_slice(&0u32.to_le_bytes());

        assert_metadata_invalid_header(&blob, "width and height must be greater than zero");
    }
}

#[test]
fn rejects_zero_lerc2_micro_block_size_in_metadata_paths() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 7.0,
        z_max: 7.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob[22..26].copy_from_slice(&0i32.to_le_bytes());

    assert_metadata_invalid_header(&blob, "micro block size must be greater than zero");
}

#[test]
fn rejects_non_finite_lerc2_numeric_fields_in_metadata_paths() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: f64::NAN,
        z_min: 7.0,
        z_max: 7.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    assert_metadata_invalid_header(&blob, "max_z_error must be finite and non-negative");

    let mut blob = build_header_v2(HeaderV2 {
        width: 1,
        height: 1,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: f64::INFINITY,
        z_max: 7.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    assert_metadata_invalid_header(&blob, "z range values must be finite");

    let mut blob = build_header_v6(HeaderV6 {
        width: 1,
        height: 1,
        depth: 2,
        valid_pixel_count: 1,
        image_type: 6,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 1.0,
        internal_no_data_value: f64::NAN,
        original_no_data_value: -9999.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    let blob = finalize_lerc2_with_checksum(blob);
    assert_metadata_invalid_header(&blob, "no-data values must be finite");
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
fn rejects_zero_sized_lerc1_dimensions_in_metadata_paths() {
    let blob = build_lerc1_blob_with_block_grid(0, 1, 1, 1);
    assert_metadata_invalid_header(&blob, "width and height must be greater than zero");
}

#[test]
fn rejects_lerc1_block_grid_larger_than_dimensions_in_metadata_paths() {
    let blob = build_lerc1_blob_with_block_grid(1, 1, 2, 1);
    assert_metadata_invalid_header(&blob, "Lerc1 block grid must not exceed raster dimensions");
}

#[test]
fn rejects_lerc1_stuffed_block_with_mismatched_valid_count() {
    let blob = build_lerc1_blob_with_stuffed_count(&[1, 0, 0, 0], &[1.0], 2);
    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_invalid_huffman_table_header() {
    let mut table_payload = Vec::new();
    table_payload.extend_from_slice(&2i32.to_le_bytes());
    table_payload.extend_from_slice(&0i32.to_le_bytes());
    table_payload.extend_from_slice(&0i32.to_le_bytes());
    table_payload.extend_from_slice(&0i32.to_le_bytes());
    let blob = build_huffman_blob(&table_payload);

    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_huffman_table_span_that_exceeds_symbol_count_without_allocating() {
    let blob = build_huffman_blob(&[
        0x7b, 0x7b, 0x7b, 0x7a, 0x7b, 0x7b, 0x7b, 0x2b, 0x02, 0x00, 0x00, 0x00, 0x2b, 0x2b, 0x86,
        0x10,
    ]);

    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_huffman_code_length_that_exceeds_bitstream_width() {
    let mut table_payload = Vec::new();
    table_payload.extend_from_slice(&2i32.to_le_bytes());
    table_payload.extend_from_slice(&1i32.to_le_bytes());
    table_payload.extend_from_slice(&0i32.to_le_bytes());
    table_payload.extend_from_slice(&1i32.to_le_bytes());
    table_payload.extend_from_slice(&[0x86, 0x01, 0x84]);
    let blob = build_huffman_blob(&table_payload);

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
fn rejects_huge_lerc2_mask_before_allocating() {
    let mut blob = build_header_v2(HeaderV2 {
        width: u32::MAX,
        height: u32::MAX,
        valid_pixel_count: 1,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 0.0,
        z_max: 1.0,
        payload_len: 2,
    });
    blob.extend_from_slice(&2u32.to_le_bytes());
    blob.extend_from_slice(&i16::MIN.to_le_bytes());

    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
}

#[test]
fn rejects_huge_lerc2_constant_output_before_allocating() {
    let mut mask = Vec::new();
    mask.extend_from_slice(&1i16.to_le_bytes());
    mask.push(0x80);
    mask.extend_from_slice(&(-32767i16).to_le_bytes());
    mask.push(0);
    mask.extend_from_slice(&i16::MIN.to_le_bytes());

    let mut blob = build_header_v6(HeaderV6 {
        width: 512,
        height: 512,
        depth: u32::MAX,
        valid_pixel_count: 1,
        image_type: 7,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 1.0,
        internal_no_data_value: -9999.0,
        original_no_data_value: -9999.0,
        payload_len: mask.len(),
    });
    blob.extend_from_slice(&(mask.len() as u32).to_le_bytes());
    blob.extend_from_slice(&mask);
    let blob = finalize_lerc2_with_checksum(blob);

    let result = lerc_reader::decode(&blob);
    assert!(matches!(result, Err(Error::InvalidBlob(_))), "{result:?}");
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
