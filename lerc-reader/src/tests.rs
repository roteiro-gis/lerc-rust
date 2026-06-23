use ndarray::IxDyn;

use crate::*;
use lerc_core::{BandLayout, DataType, Error, MaskEncoding, PixelData, Version};

#[path = "../../test-support/lerc_test.rs"]
mod lerc_test;

use lerc_test::{
    build_header_v2, build_header_v6, encode_mask_rle, finalize_lerc2_with_checksum, pack_msb_bits,
    HeaderV2, HeaderV6,
};

fn build_lerc1_blob(
    include_mask_header: bool,
    mask: Option<&[u8]>,
    values: &[f32],
    max_z_error: f64,
    max_value: f32,
) -> Vec<u8> {
    let width = 2u32;
    let height = 2u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CntZImage ");
    bytes.extend_from_slice(&11i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&max_z_error.to_le_bytes());

    if include_mask_header {
        if let Some(mask) = mask {
            let encoded_mask = encode_mask_rle(mask);
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&(encoded_mask.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
            bytes.extend_from_slice(&encoded_mask);
        } else {
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }

    let valid_count = mask
        .map(|mask| mask.iter().filter(|&&value| value != 0).count())
        .unwrap_or(values.len());
    let offset = values.iter().copied().reduce(f32::min).unwrap_or(0.0);
    let quantized: Vec<u32> = values
        .iter()
        .map(|&value| ((value - offset) / (2.0 * max_z_error as f32)).round() as u32)
        .collect();
    let max_quantized = quantized.iter().copied().max().unwrap_or(0);
    let bits_per_pixel = crate::pixel::bits_required(max_quantized as usize).max(1);
    let payload = pack_msb_bits(&quantized, bits_per_pixel);
    let pixel_section_len = 1 + 4 + 1 + 1 + payload.len();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(pixel_section_len as u32).to_le_bytes());
    bytes.extend_from_slice(&max_value.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.push((bits_per_pixel & 63) | (2 << 6));
    bytes.push(valid_count as u8);
    bytes.extend_from_slice(&payload);
    bytes
}

fn build_external_mask_lerc2_blob(values: &[u8]) -> Vec<u8> {
    assert_eq!(values.len(), 3);

    let z_min = f64::from(*values.iter().min().unwrap());
    let z_max = f64::from(*values.iter().max().unwrap());

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&z_min.to_le_bytes());
    bytes.extend_from_slice(&z_max.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.push(1);
    bytes.push(values.len() as u8);
    bytes.push(1);
    bytes.extend_from_slice(values);
    finalize_lerc2_with_checksum(bytes)
}

#[test]
fn reads_blob_info_for_constant_lerc2() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 3,
        height: 2,
        valid_pixel_count: 6,
        image_type: 3,
        max_z_error: 0.0,
        z_min: 7.0,
        z_max: 7.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());

    let info = get_blob_info(&blob).unwrap();
    assert_eq!(info.width, 3);
    assert_eq!(info.height, 2);
    assert_eq!(info.depth, 1);
    assert_eq!(info.data_type, DataType::U16);
    assert_eq!(info.blob_size, blob.len());
}

#[test]
fn decodes_constant_surface() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 2,
        height: 2,
        valid_pixel_count: 4,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 9.0,
        z_max: 9.0,
        payload_len: 0,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.mask, None);
    assert_eq!(decoded.pixels, PixelData::U8(vec![9, 9, 9, 9]));
}

#[test]
fn decodes_one_sweep_all_valid() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 2,
        height: 2,
        valid_pixel_count: 4,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 4.0,
        payload_len: 1 + 4,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(1);
    blob.extend_from_slice(&[1, 2, 3, 4]);

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.pixels, PixelData::U8(vec![1, 2, 3, 4]));
}

#[test]
fn decodes_external_mask_lerc2_via_single_blob_with_mask_apis() {
    let blob = build_external_mask_lerc2_blob(&[1, 2, 3]);
    let external_mask = [1u8, 0, 1, 1];

    assert!(matches!(decode(&blob), Err(Error::UnsupportedFeature(_))));

    let info = get_blob_info(&blob).unwrap();
    assert_eq!(info.version, Version::Lerc2(4));
    assert_eq!(info.width, 2);
    assert_eq!(info.height, 2);
    assert_eq!(info.depth, 1);
    assert_eq!(info.valid_pixel_count, 3);
    assert_eq!(info.mask_encoding, MaskEncoding::External);
    assert_eq!(info.mask_count(), 0);
    assert!(info.has_mask());
    assert!(info.uses_external_mask());
    assert!(!info.uses_no_data_value());
    assert_eq!(info.min_values.as_deref(), Some(&[1.0][..]));
    assert_eq!(info.max_values.as_deref(), Some(&[3.0][..]));

    let strict_info = get_blob_info_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(strict_info, info);

    let inspected = inspect_first_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(inspected, info);

    let decoded_first = decode_first_with_mask(&blob, &external_mask).unwrap();
    let decoded = decode_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(decoded_first, decoded);
    assert_eq!(decoded.mask, Some(external_mask.to_vec()));
    assert_eq!(decoded.pixels, PixelData::U8(vec![1, 0, 2, 3]));

    let promoted_first = decode_first_to_f64_with_mask(&blob, &external_mask).unwrap();
    let promoted = decode_to_f64_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(promoted_first, promoted);
    assert_eq!(promoted.mask, Some(external_mask.to_vec()));
    assert_eq!(promoted.pixels, vec![1.0, 0.0, 2.0, 3.0]);

    let array = decode_ndarray_with_mask::<u8>(&blob, &external_mask).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[IxDyn(&[0, 0])], 1);
    assert_eq!(array[IxDyn(&[0, 1])], 0);
    assert_eq!(array[IxDyn(&[1, 0])], 2);
    assert_eq!(array[IxDyn(&[1, 1])], 3);

    let array_f64 = decode_ndarray_f64_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(array_f64.shape(), &[2, 2]);
    assert_eq!(array_f64[IxDyn(&[0, 0])], 1.0);
    assert_eq!(array_f64[IxDyn(&[0, 1])], 0.0);
    assert_eq!(array_f64[IxDyn(&[1, 0])], 2.0);
    assert_eq!(array_f64[IxDyn(&[1, 1])], 3.0);

    let mask = decode_mask_ndarray_with_mask(&blob, &external_mask)
        .unwrap()
        .unwrap();
    assert_eq!(mask.shape(), &[2, 2]);
    assert_eq!(mask[IxDyn(&[0, 0])], 1);
    assert_eq!(mask[IxDyn(&[0, 1])], 0);
    assert_eq!(mask[IxDyn(&[1, 0])], 1);
    assert_eq!(mask[IxDyn(&[1, 1])], 1);
}

#[test]
fn decodes_external_mask_lerc2_band_set_via_with_mask_apis() {
    let external_mask = [1u8, 0, 1, 1];
    let mut blob = build_external_mask_lerc2_blob(&[1, 2, 3]);
    blob.extend_from_slice(&build_external_mask_lerc2_blob(&[4, 5, 6]));

    assert!(matches!(
        decode_band_set(&blob),
        Err(Error::UnsupportedFeature(_))
    ));

    let decoded = decode_band_set_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(decoded.info.band_count(), 2);
    assert_eq!(
        decoded.bands,
        vec![
            PixelData::U8(vec![1, 0, 2, 3]),
            PixelData::U8(vec![4, 0, 5, 6]),
        ]
    );
    assert_eq!(
        decoded.band_masks,
        vec![Some(external_mask.to_vec()), Some(external_mask.to_vec()),]
    );

    let (info, bsq) =
        decode_band_set_vec_with_mask::<u8>(&blob, &external_mask, BandLayout::Bsq).unwrap();
    assert_eq!(info.band_count(), 2);
    assert_eq!(bsq, vec![1, 0, 2, 3, 4, 0, 5, 6]);

    let mut interleaved = vec![0u8; 8];
    let info_into = decode_band_set_into_with_mask(
        &blob,
        &external_mask,
        BandLayout::Interleaved,
        &mut interleaved,
    )
    .unwrap();
    assert_eq!(info_into, info);
    assert_eq!(interleaved, vec![1, 4, 0, 0, 2, 5, 3, 6]);

    let array = decode_band_set_ndarray_with_mask::<u8>(&blob, &external_mask).unwrap();
    assert_eq!(array.shape(), &[2, 2, 2]);
    assert_eq!(array[IxDyn(&[0, 0, 0])], 1);
    assert_eq!(array[IxDyn(&[0, 0, 1])], 4);
    assert_eq!(array[IxDyn(&[0, 1, 0])], 0);
    assert_eq!(array[IxDyn(&[0, 1, 1])], 0);
    assert_eq!(array[IxDyn(&[1, 0, 0])], 2);
    assert_eq!(array[IxDyn(&[1, 0, 1])], 5);
    assert_eq!(array[IxDyn(&[1, 1, 0])], 3);
    assert_eq!(array[IxDyn(&[1, 1, 1])], 6);

    let promoted = decode_band_set_ndarray_f64_with_mask(&blob, &external_mask).unwrap();
    assert_eq!(promoted.shape(), &[2, 2, 2]);
    assert_eq!(promoted[IxDyn(&[0, 0, 0])], 1.0);
    assert_eq!(promoted[IxDyn(&[0, 0, 1])], 4.0);
    assert_eq!(promoted[IxDyn(&[0, 1, 0])], 0.0);
    assert_eq!(promoted[IxDyn(&[0, 1, 1])], 0.0);
    assert_eq!(promoted[IxDyn(&[1, 0, 0])], 2.0);
    assert_eq!(promoted[IxDyn(&[1, 0, 1])], 5.0);
    assert_eq!(promoted[IxDyn(&[1, 1, 0])], 3.0);
    assert_eq!(promoted[IxDyn(&[1, 1, 1])], 6.0);

    let band_mask = decode_band_mask_ndarray_with_mask(&blob, &external_mask)
        .unwrap()
        .unwrap();
    assert_eq!(band_mask.shape(), &[2, 2, 2]);
    assert_eq!(band_mask[IxDyn(&[0, 0, 0])], 1);
    assert_eq!(band_mask[IxDyn(&[0, 0, 1])], 1);
    assert_eq!(band_mask[IxDyn(&[0, 1, 0])], 0);
    assert_eq!(band_mask[IxDyn(&[0, 1, 1])], 0);
    assert_eq!(band_mask[IxDyn(&[1, 0, 0])], 1);
    assert_eq!(band_mask[IxDyn(&[1, 0, 1])], 1);
    assert_eq!(band_mask[IxDyn(&[1, 1, 0])], 1);
    assert_eq!(band_mask[IxDyn(&[1, 1, 1])], 1);
}

#[test]
fn decodes_lerc2_v6_no_data_values_without_public_ranges() {
    let internal_no_data = -7777.0f64;
    let original_no_data = -9999.0f64;
    let mut bytes = build_header_v6(HeaderV6 {
        width: 1,
        height: 1,
        depth: 2,
        valid_pixel_count: 1,
        image_type: 6,
        max_z_error: 0.0,
        z_min: internal_no_data,
        z_max: 5.0,
        internal_no_data_value: internal_no_data,
        original_no_data_value: original_no_data,
        payload_len: 16,
    });
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&5.0f32.to_le_bytes());
    bytes.extend_from_slice(&(internal_no_data as f32).to_le_bytes());
    bytes.extend_from_slice(&5.0f32.to_le_bytes());
    bytes.extend_from_slice(&(internal_no_data as f32).to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);

    let info = get_blob_info(&blob).unwrap();
    assert_eq!(info.version, Version::Lerc2(6));
    assert_eq!(info.width, 1);
    assert_eq!(info.height, 1);
    assert_eq!(info.depth, 2);
    assert_eq!(info.valid_pixel_count, 1);
    assert_eq!(info.mask_encoding, MaskEncoding::None);
    assert_eq!(info.mask_count(), 0);
    assert_eq!(info.no_data_value, Some(original_no_data));
    assert!(info.uses_no_data_value());
    assert_eq!(info.z_min, -1.0);
    assert_eq!(info.z_max, -1.0);
    assert_eq!(info.min_values, None);
    assert_eq!(info.max_values, None);

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.info, info);
    assert_eq!(decoded.mask, None);
    assert_eq!(
        decoded.pixels,
        PixelData::F32(vec![5.0, original_no_data as f32])
    );

    let promoted = decode_to_f64(&blob).unwrap();
    assert_eq!(promoted.info, info);
    assert_eq!(promoted.mask, None);
    assert_eq!(promoted.pixels, vec![5.0, original_no_data]);

    let mask = decode_mask_ndarray(&blob).unwrap();
    assert_eq!(mask, None);

    let array = decode_ndarray_f64(&blob).unwrap();
    assert_eq!(array.shape(), &[1, 1, 2]);
    assert_eq!(array[IxDyn(&[0, 0, 0])], 5.0);
    assert_eq!(array[IxDyn(&[0, 0, 1])], original_no_data);
}

#[test]
fn decodes_constant_surface_with_per_depth_ranges() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&10.0f64.to_le_bytes());
    bytes.extend_from_slice(&20.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);
    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.pixels, PixelData::F32(vec![10.0, 20.0, 10.0, 20.0]));
}

#[test]
fn direct_band_set_apis_return_lerc2_per_depth_ranges() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&10.0f64.to_le_bytes());
    bytes.extend_from_slice(&20.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);
    let decoded = decode_band_set(&blob).unwrap();
    assert_eq!(
        decoded.info.bands[0].min_values.as_deref(),
        Some(&[10.0, 20.0][..])
    );
    assert_eq!(
        decoded.info.bands[0].max_values.as_deref(),
        Some(&[10.0, 20.0][..])
    );

    let (info, values) = decode_band_set_vec::<f32>(&blob, BandLayout::Bsq).unwrap();
    assert_eq!(info, decoded.info);
    assert_eq!(values, vec![10.0, 20.0, 10.0, 20.0]);

    let mut out = vec![0.0f32; values.len()];
    let info_into = decode_band_set_into(&blob, BandLayout::Bsq, &mut out).unwrap();
    assert_eq!(info_into, decoded.info);
    assert_eq!(out, values);
}

#[test]
fn counts_concatenated_bands() {
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
    assert_eq!(get_band_count(&merged).unwrap(), 2);
}

#[test]
fn reads_blob_info_for_lerc1() {
    let blob = build_lerc1_blob(true, None, &[1.0, 2.0, 3.0, 4.0], 0.5, 4.0);

    let info = get_blob_info(&blob).unwrap();
    assert_eq!(info.version, Version::Lerc1(11));
    assert_eq!(info.width, 2);
    assert_eq!(info.height, 2);
    assert_eq!(info.depth, 1);
    assert_eq!(info.data_type, DataType::F32);
}

#[test]
fn decodes_lerc1_masked_bitstuffed_block() {
    let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.mask, Some(vec![1, 0, 1, 1]));
    assert_eq!(decoded.pixels, PixelData::F32(vec![1.0, 0.0, 3.0, 4.0]));
    assert_eq!(decoded.info.z_min, 1.0);
    assert_eq!(decoded.info.z_max, 4.0);
}

#[test]
fn decodes_lerc1_unsigned_byte_offsets() {
    let quantized = [0, 1, 2, 3];
    let payload = pack_msb_bits(&quantized, 2);
    let pixel_section_len = 1 + 1 + 1 + 1 + payload.len();
    let mut blob = Vec::new();
    blob.extend_from_slice(b"CntZImage ");
    blob.extend_from_slice(&11i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&2u32.to_le_bytes());
    blob.extend_from_slice(&2u32.to_le_bytes());
    blob.extend_from_slice(&0.1f64.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&(pixel_section_len as u32).to_le_bytes());
    blob.extend_from_slice(&200.6f32.to_le_bytes());
    blob.push((2 << 6) | 1);
    blob.push(200);
    blob.push(2 | (2 << 6));
    blob.push(4);
    blob.extend_from_slice(&payload);

    let decoded = decode(&blob).unwrap();
    assert_eq!(
        decoded.pixels,
        PixelData::F32(vec![200.0, 200.2, 200.4, 200.6])
    );
}

#[test]
fn lerc1_with_mask_prefers_inline_masked_blob_over_caller_mask() {
    let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);
    let caller_mask = [1u8, 1, 1, 1];

    let info = get_blob_info_with_mask(&blob, &caller_mask).unwrap();
    assert_eq!(info.mask_encoding, MaskEncoding::Explicit);
    assert_eq!(info.valid_pixel_count, 3);

    let decoded = decode_with_mask(&blob, &caller_mask).unwrap();
    assert_eq!(decoded.mask, Some(vec![1, 0, 1, 1]));
    assert_eq!(decoded.pixels, PixelData::F32(vec![1.0, 0.0, 3.0, 4.0]));

    let mask = decode_mask_ndarray_with_mask(&blob, &caller_mask)
        .unwrap()
        .unwrap();
    assert_eq!(mask.shape(), &[2, 2]);
    assert_eq!(mask[IxDyn(&[0, 0])], 1);
    assert_eq!(mask[IxDyn(&[0, 1])], 0);
    assert_eq!(mask[IxDyn(&[1, 0])], 1);
    assert_eq!(mask[IxDyn(&[1, 1])], 1);
}

#[test]
fn lerc1_with_mask_still_decodes_detached_shared_mask_blob() {
    let blob = build_lerc1_blob(false, None, &[1.0, 3.0, 4.0], 0.5, 4.0);
    let shared_mask = [1u8, 0, 1, 1];

    assert!(matches!(decode(&blob), Err(Error::Truncated { .. })));

    let info = get_blob_info_with_mask(&blob, &shared_mask).unwrap();
    assert_eq!(info.mask_encoding, MaskEncoding::External);
    assert_eq!(info.valid_pixel_count, 3);

    let decoded = decode_with_mask(&blob, &shared_mask).unwrap();
    assert_eq!(decoded.mask, Some(shared_mask.to_vec()));
    assert_eq!(decoded.pixels, PixelData::F32(vec![1.0, 0.0, 3.0, 4.0]));
}

#[test]
fn counts_concatenated_lerc1_bands_with_shared_mask() {
    let blob1 = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);
    let blob2 = build_lerc1_blob(false, None, &[1.0, 3.0, 4.0], 0.5, 4.0);
    let mut merged = blob1;
    merged.extend_from_slice(&blob2);

    assert_eq!(get_band_count(&merged).unwrap(), 2);
}

#[test]
fn direct_band_set_apis_return_decoded_lerc1_ranges() {
    let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 99.0);

    let decoded = decode_band_set(&blob).unwrap();
    assert_eq!(decoded.info.bands[0].z_min, 1.0);
    assert_eq!(decoded.info.bands[0].z_max, 4.0);

    let (info, values) = decode_band_set_vec::<f32>(&blob, BandLayout::Bsq).unwrap();
    assert_eq!(info, decoded.info);
    assert_eq!(values, vec![1.0, 0.0, 3.0, 4.0]);

    let mut out = vec![0.0f32; values.len()];
    let info_into = decode_band_set_into(&blob, BandLayout::Bsq, &mut out).unwrap();
    assert_eq!(info_into, decoded.info);
    assert_eq!(out, values);
}

#[test]
fn decodes_lerc2_to_ndarray() {
    let mut blob = build_header_v2(HeaderV2 {
        width: 2,
        height: 2,
        valid_pixel_count: 4,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 4.0,
        payload_len: 1 + 4,
    });
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(1);
    blob.extend_from_slice(&[1, 2, 3, 4]);

    let array = decode_ndarray::<u8>(&blob).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[IxDyn(&[0, 0])], 1);
    assert_eq!(array[IxDyn(&[0, 1])], 2);
    assert_eq!(array[IxDyn(&[1, 0])], 3);
    assert_eq!(array[IxDyn(&[1, 1])], 4);
}

#[test]
fn decodes_multidimensional_lerc2_to_f64_ndarray() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&10.0f64.to_le_bytes());
    bytes.extend_from_slice(&20.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());
    bytes.extend_from_slice(&10.0f32.to_le_bytes());
    bytes.extend_from_slice(&20.0f32.to_le_bytes());

    let blob = finalize_lerc2_with_checksum(bytes);
    let array = decode_ndarray_f64(&blob).unwrap();
    assert_eq!(array.shape(), &[1, 2, 2]);
    assert_eq!(array[IxDyn(&[0, 0, 0])], 10.0);
    assert_eq!(array[IxDyn(&[0, 0, 1])], 20.0);
    assert_eq!(array[IxDyn(&[0, 1, 0])], 10.0);
    assert_eq!(array[IxDyn(&[0, 1, 1])], 20.0);
}

#[test]
fn decodes_lerc1_mask_to_ndarray() {
    let blob = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);

    let mask = decode_mask_ndarray(&blob).unwrap().unwrap();
    assert_eq!(mask.shape(), &[2, 2]);
    assert_eq!(mask[IxDyn(&[0, 0])], 1);
    assert_eq!(mask[IxDyn(&[0, 1])], 0);
    assert_eq!(mask[IxDyn(&[1, 0])], 1);
    assert_eq!(mask[IxDyn(&[1, 1])], 1);
}

#[test]
fn strict_single_blob_apis_reject_trailing_bytes() {
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
    let mut merged = blob1.clone();
    merged.extend_from_slice(&blob2);

    assert!(matches!(
        get_blob_info(&merged),
        Err(lerc_core::Error::InvalidBlob(_))
    ));
    assert!(matches!(
        decode(&merged),
        Err(lerc_core::Error::InvalidBlob(_))
    ));

    let info = inspect_first(&merged).unwrap();
    assert_eq!(info.blob_size, blob1.len());
    let decoded = decode_first(&merged).unwrap();
    assert_eq!(decoded.info.blob_size, blob1.len());
}

#[test]
fn decode_band_set_into_supports_bsq_layout() {
    let mut blob1 = build_header_v2(HeaderV2 {
        width: 1,
        height: 2,
        valid_pixel_count: 2,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 2.0,
        payload_len: 1 + 2,
    });
    blob1.extend_from_slice(&0u32.to_le_bytes());
    blob1.push(1);
    blob1.extend_from_slice(&[1, 2]);
    let mut blob2 = build_header_v2(HeaderV2 {
        width: 1,
        height: 2,
        valid_pixel_count: 2,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 3.0,
        z_max: 4.0,
        payload_len: 1 + 2,
    });
    blob2.extend_from_slice(&0u32.to_le_bytes());
    blob2.push(1);
    blob2.extend_from_slice(&[3, 4]);

    let mut merged = blob1;
    merged.extend_from_slice(&blob2);

    let mut out = vec![0u8; 4];
    let info = decode_band_set_into(&merged, BandLayout::Bsq, &mut out).unwrap();
    assert_eq!(info.band_count(), 2);
    assert_eq!(out, vec![1, 2, 3, 4]);
}

#[test]
fn decode_band_set_ndarray_f64_promotes_concatenated_bands_directly() {
    let mut blob1 = build_header_v2(HeaderV2 {
        width: 1,
        height: 2,
        valid_pixel_count: 2,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 1.0,
        z_max: 2.0,
        payload_len: 1 + 2,
    });
    blob1.extend_from_slice(&0u32.to_le_bytes());
    blob1.push(1);
    blob1.extend_from_slice(&[1, 2]);
    let mut blob2 = build_header_v2(HeaderV2 {
        width: 1,
        height: 2,
        valid_pixel_count: 2,
        image_type: 1,
        max_z_error: 0.0,
        z_min: 3.0,
        z_max: 4.0,
        payload_len: 1 + 2,
    });
    blob2.extend_from_slice(&0u32.to_le_bytes());
    blob2.push(1);
    blob2.extend_from_slice(&[3, 4]);

    let mut merged = blob1;
    merged.extend_from_slice(&blob2);

    let array = decode_band_set_ndarray_f64(&merged).unwrap();
    assert_eq!(array.shape(), &[2, 1, 2]);
    assert_eq!(array[IxDyn(&[0, 0, 0])], 1.0);
    assert_eq!(array[IxDyn(&[0, 0, 1])], 3.0);
    assert_eq!(array[IxDyn(&[1, 0, 0])], 2.0);
    assert_eq!(array[IxDyn(&[1, 0, 1])], 4.0);
}
