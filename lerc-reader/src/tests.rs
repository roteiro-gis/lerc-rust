use ndarray::IxDyn;

use crate::*;
use lerc_core::{BandLayout, DataType, Error, MaskEncoding, PixelData, Version};

#[allow(clippy::too_many_arguments)]
fn build_header_v2(
    width: u32,
    height: u32,
    valid_pixel_count: u32,
    image_type: i32,
    max_z_error: f64,
    z_min: f64,
    z_max: f64,
    payload_len: usize,
) -> Vec<u8> {
    let blob_size = 58 + 4 + payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&image_type.to_le_bytes());
    bytes.extend_from_slice(&max_z_error.to_le_bytes());
    bytes.extend_from_slice(&z_min.to_le_bytes());
    bytes.extend_from_slice(&z_max.to_le_bytes());
    bytes
}

#[allow(clippy::too_many_arguments)]
fn build_header_v6(
    width: u32,
    height: u32,
    depth: u32,
    valid_pixel_count: u32,
    image_type: i32,
    max_z_error: f64,
    z_min: f64,
    z_max: f64,
    internal_no_data_value: f64,
    original_no_data_value: f64,
    payload_len: usize,
) -> Vec<u8> {
    let blob_size = 90 + 4 + payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&depth.to_le_bytes());
    bytes.extend_from_slice(&valid_pixel_count.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&image_type.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.push(0);
    bytes.push(0);
    bytes.extend_from_slice(&max_z_error.to_le_bytes());
    bytes.extend_from_slice(&z_min.to_le_bytes());
    bytes.extend_from_slice(&z_max.to_le_bytes());
    bytes.extend_from_slice(&internal_no_data_value.to_le_bytes());
    bytes.extend_from_slice(&original_no_data_value.to_le_bytes());
    bytes
}

fn finalize_v4_with_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
    let blob_size = bytes.len() as i32;
    bytes[34..38].copy_from_slice(&blob_size.to_le_bytes());
    let checksum = crate::pixel::fletcher32(&bytes[14..blob_size as usize]);
    bytes[10..14].copy_from_slice(&checksum.to_le_bytes());
    bytes
}

fn encode_mask_rle(mask: &[u8]) -> Vec<u8> {
    let bitset_len = mask.len().div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    for (index, &value) in mask.iter().enumerate() {
        if value != 0 {
            bitset[index >> 3] |= 1 << (7 - (index & 7));
        }
    }

    let mut encoded = Vec::with_capacity(bitset_len + 4);
    encoded.extend_from_slice(&(bitset_len as i16).to_le_bytes());
    encoded.extend_from_slice(&bitset);
    encoded.extend_from_slice(&i16::MIN.to_le_bytes());
    encoded
}

fn pack_msb_bits(values: &[u32], bits_per_pixel: u8) -> Vec<u8> {
    let total_bits = values.len() * usize::from(bits_per_pixel);
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_offset = 0usize;
    for &value in values {
        for bit in (0..bits_per_pixel).rev() {
            if ((value >> bit) & 1) != 0 {
                let byte_index = bit_offset / 8;
                let bit_index = 7 - (bit_offset % 8);
                bytes[byte_index] |= 1 << bit_index;
            }
            bit_offset += 1;
        }
    }
    bytes
}

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

#[test]
fn reads_blob_info_for_constant_lerc2() {
    let mut blob = build_header_v2(3, 2, 6, 3, 0.0, 7.0, 7.0, 0);
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
    let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 9.0, 9.0, 0);
    blob.extend_from_slice(&0u32.to_le_bytes());

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.mask, None);
    assert_eq!(decoded.pixels, PixelData::U8(vec![9, 9, 9, 9]));
}

#[test]
fn decodes_one_sweep_all_valid() {
    let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 1.0, 4.0, 1 + 4);
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.push(1);
    blob.extend_from_slice(&[1, 2, 3, 4]);

    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.pixels, PixelData::U8(vec![1, 2, 3, 4]));
}

#[test]
fn decodes_external_mask_lerc2_via_single_blob_with_mask_apis() {
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
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&3.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.push(1);
    bytes.push(3);
    bytes.push(1);
    bytes.extend_from_slice(&[1, 2, 3]);

    let blob = finalize_v4_with_checksum(bytes);
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
fn decodes_lerc2_v6_no_data_values_without_public_ranges() {
    let internal_no_data = -7777.0f64;
    let original_no_data = -9999.0f64;
    let mut bytes = build_header_v6(
        1,
        1,
        2,
        1,
        6,
        0.0,
        internal_no_data,
        5.0,
        internal_no_data,
        original_no_data,
        16,
    );
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&5.0f32.to_le_bytes());
    bytes.extend_from_slice(&(internal_no_data as f32).to_le_bytes());
    bytes.extend_from_slice(&5.0f32.to_le_bytes());
    bytes.extend_from_slice(&(internal_no_data as f32).to_le_bytes());

    let blob = finalize_v4_with_checksum(bytes);

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

    let blob = finalize_v4_with_checksum(bytes);
    let decoded = decode(&blob).unwrap();
    assert_eq!(decoded.pixels, PixelData::F32(vec![10.0, 20.0, 10.0, 20.0]));
}

#[test]
fn counts_concatenated_bands() {
    let mut blob1 = build_header_v2(1, 1, 1, 1, 0.0, 3.0, 3.0, 0);
    blob1.extend_from_slice(&0u32.to_le_bytes());
    let mut blob2 = build_header_v2(1, 1, 1, 1, 0.0, 4.0, 4.0, 0);
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
fn counts_concatenated_lerc1_bands_with_shared_mask() {
    let blob1 = build_lerc1_blob(true, Some(&[1, 0, 1, 1]), &[1.0, 3.0, 4.0], 0.5, 4.0);
    let blob2 = build_lerc1_blob(false, None, &[1.0, 3.0, 4.0], 0.5, 4.0);
    let mut merged = blob1;
    merged.extend_from_slice(&blob2);

    assert_eq!(get_band_count(&merged).unwrap(), 2);
}

#[test]
fn decodes_lerc2_to_ndarray() {
    let mut blob = build_header_v2(2, 2, 4, 1, 0.0, 1.0, 4.0, 1 + 4);
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

    let blob = finalize_v4_with_checksum(bytes);
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
    let mut blob1 = build_header_v2(1, 1, 1, 1, 0.0, 3.0, 3.0, 0);
    blob1.extend_from_slice(&0u32.to_le_bytes());
    let mut blob2 = build_header_v2(1, 1, 1, 1, 0.0, 4.0, 4.0, 0);
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
    let mut blob1 = build_header_v2(1, 2, 2, 1, 0.0, 1.0, 2.0, 1 + 2);
    blob1.extend_from_slice(&0u32.to_le_bytes());
    blob1.push(1);
    blob1.extend_from_slice(&[1, 2]);
    let mut blob2 = build_header_v2(1, 2, 2, 1, 0.0, 3.0, 4.0, 1 + 2);
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
    let mut blob1 = build_header_v2(1, 2, 2, 1, 0.0, 1.0, 2.0, 1 + 2);
    blob1.extend_from_slice(&0u32.to_le_bytes());
    blob1.push(1);
    blob1.extend_from_slice(&[1, 2]);
    let mut blob2 = build_header_v2(1, 2, 2, 1, 0.0, 3.0, 4.0, 1 + 2);
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
