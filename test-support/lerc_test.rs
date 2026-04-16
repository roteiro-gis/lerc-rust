#![allow(dead_code)]

use lerc_core::fletcher32;

pub struct HeaderV2 {
    pub width: u32,
    pub height: u32,
    pub valid_pixel_count: u32,
    pub image_type: i32,
    pub max_z_error: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub payload_len: usize,
}

pub struct HeaderV6 {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub valid_pixel_count: u32,
    pub image_type: i32,
    pub max_z_error: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub internal_no_data_value: f64,
    pub original_no_data_value: f64,
    pub payload_len: usize,
}

pub fn build_header_v2(header: HeaderV2) -> Vec<u8> {
    let blob_size = 58 + 4 + header.payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&header.height.to_le_bytes());
    bytes.extend_from_slice(&header.width.to_le_bytes());
    bytes.extend_from_slice(&header.valid_pixel_count.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&header.image_type.to_le_bytes());
    bytes.extend_from_slice(&header.max_z_error.to_le_bytes());
    bytes.extend_from_slice(&header.z_min.to_le_bytes());
    bytes.extend_from_slice(&header.z_max.to_le_bytes());
    bytes
}

pub fn build_header_v6(header: HeaderV6) -> Vec<u8> {
    let blob_size = 90 + 4 + header.payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&6i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&header.height.to_le_bytes());
    bytes.extend_from_slice(&header.width.to_le_bytes());
    bytes.extend_from_slice(&header.depth.to_le_bytes());
    bytes.extend_from_slice(&header.valid_pixel_count.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&header.image_type.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.push(0);
    bytes.push(0);
    bytes.extend_from_slice(&header.max_z_error.to_le_bytes());
    bytes.extend_from_slice(&header.z_min.to_le_bytes());
    bytes.extend_from_slice(&header.z_max.to_le_bytes());
    bytes.extend_from_slice(&header.internal_no_data_value.to_le_bytes());
    bytes.extend_from_slice(&header.original_no_data_value.to_le_bytes());
    bytes
}

pub fn finalize_lerc2_with_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
    let blob_size = bytes.len() as i32;
    bytes[34..38].copy_from_slice(&blob_size.to_le_bytes());
    let checksum = fletcher32(&bytes[14..blob_size as usize]);
    bytes[10..14].copy_from_slice(&checksum.to_le_bytes());
    bytes
}

pub fn encode_mask_rle(mask: &[u8]) -> Vec<u8> {
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

pub fn pack_msb_bits(values: &[u32], bits_per_pixel: u8) -> Vec<u8> {
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
