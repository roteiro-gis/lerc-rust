#![no_main]

use libfuzzer_sys::fuzz_target;

fn pack_bits(values: &[u8], bits_per_pixel: u8) -> Vec<u8> {
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

fn build_lerc1_bitstuff_blob(payload: &[u8]) -> Vec<u8> {
    let bits_per_pixel = 1u8;
    let stuffed = pack_bits(payload, bits_per_pixel);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CntZImage ");
    bytes.extend_from_slice(&11i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0.5f64.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(1 + 4 + 1 + 1 + stuffed.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&255.0f32.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.push((bits_per_pixel & 63) | (2 << 6));
    bytes.push(payload.len().min(4) as u8);
    bytes.extend_from_slice(&stuffed);
    bytes
}

fuzz_target!(|data: &[u8]| {
    let blob = build_lerc1_bitstuff_blob(data);
    let _ = lerc_reader::inspect_first(&blob);
    let _ = lerc_reader::decode_first(&blob);
});
