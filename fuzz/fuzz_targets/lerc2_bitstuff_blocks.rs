#![no_main]

use lerc_core::fletcher32;
use libfuzzer_sys::fuzz_target;

fn build_lerc2_bitstuff_blob(payload: &[u8]) -> Vec<u8> {
    let width = 4u32;
    let height = 4u32;
    let payload_len = 4 + 1 + 1 + payload.len();
    let blob_size = 62 + payload_len;
    let mut bytes = Vec::with_capacity(blob_size);

    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&3i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&(width * height).to_le_bytes());
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&255.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    bytes.push(0);
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(payload);

    let checksum = fletcher32(&bytes[14..blob_size]);
    bytes[10..14].copy_from_slice(&checksum.to_le_bytes());
    bytes
}

fuzz_target!(|data: &[u8]| {
    let blob = build_lerc2_bitstuff_blob(data);
    let _ = lerc_reader::inspect_first(&blob);
    let _ = lerc_reader::decode_first(&blob);
    let _ = lerc_reader::decode_to_f64(&blob);
});
