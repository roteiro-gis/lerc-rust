#![no_main]

use libfuzzer_sys::fuzz_target;

fn build_huffman_blob(table_payload: &[u8]) -> Vec<u8> {
    let payload_len = 1 + 1 + table_payload.len();
    let blob_size = 58 + 4 + payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.5f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&255.0f64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.push(0);
    bytes.push(1);
    bytes.extend_from_slice(table_payload);
    bytes
}

fuzz_target!(|data: &[u8]| {
    let blob = build_huffman_blob(data);
    let _ = lerc_reader::decode(&blob);
});
