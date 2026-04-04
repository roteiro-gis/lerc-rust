#![no_main]

use libfuzzer_sys::fuzz_target;

fn build_masked_lerc2(mask_payload: &[u8], samples: &[u8]) -> Vec<u8> {
    let payload_len = 1 + samples.len() + mask_payload.len();
    let blob_size = 58 + 4 + payload_len;
    let mut bytes = Vec::with_capacity(blob_size);
    bytes.extend_from_slice(b"Lerc2 ");
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(blob_size as i32).to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&255.0f64.to_le_bytes());
    bytes.extend_from_slice(&(mask_payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(mask_payload);
    bytes.push(1);
    bytes.extend_from_slice(samples);
    bytes
}

fuzz_target!(|data: &[u8]| {
    let split = data.len().min(8);
    let blob = build_masked_lerc2(&data[..split], &data[split..]);
    let _ = lerc_reader::decode(&blob);
    let _ = lerc_reader::decode_mask_ndarray(&blob);
});
