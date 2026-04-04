#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mid = data.len() / 2;
    let mut blob = Vec::with_capacity(data.len() + 12);
    blob.extend_from_slice(&data[..mid]);
    blob.extend_from_slice(&data[mid..]);

    let _ = lerc_reader::get_band_count(&blob);
    let _ = lerc_reader::decode_band_set(&blob);
});
