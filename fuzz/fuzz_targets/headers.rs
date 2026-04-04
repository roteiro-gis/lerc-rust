#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = lerc_reader::inspect_first(data);
    let _ = lerc_reader::get_blob_info(data);
    let _ = lerc_reader::decode_first(data);
    let _ = lerc_reader::decode(data);
});
