#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = lerc_reader::inspect_first(data);
    let _ = lerc_reader::inspect_first_with_options(
        data,
        lerc_reader::InspectOptions::new().with_compute_value_range(false),
    );
    let _ = lerc_reader::get_blob_info(data);
    let _ = lerc_reader::decode_first(data);
    let _ = lerc_reader::decode(data);
    let _ = lerc_reader::decode_from_reader(&mut std::io::Cursor::new(data));
});
