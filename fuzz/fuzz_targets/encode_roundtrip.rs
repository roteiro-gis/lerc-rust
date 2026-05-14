#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let width = usize::from(data[0] % 8) + 1;
    let height = usize::from(data[1] % 8) + 1;
    let depth = usize::from(data[2] % 3) + 1;
    let pixel_count = width * height;
    let sample_count = pixel_count * depth;
    if data.len() < 4 + sample_count + pixel_count {
        return;
    }

    let options = lerc_writer::EncodeOptions {
        max_z_error: if data[3] & 1 == 0 { 0.0 } else { 0.5 },
        micro_block_size: u32::from(data[3] % 8) + 1,
        no_data_value: None,
    };

    let pixels = &data[4..4 + sample_count];
    let mask_bytes = &data[4 + sample_count..4 + sample_count + pixel_count];
    let mask = if mask_bytes.iter().all(|&byte| byte != 0) {
        None
    } else {
        Some(lerc_core::MaskView::new(
            width as u32,
            height as u32,
            mask_bytes,
        )
        .unwrap())
    };

    let raster = lerc_core::RasterView::new(width as u32, height as u32, depth as u32, pixels)
        .unwrap();
    let Ok(blob) = lerc_writer::encode(raster, mask, options) else {
        return;
    };
    let Ok(decoded) = lerc_reader::decode(&blob) else {
        panic!("writer produced a blob the reader rejected");
    };

    match decoded.pixels {
        lerc_core::PixelData::U8(values) => {
            assert_eq!(values.len(), sample_count);
        }
        other => panic!("expected U8 decode output, got {other:?}"),
    }
    if let Some(mask) = decoded.mask {
        assert_eq!(mask.len(), pixel_count);
    }
});
