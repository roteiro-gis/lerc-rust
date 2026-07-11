#![no_main]

use lerc_core::{BandLayout, BandSetView, MaskView};
use libfuzzer_sys::fuzz_target;

fn body_offset(blob: &[u8], info: &lerc_core::BlobInfo) -> usize {
    let header_len = match info.version {
        lerc_core::Version::Lerc2(version) if version >= 6 => 90,
        lerc_core::Version::Lerc2(_) => 66,
        lerc_core::Version::Lerc1(_) => unreachable!("writer only emits Lerc2"),
    };
    let mask_num_bytes =
        u32::from_le_bytes(blob[header_len..header_len + 4].try_into().unwrap()) as usize;
    let range_len = info
        .min_values
        .as_ref()
        .map(|_| info.depth as usize * 2 * info.data_type.byte_len())
        .unwrap_or(0);
    header_len + 4 + mask_num_bytes + range_len
}

fn assert_direct_band_set_ndarrays_match<T>(blob: &[u8])
where
    T: lerc_reader::BandElement + std::fmt::Debug + PartialEq,
{
    for layout in [BandLayout::Interleaved, BandLayout::Bsq] {
        let from_band_set = lerc_reader::decode_band_set(blob)
            .unwrap()
            .into_ndarray_with_layout::<T>(layout)
            .unwrap();
        let direct = lerc_reader::decode_band_set_ndarray_with_layout::<T>(blob, layout).unwrap();
        assert_eq!(from_band_set.shape(), direct.shape());
        assert_eq!(
            from_band_set.as_slice_memory_order(),
            direct.as_slice_memory_order()
        );
    }

    let from_band_set = lerc_reader::decode_band_set(blob)
        .unwrap()
        .into_ndarray::<f64>()
        .unwrap();
    let direct = lerc_reader::decode_band_set_ndarray_f64(blob).unwrap();
    assert_eq!(from_band_set.shape(), direct.shape());
    assert_eq!(
        from_band_set.as_slice_memory_order(),
        direct.as_slice_memory_order()
    );
}

fn assert_direct_band_set_ndarrays_with_mask_match<T>(blob: &[u8], mask: &[u8])
where
    T: lerc_reader::BandElement + std::fmt::Debug + PartialEq,
{
    for layout in [BandLayout::Interleaved, BandLayout::Bsq] {
        let from_band_set = lerc_reader::decode_band_set_with_mask(blob, mask)
            .unwrap()
            .into_ndarray_with_layout::<T>(layout)
            .unwrap();
        let direct =
            lerc_reader::decode_band_set_ndarray_with_layout_and_mask::<T>(blob, layout, mask)
                .unwrap();
        assert_eq!(from_band_set.shape(), direct.shape());
        assert_eq!(
            from_band_set.as_slice_memory_order(),
            direct.as_slice_memory_order()
        );
    }

    let from_band_set = lerc_reader::decode_band_set_with_mask(blob, mask)
        .unwrap()
        .into_ndarray::<f64>()
        .unwrap();
    let direct = lerc_reader::decode_band_set_ndarray_f64_with_mask(blob, mask).unwrap();
    assert_eq!(from_band_set.shape(), direct.shape());
    assert_eq!(
        from_band_set.as_slice_memory_order(),
        direct.as_slice_memory_order()
    );
}

fn encode_and_check_u8_band_set(
    width: usize,
    height: usize,
    depth: usize,
    band_count: usize,
    layout: BandLayout,
    values: &[u8],
    mask: Option<&[u8]>,
    options: lerc_writer::EncodeOptions,
) -> Option<Vec<u8>> {
    let band_set = BandSetView::new(
        width as u32,
        height as u32,
        depth as u32,
        band_count,
        layout,
        values,
    )
    .unwrap();
    let mask_view = mask.map(|mask| MaskView::new(width as u32, height as u32, mask).unwrap());
    let Ok(blob) = lerc_writer::encode_band_set(band_set, mask_view, options) else {
        return None;
    };

    assert_direct_band_set_ndarrays_match::<u8>(&blob);
    if let Some(mask) = mask {
        assert_direct_band_set_ndarrays_with_mask_match::<u8>(&blob, mask);
    }
    Some(blob)
}

fn generated_band_set_roundtrips(data: &[u8]) {
    if data.len() < 6 {
        return;
    }

    let width = usize::from(data[0] % 5) + 1;
    let height = usize::from(data[1] % 5) + 1;
    let depth = usize::from(data[2] % 2) + 1;
    let band_count = usize::from(data[3] % 3) + 2;
    let pixel_count = width * height;
    let value_count = pixel_count * depth * band_count;
    let needed = 6 + value_count + pixel_count;
    if data.len() < needed {
        return;
    }

    let values = &data[6..6 + value_count];
    let mask_bytes = &data[6 + value_count..needed];
    let mask = if data[4] & 1 == 0 {
        None
    } else {
        Some(mask_bytes)
    };
    let layout = if data[4] & 2 == 0 {
        BandLayout::Interleaved
    } else {
        BandLayout::Bsq
    };
    let options = lerc_writer::EncodeOptions::new()
        .with_max_z_error(if data[5] & 1 == 0 { 0.0 } else { 0.5 })
        .with_micro_block_size(u32::from(data[5] % 5) + 2);

    let _ = encode_and_check_u8_band_set(
        width, height, depth, band_count, layout, values, mask, options,
    );
}

fn zero_block_band_set_roundtrip(data: &[u8]) {
    let width = 16usize;
    let height = 8usize;
    let band_count = 2usize;
    let mut values = vec![0u8; width * height * band_count];
    for row in 0..height {
        for col in 8..width {
            let pixel = row * width + col;
            values[pixel * band_count] = data.first().copied().unwrap_or(7).max(1);
            values[pixel * band_count + 1] = data.get(1).copied().unwrap_or(11).max(1);
        }
    }

    let Some(blob) = encode_and_check_u8_band_set(
        width,
        height,
        1,
        band_count,
        BandLayout::Interleaved,
        &values,
        None,
        lerc_writer::EncodeOptions::new().with_micro_block_size(8),
    ) else {
        return;
    };
    let info = lerc_reader::inspect_first(&blob).unwrap();
    let offset = body_offset(&blob, &info);
    assert_eq!(blob[offset], 0);
    assert_eq!(blob[offset + 1], 0);
    assert_eq!(blob[offset + 2] & 3, 2);
}

fn huffman_band_set_roundtrip(data: &[u8]) {
    let width = 16usize;
    let height = 16usize;
    let band_count = 2usize;
    let low = data.first().copied().unwrap_or(7) % 200;
    let high = low + 2;
    let mut values = Vec::with_capacity(width * height * band_count);
    for pixel in 0..(width * height) {
        let repeated = if pixel % 64 < 48 { low } else { high };
        values.push(repeated);
        values.push(repeated.wrapping_add(1));
    }

    let Some(blob) = encode_and_check_u8_band_set(
        width,
        height,
        1,
        band_count,
        BandLayout::Interleaved,
        &values,
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(2),
    ) else {
        return;
    };
    let info = lerc_reader::inspect_first(&blob).unwrap();
    let offset = body_offset(&blob, &info);
    assert_eq!(blob[offset], 0);
    assert_ne!(blob[offset + 1], 0);
}

fn diff_tile_band_set_roundtrip(data: &[u8]) {
    let width = 4usize;
    let height = 2usize;
    let depth = 2usize;
    let band_count = 2usize;
    let mut values = Vec::with_capacity(width * height * depth * band_count);
    for pixel in 0..(width * height) {
        let base = (pixel as u16)
            .wrapping_mul(3)
            .wrapping_add(u16::from(data[pixel % data.len()]));
        values.push(base);
        values.push(base);
        values.push(base.wrapping_add(3));
        values.push(base.wrapping_add(3));
    }
    let band_set = BandSetView::new(
        width as u32,
        height as u32,
        depth as u32,
        band_count,
        BandLayout::Interleaved,
        &values,
    )
    .unwrap();
    let blob = lerc_writer::encode_band_set(
        band_set,
        None,
        lerc_writer::EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(8),
    )
    .unwrap();

    assert_eq!(
        lerc_reader::inspect_first(&blob).unwrap().version,
        lerc_core::Version::Lerc2(5)
    );
    assert_direct_band_set_ndarrays_match::<u16>(&blob);
}

fn arbitrary_typed_roundtrip<T: lerc_core::Sample>(
    data: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    mask_bytes: &[u8],
) {
    let sample_count = width * height * depth;
    let data_type = T::DATA_TYPE;
    let sentinel = match data_type {
        lerc_core::DataType::I8 => -120.0,
        lerc_core::DataType::U8 => 250.0,
        lerc_core::DataType::I16 => -30_000.0,
        lerc_core::DataType::U16 => 65_000.0,
        lerc_core::DataType::I32 => -2_000_000_000.0,
        lerc_core::DataType::U32 => 4_000_000_000.0,
        lerc_core::DataType::F32 | lerc_core::DataType::F64 => -9_999.0,
    };
    let use_no_data = depth > 1 && data[3] & 2 != 0;
    let values: Vec<T> = (0..sample_count)
        .map(|index| {
            let byte = data[4 + index];
            let value = match data_type {
                lerc_core::DataType::I8 => f64::from(byte as i8),
                lerc_core::DataType::U8 => f64::from(byte),
                lerc_core::DataType::I16 => f64::from(byte as i8) * 257.0,
                lerc_core::DataType::U16 => f64::from(byte) * 257.0,
                lerc_core::DataType::I32 => f64::from(byte as i8) * 1_000_003.0,
                lerc_core::DataType::U32 => f64::from(byte) * 16_000_001.0,
                lerc_core::DataType::F32 | lerc_core::DataType::F64 => {
                    f64::from(byte as i8) * 0.25 + (index % depth) as f64 * 0.01
                }
            };
            T::from_f64(if use_no_data && byte & 7 == 0 {
                sentinel
            } else {
                value
            })
        })
        .collect();
    let mask = if mask_bytes.iter().all(|&byte| byte != 0) {
        None
    } else {
        Some(
            lerc_core::MaskView::new(width as u32, height as u32, mask_bytes).unwrap(),
        )
    };
    let mut options = lerc_writer::EncodeOptions::new()
        .with_max_z_error(if data[3] & 1 == 0 { 0.0 } else { 0.5 })
        .with_micro_block_size(u32::from(data[3] % 8) + 2);
    if use_no_data {
        options = options.with_no_data_value(sentinel);
    }
    let raster = lerc_core::RasterView::new(
        width as u32,
        height as u32,
        depth as u32,
        &values,
    )
    .unwrap();
    let blob = lerc_writer::encode(raster, mask, options)
        .expect("valid generated raster must be encodable");
    let decoded = lerc_reader::decode_to_f64(&blob)
        .expect("writer produced a blob the reader rejected");
    assert_eq!(decoded.pixels.len(), sample_count);
    if let Some(mask) = decoded.mask.as_ref() {
        assert_eq!(mask.len(), width * height);
    }
    let tolerance = if T::IS_INTEGER { 0.0 } else { options.max_z_error } + 1e-6;
    for pixel in 0..(width * height) {
        let input_valid = mask_bytes[pixel] != 0;
        let pixel_values = &values[pixel * depth..(pixel + 1) * depth];
        let all_no_data = use_no_data
            && pixel_values
                .iter()
                .all(|value| value.to_f64() == sentinel);
        let expected_valid = input_valid && !all_no_data;
        let decoded_valid = decoded
            .mask
            .as_ref()
            .map_or(true, |mask| mask[pixel] != 0);
        assert_eq!(decoded_valid, expected_valid);
        if !expected_valid {
            continue;
        }

        for (dim, expected) in pixel_values.iter().enumerate() {
            let expected = expected.to_f64();
            let actual = decoded.pixels[pixel * depth + dim];
            if use_no_data && expected == sentinel {
                assert_eq!(actual, sentinel);
            } else {
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }
}

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

    let mask_bytes = &data[4 + sample_count..4 + sample_count + pixel_count];
    match data[3] >> 5 {
        0 => arbitrary_typed_roundtrip::<i8>(data, width, height, depth, mask_bytes),
        1 => arbitrary_typed_roundtrip::<u8>(data, width, height, depth, mask_bytes),
        2 => arbitrary_typed_roundtrip::<i16>(data, width, height, depth, mask_bytes),
        3 => arbitrary_typed_roundtrip::<u16>(data, width, height, depth, mask_bytes),
        4 => arbitrary_typed_roundtrip::<i32>(data, width, height, depth, mask_bytes),
        5 => arbitrary_typed_roundtrip::<u32>(data, width, height, depth, mask_bytes),
        6 => arbitrary_typed_roundtrip::<f32>(data, width, height, depth, mask_bytes),
        _ => arbitrary_typed_roundtrip::<f64>(data, width, height, depth, mask_bytes),
    }

    generated_band_set_roundtrips(data);
    zero_block_band_set_roundtrip(data);
    huffman_band_set_roundtrip(data);
    diff_tile_band_set_roundtrip(data);
});
