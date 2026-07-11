#![allow(missing_docs)]

use lerc_core::{BandLayout, BandSetView, RasterView};
use lerc_writer::{
    encode, encoded_band_set_len_upper_bound, encoded_len_upper_bound, EncodeOptions,
};

fn sample_blob() -> Vec<u8> {
    let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    encode(
        RasterView::new(4, 2, 1, &pixels).unwrap(),
        None,
        EncodeOptions::new()
            .with_max_z_error(0.5)
            .with_micro_block_size(2),
    )
    .unwrap()
}

#[test]
fn rejects_blob_with_corrupted_checksum() {
    let mut blob = sample_blob();
    blob[10..14].fill(0);

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(lerc_core::Error::ChecksumMismatch { .. })
    ));
}

#[test]
fn rejects_blob_with_oversized_declared_length() {
    let mut blob = sample_blob();
    let declared = (blob.len() as i32 + 1).to_le_bytes();
    blob[34..38].copy_from_slice(&declared);

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(lerc_core::Error::Truncated { .. })
    ));
}

#[test]
fn rejects_blob_with_invalid_mask_length() {
    let pixels = vec![1u8, 2, 3, 4];
    let mask = vec![1u8, 0, 1, 1];
    let mut blob = encode(
        RasterView::new(2, 2, 1, &pixels).unwrap(),
        Some(lerc_core::MaskView::new(2, 2, &mask).unwrap()),
        EncodeOptions::default(),
    )
    .unwrap();

    blob[66..70].copy_from_slice(&1u32.to_le_bytes());
    let checksum = lerc_core::fletcher32(&blob[14..]);
    blob[10..14].copy_from_slice(&checksum.to_le_bytes());

    assert!(matches!(
        lerc_reader::decode(&blob),
        Err(lerc_core::Error::Truncated { .. })
    ));
}

#[test]
fn rejects_no_data_for_unit_depth_rasters() {
    let pixels = vec![1.0f32, -9999.0, 3.0, 4.0];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();
    let options = EncodeOptions::new().with_no_data_value(-9999.0);

    assert!(matches!(
        encode(raster, None, options),
        Err(lerc_core::Error::InvalidArgument(_))
    ));
    assert!(matches!(
        encoded_len_upper_bound(raster, None, options),
        Err(lerc_core::Error::InvalidArgument(_))
    ));

    let band_set = BandSetView::new(2, 2, 1, 1, BandLayout::Bsq, &pixels).unwrap();
    assert!(matches!(
        encoded_band_set_len_upper_bound(band_set, None, options),
        Err(lerc_core::Error::InvalidArgument(_))
    ));
}

#[test]
fn rejects_non_finite_no_data_value() {
    let pixels = vec![1.0f32, -9999.0, 3.0, 4.0];

    assert!(matches!(
        encode(
            RasterView::new(1, 2, 2, &pixels).unwrap(),
            None,
            EncodeOptions::new().with_no_data_value(f64::NAN),
        ),
        Err(lerc_core::Error::InvalidArgument(_))
    ));
}

#[test]
fn rejects_micro_block_sizes_outside_the_supported_range() {
    let pixels = vec![1u8, 2, 3, 4];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();

    for micro_block_size in [0, 1, 65, u32::MAX] {
        let result = encode(
            raster,
            None,
            EncodeOptions::new().with_micro_block_size(micro_block_size),
        );
        assert!(
            matches!(result, Err(lerc_core::Error::InvalidArgument(_))),
            "micro_block_size={micro_block_size}: {result:?}"
        );
    }
}

#[test]
fn rejects_no_data_outside_the_raster_type_range() {
    let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let raster = RasterView::new(2, 2, 2, &pixels).unwrap();
    let result = encode(raster, None, EncodeOptions::new().with_no_data_value(256.0));
    assert!(matches!(result, Err(lerc_core::Error::InvalidArgument(_))));
}
