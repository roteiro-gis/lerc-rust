use lerc_core::RasterView;
use lerc_writer::{encode, EncodeOptions};

fn sample_blob() -> Vec<u8> {
    let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    encode(
        RasterView::new(4, 2, 1, &pixels).unwrap(),
        None,
        EncodeOptions {
            max_z_error: 0.5,
            micro_block_size: 2,
        },
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
        Err(lerc_core::Error::InvalidBlob(_) | lerc_core::Error::Truncated { .. })
    ));
}
