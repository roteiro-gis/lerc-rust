#![allow(missing_docs)]

use lerc_core::{BandLayout, DataType, PixelData};

#[test]
fn signed_delta_huffman_preserves_negative_values() {
    let cases: [(&[u8], usize, bool); 4] = [
        (
            include_bytes!("../../testdata/interoperability/liblerc-v4-i8-delta-1d.lerc2"),
            1,
            false,
        ),
        (
            include_bytes!("../../testdata/interoperability/liblerc-v4-i8-delta-1d-masked.lerc2"),
            1,
            true,
        ),
        (
            include_bytes!("../../testdata/interoperability/liblerc-v4-i8-delta-3d.lerc2"),
            3,
            false,
        ),
        (
            include_bytes!("../../testdata/interoperability/liblerc-v4-i8-delta-3d-masked.lerc2"),
            3,
            true,
        ),
    ];
    for (blob, depth, masked) in cases {
        let info = lerc_reader::get_blob_info(blob).unwrap();
        assert_eq!(info.data_type, DataType::I8);
        assert_eq!(
            (info.width, info.height, info.depth),
            (64, 16, depth as u32)
        );
        // Pin delta Huffman mode; plain Huffman does not exercise wrapping.
        let mask_len = u32::from_le_bytes(blob[66..70].try_into().unwrap()) as usize;
        let body_offset = 70 + mask_len + 2 * depth;
        assert_eq!(&blob[body_offset..body_offset + 2], &[0, 1]);

        let mask: Vec<u8> = (0..64 * 16)
            .map(|pixel| u8::from(!masked || (pixel % 11 != 0 && pixel % 17 != 0)))
            .collect();
        let expected: Vec<i8> = (0..64 * 16 * depth)
            .map(|index| {
                let (pixel, dim) = (index / depth, index % depth);
                if mask[pixel] != 0 {
                    (((pixel + dim * 73) % 256) as i16 - 128) as i8
                } else {
                    0
                }
            })
            .collect();
        let decoded = lerc_reader::decode(blob).unwrap();
        assert_eq!(decoded.mask.as_deref(), masked.then_some(mask.as_slice()));
        assert_eq!(
            decoded.pixels,
            PixelData::I8(expected.clone()),
            "depth={depth}, masked={masked}"
        );

        let promoted = lerc_reader::decode_to_f64(blob).unwrap();
        assert_eq!(
            promoted.pixels,
            expected.iter().copied().map(f64::from).collect::<Vec<_>>()
        );

        for layout in [BandLayout::Interleaved, BandLayout::Bsq] {
            let mut output = vec![0i8; expected.len()];
            lerc_reader::decode_band_set_into(blob, layout, &mut output).unwrap();
            assert_eq!(
                output, expected,
                "depth={depth}, masked={masked}, layout={layout:?}"
            );
        }
    }
}
