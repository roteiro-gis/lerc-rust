#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn synthetic_u8() -> Vec<u8> {
    (0..(256 * 256)).map(|index| (index % 251) as u8).collect()
}

fn synthetic_f32() -> Vec<f32> {
    (0..(256 * 256))
        .map(|index| ((index % 1024) as f32) * 0.25)
        .collect()
}

fn encode_benchmarks(c: &mut Criterion) {
    let u8_pixels = synthetic_u8();
    let u8_raster = lerc_core::RasterView::new(256, 256, 1, &u8_pixels).unwrap();
    let u8_options = lerc_writer::EncodeOptions::new()
        .with_max_z_error(0.5)
        .with_micro_block_size(8);
    c.bench_function("lerc-writer/encode/u8-bitstuff", |b| {
        b.iter(|| {
            let blob =
                lerc_writer::encode(black_box(u8_raster), None, black_box(u8_options)).unwrap();
            black_box(blob);
        });
    });

    let f32_pixels = synthetic_f32();
    let f32_raster = lerc_core::RasterView::new(256, 256, 1, &f32_pixels).unwrap();
    let f32_options = lerc_writer::EncodeOptions::new()
        .with_max_z_error(0.125)
        .with_micro_block_size(8);
    c.bench_function("lerc-writer/encode-plus-decode/f32", |b| {
        b.iter(|| {
            let blob =
                lerc_writer::encode(black_box(f32_raster), None, black_box(f32_options)).unwrap();
            let raster = lerc_reader::decode_ndarray_f64(black_box(&blob)).unwrap();
            black_box(raster);
        });
    });
}

criterion_group!(benches, encode_benchmarks);
criterion_main!(benches);
