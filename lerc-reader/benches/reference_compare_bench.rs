use std::path::Path;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::ArrayD;

#[path = "../../test-support/reference.rs"]
mod reference;

const RUST_IMPL_NAME: &str = "lerc-rust";
const REFERENCE_IMPL_NAME: &str = "libLerc";

#[derive(Clone, Copy)]
enum FixtureKind {
    F32,
    BandSetU8,
}

fn fixture(path: &str) -> std::path::PathBuf {
    reference::fixture(env!("CARGO_MANIFEST_DIR"), path)
}

fn load_blob(path: &Path) -> Vec<u8> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("csv") {
        std::fs::read_to_string(path)
            .unwrap()
            .trim()
            .split(',')
            .map(|value| value.parse::<u8>().unwrap())
            .collect()
    } else {
        std::fs::read(path).unwrap()
    }
}

fn rust_hash(path: &Path, kind: FixtureKind) -> (usize, String) {
    let blob = load_blob(path);
    match kind {
        FixtureKind::F32 => {
            let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
            reference::array_hash(&raster)
        }
        FixtureKind::BandSetU8 => {
            let raster: ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob).unwrap();
            reference::array_hash(&raster)
        }
    }
}

fn rust_checksum(path: &Path, kind: FixtureKind) -> (f64, usize) {
    let blob = load_blob(path);
    match kind {
        FixtureKind::F32 => {
            let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
            let checksum = raster.iter().map(|value| f64::from(*value)).sum::<f64>();
            let valid_sum = lerc_reader::decode_mask_ndarray(&blob)
                .unwrap()
                .map(|mask| mask.iter().map(|value| usize::from(*value)).sum::<usize>())
                .unwrap_or(0);
            (checksum, valid_sum)
        }
        FixtureKind::BandSetU8 => {
            let raster: ArrayD<u8> = lerc_reader::decode_band_set_ndarray(&blob).unwrap();
            let checksum = raster.iter().map(|value| f64::from(*value)).sum::<f64>();
            let valid_sum = lerc_reader::decode_band_mask_ndarray(&blob)
                .unwrap()
                .map(|mask| mask.iter().map(|value| usize::from(*value)).sum::<usize>())
                .unwrap_or(0);
            (checksum, valid_sum)
        }
    }
}

fn reference_hash(helper: &Path, fixture_path: &str) -> (usize, String) {
    let reference_json = reference::run_reference_json(helper, &["hash", fixture_path]);
    (
        reference_json["pixel_byte_len"].as_u64().unwrap() as usize,
        reference_json["pixel_hash"].as_str().unwrap().to_string(),
    )
}

fn reference_benchmark_duration(
    helper: &Path,
    fixture_path: &str,
    iterations: usize,
    expected_hash: &(usize, String),
    expected_checksum: f64,
    expected_valid_sum: usize,
) -> Duration {
    let iterations_arg = iterations.to_string();
    let reference_json = reference::run_reference_json(
        helper,
        &["benchmark", fixture_path, "--iterations", &iterations_arg],
    );

    assert_eq!(
        reference_json["pixel_byte_len"].as_u64().unwrap() as usize,
        expected_hash.0
    );
    assert_eq!(
        reference_json["pixel_hash"].as_str().unwrap(),
        expected_hash.1
    );
    assert_eq!(
        reference_json["valid_sum"].as_u64().unwrap() as usize,
        expected_valid_sum * iterations
    );
    assert!(
        (reference_json["checksum"].as_f64().unwrap() - expected_checksum * iterations as f64)
            .abs()
            <= expected_checksum.abs().max(1.0) * 1e-12 * iterations as f64,
        "reference checksum drift for {fixture_path}"
    );

    Duration::from_secs_f64(reference_json["total_seconds"].as_f64().unwrap())
}

fn bench_fixture(c: &mut Criterion, fixture_name: &str, kind: FixtureKind, group_name: &str) {
    let path = fixture(fixture_name);
    let blob = load_blob(&path);
    let helper = reference::helper_path()
        .expect("LERC_READER_REFERENCE_HELPER should be configured before calling bench_fixture");
    let fixture_path = path.to_str().unwrap().to_string();

    let expected_hash = rust_hash(&path, kind);
    assert_eq!(
        reference_hash(&helper, &fixture_path),
        expected_hash,
        "reference hash mismatch for {fixture_name}"
    );
    let (expected_checksum, expected_valid_sum) = rust_checksum(&path, kind);

    let mut decode_only_group = c.benchmark_group(format!("{group_name}/decode-only"));
    decode_only_group.throughput(Throughput::Bytes(blob.len() as u64));

    decode_only_group.bench_function(BenchmarkId::new(RUST_IMPL_NAME, fixture_name), |b| {
        b.iter(|| match kind {
            FixtureKind::F32 => {
                let raster: ArrayD<f32> = lerc_reader::decode_ndarray(black_box(&blob)).unwrap();
                black_box(raster);
            }
            FixtureKind::BandSetU8 => {
                let raster: ArrayD<u8> =
                    lerc_reader::decode_band_set_ndarray(black_box(&blob)).unwrap();
                black_box(raster);
            }
        });
    });
    decode_only_group.finish();

    let mut group = c.benchmark_group(format!("{group_name}/load-plus-decode"));
    group.throughput(Throughput::Bytes(blob.len() as u64));

    group.bench_function(BenchmarkId::new(RUST_IMPL_NAME, fixture_name), |b| {
        b.iter_custom(|iters| {
            let iterations = usize::try_from(iters).expect("criterion iteration count overflowed");
            let start = Instant::now();
            for _ in 0..iterations {
                let blob = load_blob(&path);
                match kind {
                    FixtureKind::F32 => {
                        let raster: ArrayD<f32> = lerc_reader::decode_ndarray(&blob).unwrap();
                        black_box(raster);
                    }
                    FixtureKind::BandSetU8 => {
                        let raster: ArrayD<u8> =
                            lerc_reader::decode_band_set_ndarray(&blob).unwrap();
                        black_box(raster);
                    }
                }
            }
            start.elapsed()
        });
    });

    group.bench_function(BenchmarkId::new(REFERENCE_IMPL_NAME, fixture_name), |b| {
        b.iter_custom(|iters| {
            let iterations = usize::try_from(iters).expect("criterion iteration count overflowed");
            reference_benchmark_duration(
                &helper,
                &fixture_path,
                iterations,
                &expected_hash,
                expected_checksum,
                expected_valid_sum,
            )
        });
    });

    group.finish();
}

fn compare_against_liblerc(c: &mut Criterion) {
    if reference::helper_path().is_none() {
        eprintln!("skipping libLerc benchmark because LERC_READER_REFERENCE_HELPER is unset");
        return;
    }

    bench_fixture(
        c,
        "world.lerc1",
        FixtureKind::F32,
        "lerc-reader/full-decode-vs-libLerc",
    );
    bench_fixture(
        c,
        "california_400_400_1_float.lerc2",
        FixtureKind::F32,
        "lerc-reader/full-decode-vs-libLerc",
    );
    bench_fixture(
        c,
        "bluemarble_256_256_3_byte.lerc2",
        FixtureKind::BandSetU8,
        "lerc-reader/band-set-decode-vs-libLerc",
    );
}

criterion_group!(benches, compare_against_liblerc);
criterion_main!(benches);
