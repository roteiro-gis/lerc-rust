#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use ndarray::ArrayD;
use serde_json::Value;

pub fn workspace_root(manifest_dir: &str) -> PathBuf {
    Path::new(manifest_dir)
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(manifest_dir).join(".."))
}

pub fn fixture(manifest_dir: &str, relative_path: &str) -> PathBuf {
    workspace_root(manifest_dir)
        .join("testdata")
        .join("interoperability")
        .join(relative_path)
}

pub fn write_temp_bytes(prefix: &str, extension: &str, bytes: &[u8]) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("{prefix}-{nanos}.{extension}"));
    std::fs::write(&path, bytes).unwrap();
    path
}

pub fn helper_path() -> Option<PathBuf> {
    static HELPER: OnceLock<Option<PathBuf>> = OnceLock::new();
    let helper = HELPER
        .get_or_init(|| {
            std::env::var_os("LERC_READER_REFERENCE_HELPER")
                .map(PathBuf::from)
                .filter(|path| path.is_file())
        })
        .clone();
    if helper.is_none() && std::env::var_os("LERC_PARITY_REQUIRED").is_some() {
        panic!(
            "LERC_PARITY_REQUIRED is set but LERC_READER_REFERENCE_HELPER is missing or invalid"
        );
    }
    helper
}

pub fn run_reference_json(helper: &Path, args: &[&str]) -> Value {
    let output = Command::new(helper)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run LERC reference helper: {err}"));
    assert!(
        output.status.success(),
        "LERC reference helper failed for {:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|err| panic!("failed to parse LERC reference JSON: {err}"))
}

pub fn fnv1a64(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

pub trait SampleBytes {
    const LERC_DATA_TYPE: u8;

    fn append_ne_bytes(&self, out: &mut Vec<u8>);
}

impl SampleBytes for u8 {
    const LERC_DATA_TYPE: u8 = 1;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
}

impl SampleBytes for i8 {
    const LERC_DATA_TYPE: u8 = 0;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}

impl SampleBytes for u16 {
    const LERC_DATA_TYPE: u8 = 3;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i16 {
    const LERC_DATA_TYPE: u8 = 2;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for u32 {
    const LERC_DATA_TYPE: u8 = 5;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i32 {
    const LERC_DATA_TYPE: u8 = 4;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f32 {
    const LERC_DATA_TYPE: u8 = 6;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f64 {
    const LERC_DATA_TYPE: u8 = 7;

    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReferenceEncodeOptions {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub max_z_error: f64,
    pub codec_version: u8,
    pub no_data_value: Option<f64>,
}

pub fn encode_with_reference<T: SampleBytes>(
    helper: &Path,
    samples: &[T],
    mask: Option<&[u8]>,
    options: ReferenceEncodeOptions,
) -> Vec<u8> {
    let pixel_count = options
        .width
        .checked_mul(options.height)
        .expect("reference raster pixel count overflowed");
    let sample_count = pixel_count
        .checked_mul(options.depth)
        .expect("reference raster sample count overflowed");
    assert_eq!(samples.len(), sample_count);
    if let Some(mask) = mask {
        assert_eq!(mask.len(), pixel_count);
    }

    let mut sample_bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        sample.append_ne_bytes(&mut sample_bytes);
    }
    let input_path = write_temp_bytes("lerc-reference-input", "bin", &sample_bytes);
    let output_path = write_temp_bytes("lerc-reference-output", "lerc2", &[]);
    let mask_path = mask.map(|mask| write_temp_bytes("lerc-reference-mask", "bin", mask));
    let mask_count = usize::from(mask.is_some()).to_string();
    let mask_arg = mask_path
        .as_deref()
        .map_or_else(|| std::ffi::OsStr::new("-"), Path::as_os_str);
    let no_data = options
        .no_data_value
        .map_or_else(|| "-".to_owned(), |value| value.to_string());

    let output = Command::new(helper)
        .arg("encode")
        .arg(&input_path)
        .arg(&output_path)
        .arg(T::LERC_DATA_TYPE.to_string())
        .arg(options.depth.to_string())
        .arg(options.width.to_string())
        .arg(options.height.to_string())
        .arg("1")
        .arg(mask_count)
        .arg(mask_arg)
        .arg(options.max_z_error.to_string())
        .arg(options.codec_version.to_string())
        .arg(no_data)
        .output()
        .unwrap_or_else(|err| panic!("failed to run LERC reference encoder: {err}"));
    assert!(
        output.status.success(),
        "LERC reference encoder failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let encoded = std::fs::read(&output_path).expect("reference encoder did not write its blob");
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|err| panic!("failed to parse LERC reference encoder JSON: {err}"));
    assert_eq!(
        metadata["blob_size"].as_u64().unwrap() as usize,
        encoded.len()
    );
    assert_eq!(metadata["blob_hash"].as_str().unwrap(), fnv1a64(&encoded));

    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(output_path);
    if let Some(mask_path) = mask_path {
        let _ = std::fs::remove_file(mask_path);
    }
    encoded
}

pub fn array_hash<T: SampleBytes>(array: &ArrayD<T>) -> (usize, String) {
    let element_size = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(array.len() * element_size);
    for value in array {
        value.append_ne_bytes(&mut bytes);
    }
    let len = bytes.len();
    (len, fnv1a64(&bytes))
}
