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
    HELPER
        .get_or_init(|| {
            std::env::var_os("LERC_READER_REFERENCE_HELPER")
                .map(PathBuf::from)
                .filter(|path| path.is_file())
        })
        .clone()
}

pub fn run_reference_json(helper: &Path, args: &[&str]) -> Value {
    let output = Command::new(helper)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run LERC reference helper: {err}"));
    assert!(
        output.status.success(),
        "LERC reference helper failed: {}",
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
    fn append_ne_bytes(&self, out: &mut Vec<u8>);
}

impl SampleBytes for u8 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
}

impl SampleBytes for i8 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}

impl SampleBytes for u16 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i16 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for u32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for i32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f32 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl SampleBytes for f64 {
    fn append_ne_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
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
