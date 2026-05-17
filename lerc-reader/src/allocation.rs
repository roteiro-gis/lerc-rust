use std::mem;

use lerc_core::{Error, Result};

const MAX_DECODED_ALLOCATION_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn checked_mul(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::InvalidBlob(format!("{label} overflows usize")))
}

pub(crate) fn check_allocation<T>(len: usize, label: &str) -> Result<()> {
    let bytes = checked_mul(len, mem::size_of::<T>(), label)?;
    if bytes > MAX_DECODED_ALLOCATION_BYTES {
        return Err(Error::InvalidBlob(format!(
            "{label} allocation request of {bytes} bytes exceeds decoder limit of {MAX_DECODED_ALLOCATION_BYTES} bytes"
        )));
    }
    Ok(())
}

pub(crate) fn default_vec<T: Default + Clone>(len: usize, label: &str) -> Result<Vec<T>> {
    check_allocation::<T>(len, label)?;
    Ok(vec![T::default(); len])
}

pub(crate) fn vec_with_capacity<T>(len: usize, label: &str) -> Result<Vec<T>> {
    check_allocation::<T>(len, label)?;
    Ok(Vec::with_capacity(len))
}
