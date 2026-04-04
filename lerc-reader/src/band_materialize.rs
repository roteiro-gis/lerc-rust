use std::mem::MaybeUninit;

use lerc_core::{BandLayout, BandSetInfo, Error, NdArrayElement, Result};

pub(crate) struct BandMaterializer<T> {
    info: BandSetInfo,
    layout: BandLayout,
    out: Vec<MaybeUninit<T>>,
    written_bands: Vec<bool>,
}

impl<T: Clone> BandMaterializer<T> {
    pub(crate) fn new(info: &BandSetInfo, layout: BandLayout) -> Result<Self> {
        let sample_count = info.value_count()?;
        let mut out = Vec::with_capacity(sample_count);
        if sample_count != 0 {
            unsafe {
                out.set_len(sample_count);
            }
        }
        Ok(Self {
            info: info.clone(),
            layout,
            out,
            written_bands: vec![false; info.band_count()],
        })
    }

    pub(crate) fn copy_band(&mut self, band_index: usize, values: &[T]) -> Result<()> {
        if band_index >= self.info.band_count() {
            return Err(Error::InvalidBlob(format!(
                "band index {} exceeds band count {}",
                band_index,
                self.info.band_count()
            )));
        }
        if self.written_bands[band_index] {
            return Err(Error::InvalidBlob(format!(
                "band index {} was materialized more than once",
                band_index
            )));
        }

        let pixel_count = self.info.bands[0].pixel_count()?;
        let depth = self.info.depth() as usize;
        write_band_values_into_uninit_slice(
            &mut self.out,
            values,
            pixel_count,
            depth,
            band_index,
            self.info.band_count(),
            self.layout,
        )?;
        self.written_bands[band_index] = true;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Vec<T>> {
        let mut this = self;
        if this.written_bands.iter().any(|written| !written) {
            return Err(Error::InvalidBlob(
                "not all decoded bands were materialized into the output buffer".into(),
            ));
        }

        let out = std::mem::take(&mut this.out);
        this.written_bands.fill(false);
        Ok(unsafe { assume_init_vec(out) })
    }
}

impl<T> Drop for BandMaterializer<T> {
    fn drop(&mut self) {
        if self.out.is_empty() || self.written_bands.iter().all(|written| !written) {
            return;
        }

        let Ok(pixel_count) = self.info.bands[0].pixel_count() else {
            return;
        };
        let depth = (self.info.depth() as usize).max(1);
        let band_count = self.info.band_count();

        for (band_index, written) in self.written_bands.iter().copied().enumerate() {
            if !written {
                continue;
            }
            drop_written_band(
                &mut self.out,
                pixel_count,
                depth,
                band_index,
                band_count,
                self.layout,
            );
        }
    }
}

pub(crate) fn copy_band_values_into_slice<T: NdArrayElement>(
    out: &mut [T],
    values: &[T],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
) -> Result<()> {
    let band_len = pixel_count
        .checked_mul(depth.max(1))
        .ok_or_else(|| Error::InvalidBlob("decoded band length overflows usize".into()))?;
    if values.len() != band_len {
        return Err(Error::InvalidBlob(
            "decoded band length does not match its metadata".into(),
        ));
    }

    match layout {
        BandLayout::Interleaved => {
            if depth <= 1 {
                for pixel in 0..pixel_count {
                    out[pixel * band_count + band_index] = values[pixel].clone();
                }
            } else {
                for pixel in 0..pixel_count {
                    let src_base = pixel * depth;
                    let dst_base = (pixel * band_count + band_index) * depth;
                    out[dst_base..dst_base + depth]
                        .clone_from_slice(&values[src_base..src_base + depth]);
                }
            }
        }
        BandLayout::Bsq => {
            let dst_base = band_index * band_len;
            out[dst_base..dst_base + band_len].clone_from_slice(values);
        }
    }

    Ok(())
}

fn write_band_values_into_uninit_slice<T: Clone>(
    out: &mut [MaybeUninit<T>],
    values: &[T],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
) -> Result<()> {
    let band_len = pixel_count
        .checked_mul(depth.max(1))
        .ok_or_else(|| Error::InvalidBlob("decoded band length overflows usize".into()))?;
    if values.len() != band_len {
        return Err(Error::InvalidBlob(
            "decoded band length does not match its metadata".into(),
        ));
    }

    match layout {
        BandLayout::Interleaved => {
            if depth <= 1 {
                for pixel in 0..pixel_count {
                    out[pixel * band_count + band_index].write(values[pixel].clone());
                }
            } else {
                for pixel in 0..pixel_count {
                    let src_base = pixel * depth;
                    let dst_base = (pixel * band_count + band_index) * depth;
                    for offset in 0..depth {
                        out[dst_base + offset].write(values[src_base + offset].clone());
                    }
                }
            }
        }
        BandLayout::Bsq => {
            let dst_base = band_index * band_len;
            for (index, value) in values.iter().enumerate() {
                out[dst_base + index].write(value.clone());
            }
        }
    }

    Ok(())
}

unsafe fn assume_init_vec<T>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
    let len = values.len();
    let cap = values.capacity();
    let ptr = values.as_ptr() as *mut T;
    std::mem::forget(values);
    Vec::from_raw_parts(ptr, len, cap)
}

fn drop_written_band<T>(
    out: &mut [MaybeUninit<T>],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
) {
    match layout {
        BandLayout::Interleaved => {
            if depth <= 1 {
                for pixel in 0..pixel_count {
                    unsafe {
                        out[pixel * band_count + band_index].assume_init_drop();
                    }
                }
            } else {
                for pixel in 0..pixel_count {
                    let dst_base = (pixel * band_count + band_index) * depth;
                    for offset in 0..depth {
                        unsafe {
                            out[dst_base + offset].assume_init_drop();
                        }
                    }
                }
            }
        }
        BandLayout::Bsq => {
            let band_len = pixel_count * depth;
            let dst_base = band_index * band_len;
            for index in 0..band_len {
                unsafe {
                    out[dst_base + index].assume_init_drop();
                }
            }
        }
    }
}
