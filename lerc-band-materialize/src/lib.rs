use std::fmt;
use std::mem::MaybeUninit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandLayout {
    Interleaved,
    Bsq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandWriteOrder {
    PixelMajor,
    DimMajor,
    Arbitrary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeError {
    message: String,
}

impl MaterializeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MaterializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MaterializeError {}

pub type Result<T> = std::result::Result<T, MaterializeError>;

pub trait BandWriter<T: Copy + Default> {
    fn fill_default(&mut self);
    fn write(&mut self, pixel: usize, dim: usize, value: T);
    fn read(&self, pixel: usize, dim: usize) -> T;
    fn depth(&self) -> usize;
    fn set_write_order(&mut self, _order: BandWriteOrder) {}
}

pub fn copy_band_values_into_slice<T: Clone>(
    out: &mut [T],
    values: &[T],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
) -> Result<()> {
    let band_len = band_len(pixel_count, depth)?;
    if values.len() != band_len {
        return Err(MaterializeError::new(
            "decoded band length does not match its metadata",
        ));
    }
    if band_index >= band_count {
        return Err(MaterializeError::new(format!(
            "band index {band_index} exceeds band count {band_count}"
        )));
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

#[derive(Debug)]
pub struct BandSink<'a, T> {
    out: &'a mut [T],
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
}

impl<'a, T: Copy + Default> BandSink<'a, T> {
    pub fn new(
        out: &'a mut [T],
        pixel_count: usize,
        depth: usize,
        band_index: usize,
        band_count: usize,
        layout: BandLayout,
    ) -> Self {
        Self {
            out,
            shape: BandShape {
                pixel_count,
                depth: depth.max(1),
                band_count,
            },
            band_index,
            layout,
        }
    }

    pub fn fill_default(&mut self) {
        match self.layout {
            BandLayout::Interleaved => {
                for pixel in 0..self.shape.pixel_count {
                    let base = (pixel * self.shape.band_count + self.band_index) * self.shape.depth;
                    self.out[base..base + self.shape.depth].fill(T::default());
                }
            }
            BandLayout::Bsq => {
                let band_len = self.shape.pixel_count * self.shape.depth;
                let base = self.band_index * band_len;
                self.out[base..base + band_len].fill(T::default());
            }
        }
    }

    pub fn write(&mut self, pixel: usize, dim: usize, value: T) {
        let index =
            band_value_index_for_pixel(self.shape, self.band_index, self.layout, pixel, dim);
        self.out[index] = value;
    }

    pub fn read(&self, pixel: usize, dim: usize) -> T {
        self.out[band_value_index_for_pixel(self.shape, self.band_index, self.layout, pixel, dim)]
    }

    pub fn depth(&self) -> usize {
        self.shape.depth
    }
}

impl<T: Copy + Default> BandWriter<T> for BandSink<'_, T> {
    fn fill_default(&mut self) {
        Self::fill_default(self);
    }

    fn write(&mut self, pixel: usize, dim: usize, value: T) {
        Self::write(self, pixel, dim, value);
    }

    fn read(&self, pixel: usize, dim: usize) -> T {
        Self::read(self, pixel, dim)
    }

    fn depth(&self) -> usize {
        Self::depth(self)
    }
}

pub struct BandMaterializer<T> {
    shape: BandShape,
    layout: BandLayout,
    out: Vec<MaybeUninit<T>>,
    written_bands: Vec<bool>,
    active_band: Option<ActiveBand>,
}

#[derive(Debug, Clone, Copy)]
struct BandShape {
    pixel_count: usize,
    depth: usize,
    band_count: usize,
}

#[derive(Debug, Clone)]
struct ActiveBand {
    band_index: usize,
    order: BandWriteOrder,
    progress: ActiveBandProgress,
}

#[derive(Debug, Clone)]
enum ActiveBandProgress {
    Prefix(usize),
    Sparse(Vec<bool>),
}

impl<T> BandMaterializer<T> {
    pub fn new(
        pixel_count: usize,
        depth: usize,
        band_count: usize,
        layout: BandLayout,
    ) -> Result<Self> {
        let sample_count = total_len(pixel_count, depth, band_count)?;
        let mut out = Vec::with_capacity(sample_count);
        if sample_count != 0 {
            unsafe {
                out.set_len(sample_count);
            }
        }
        Ok(Self {
            shape: BandShape {
                pixel_count,
                depth,
                band_count,
            },
            layout,
            out,
            written_bands: vec![false; band_count],
            active_band: None,
        })
    }

    pub fn copy_band_with<F>(&mut self, band_index: usize, mut value_at: F) -> Result<()>
    where
        F: FnMut(usize) -> T,
    {
        if band_index >= self.shape.band_count {
            return Err(MaterializeError::new(format!(
                "band index {band_index} exceeds band count {}",
                self.shape.band_count
            )));
        }
        if self.written_bands[band_index] {
            return Err(MaterializeError::new(format!(
                "band index {band_index} was materialized more than once"
            )));
        }

        let band_len = band_len(self.shape.pixel_count, self.shape.depth)?;
        self.active_band = Some(ActiveBand {
            band_index,
            order: BandWriteOrder::PixelMajor,
            progress: ActiveBandProgress::Prefix(0),
        });
        for value_index in 0..band_len {
            let out_index = band_value_index(self.shape, band_index, self.layout, value_index);
            self.out[out_index].write(value_at(value_index));
            let ActiveBandProgress::Prefix(written_values) =
                &mut self.active_band.as_mut().unwrap().progress
            else {
                unreachable!();
            };
            *written_values += 1;
        }
        self.active_band = None;
        self.written_bands[band_index] = true;
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<T>> {
        let mut this = self;
        if let Some(active_band) = this.active_band.as_ref() {
            validate_active_band_complete(this.shape, active_band)?;
            return Err(MaterializeError::new(format!(
                "band {} was not finalized before finishing the output buffer",
                active_band.band_index
            )));
        }
        if this.written_bands.iter().any(|written| !written) {
            return Err(MaterializeError::new(
                "not all decoded bands were materialized into the output buffer",
            ));
        }

        let out = std::mem::take(&mut this.out);
        this.active_band = None;
        this.written_bands.fill(false);
        Ok(unsafe { assume_init_vec(out) })
    }

    pub fn band_writer(&mut self, band_index: usize) -> Result<MaterializerBandWriter<'_, T>>
    where
        T: Copy + Default,
    {
        if band_index >= self.shape.band_count {
            return Err(MaterializeError::new(format!(
                "band index {band_index} exceeds band count {}",
                self.shape.band_count
            )));
        }
        if self.written_bands[band_index] {
            return Err(MaterializeError::new(format!(
                "band index {band_index} was materialized more than once"
            )));
        }
        self.active_band = Some(ActiveBand {
            band_index,
            order: BandWriteOrder::PixelMajor,
            progress: ActiveBandProgress::Prefix(0),
        });
        Ok(MaterializerBandWriter {
            materializer: self,
            band_index,
            finished: false,
        })
    }
}

impl<T: Clone> BandMaterializer<T> {
    pub fn copy_band(&mut self, band_index: usize, values: &[T]) -> Result<()> {
        if band_index >= self.shape.band_count {
            return Err(MaterializeError::new(format!(
                "band index {band_index} exceeds band count {}",
                self.shape.band_count
            )));
        }
        if self.written_bands[band_index] {
            return Err(MaterializeError::new(format!(
                "band index {band_index} was materialized more than once"
            )));
        }

        let band_len = band_len(self.shape.pixel_count, self.shape.depth)?;
        if values.len() != band_len {
            return Err(MaterializeError::new(
                "decoded band length does not match its metadata",
            ));
        }

        self.active_band = Some(ActiveBand {
            band_index,
            order: BandWriteOrder::PixelMajor,
            progress: ActiveBandProgress::Prefix(0),
        });
        write_band_values_into_uninit_slice(
            &mut self.out,
            values,
            self.shape,
            self.layout,
            self.active_band.as_mut().unwrap(),
        );
        self.active_band = None;
        self.written_bands[band_index] = true;
        Ok(())
    }
}

pub struct MaterializerBandWriter<'a, T> {
    materializer: &'a mut BandMaterializer<T>,
    band_index: usize,
    finished: bool,
}

impl<T: Copy + Default> MaterializerBandWriter<'_, T> {
    pub fn finish(mut self) -> Result<()> {
        let active_band = self.materializer.active_band.as_ref().ok_or_else(|| {
            MaterializeError::new("cannot finalize a band that is no longer active")
        })?;
        validate_active_band_complete(self.materializer.shape, active_band)?;
        self.materializer.written_bands[self.band_index] = true;
        self.materializer.active_band = None;
        self.finished = true;
        Ok(())
    }
}

impl<T: Copy + Default> BandWriter<T> for MaterializerBandWriter<'_, T> {
    fn fill_default(&mut self) {
        let band_len = band_len(
            self.materializer.shape.pixel_count,
            self.materializer.shape.depth,
        )
        .expect("band length was validated when the materializer was created");
        match &mut self.materializer.active_band.as_mut().unwrap().progress {
            ActiveBandProgress::Prefix(written_values) => {
                for value_index in 0..band_len {
                    let out_index = band_value_index(
                        self.materializer.shape,
                        self.band_index,
                        self.materializer.layout,
                        value_index,
                    );
                    if value_index < *written_values {
                        unsafe {
                            *self.materializer.out[out_index].assume_init_mut() = T::default();
                        }
                    } else {
                        self.materializer.out[out_index].write(T::default());
                    }
                }
                *written_values = band_len;
            }
            ActiveBandProgress::Sparse(initialized) => {
                for (value_index, is_initialized) in
                    initialized.iter_mut().enumerate().take(band_len)
                {
                    let out_index = band_value_index(
                        self.materializer.shape,
                        self.band_index,
                        self.materializer.layout,
                        value_index,
                    );
                    if *is_initialized {
                        unsafe {
                            *self.materializer.out[out_index].assume_init_mut() = T::default();
                        }
                    } else {
                        self.materializer.out[out_index].write(T::default());
                        *is_initialized = true;
                    }
                }
            }
        }
    }

    fn write(&mut self, pixel: usize, dim: usize, value: T) {
        let logical_index = pixel * self.materializer.shape.depth + dim;
        loop {
            let out_index = band_value_index_for_pixel(
                self.materializer.shape,
                self.band_index,
                self.materializer.layout,
                pixel,
                dim,
            );
            let active_band = self.materializer.active_band.as_mut().unwrap();
            match &mut active_band.progress {
                ActiveBandProgress::Prefix(written_values) => {
                    let ordered_index = match active_band.order {
                        BandWriteOrder::PixelMajor => logical_index,
                        BandWriteOrder::DimMajor => {
                            dim * self.materializer.shape.pixel_count + pixel
                        }
                        BandWriteOrder::Arbitrary => usize::MAX,
                    };
                    if ordered_index < *written_values {
                        unsafe {
                            *self.materializer.out[out_index].assume_init_mut() = value;
                        }
                        return;
                    }
                    if ordered_index == *written_values {
                        self.materializer.out[out_index].write(value);
                        *written_values += 1;
                        return;
                    }

                    let mut initialized =
                        vec![
                            false;
                            band_len(
                                self.materializer.shape.pixel_count,
                                self.materializer.shape.depth,
                            )
                            .expect("band length was validated when the materializer was created")
                        ];
                    for step_index in 0..*written_values {
                        initialized[logical_index_for_step(
                            self.materializer.shape,
                            active_band.order,
                            step_index,
                        )] = true;
                    }
                    active_band.progress = ActiveBandProgress::Sparse(initialized);
                }
                ActiveBandProgress::Sparse(initialized) => {
                    if initialized[logical_index] {
                        unsafe {
                            *self.materializer.out[out_index].assume_init_mut() = value;
                        }
                    } else {
                        self.materializer.out[out_index].write(value);
                        initialized[logical_index] = true;
                    }
                    return;
                }
            }
        }
    }

    fn read(&self, pixel: usize, dim: usize) -> T {
        let out_index = band_value_index_for_pixel(
            self.materializer.shape,
            self.band_index,
            self.materializer.layout,
            pixel,
            dim,
        );
        let logical_index = pixel * self.materializer.shape.depth + dim;
        let active_band = self.materializer.active_band.as_ref().unwrap();
        match &active_band.progress {
            ActiveBandProgress::Prefix(written_values) => {
                let ordered_index = match active_band.order {
                    BandWriteOrder::PixelMajor => logical_index,
                    BandWriteOrder::DimMajor => dim * self.materializer.shape.pixel_count + pixel,
                    BandWriteOrder::Arbitrary => usize::MAX,
                };
                assert!(
                    ordered_index < *written_values,
                    "attempted to read an uninitialized materialized sample"
                );
            }
            ActiveBandProgress::Sparse(initialized) => {
                assert!(
                    initialized[logical_index],
                    "attempted to read an uninitialized materialized sample"
                );
            }
        }
        unsafe { *self.materializer.out[out_index].assume_init_ref() }
    }

    fn depth(&self) -> usize {
        self.materializer.shape.depth
    }

    fn set_write_order(&mut self, order: BandWriteOrder) {
        let active_band = self.materializer.active_band.as_mut().unwrap();
        match &active_band.progress {
            ActiveBandProgress::Prefix(0) => active_band.order = order,
            _ => debug_assert_eq!(active_band.order, order),
        }
    }
}

impl<T> Drop for BandMaterializer<T> {
    fn drop(&mut self) {
        if self.out.is_empty() {
            return;
        }

        for (band_index, written) in self.written_bands.iter().copied().enumerate() {
            if !written {
                continue;
            }
            drop_band_prefix(
                &mut self.out,
                self.shape,
                band_index,
                self.layout,
                BandWriteOrder::PixelMajor,
                band_len(self.shape.pixel_count, self.shape.depth).unwrap_or(0),
            );
        }

        if let Some(active_band) = self.active_band.take() {
            match active_band.progress {
                ActiveBandProgress::Prefix(written_values) => drop_band_prefix(
                    &mut self.out,
                    self.shape,
                    active_band.band_index,
                    self.layout,
                    active_band.order,
                    written_values,
                ),
                ActiveBandProgress::Sparse(initialized) => drop_sparse_band(
                    &mut self.out,
                    self.shape,
                    active_band.band_index,
                    self.layout,
                    &initialized,
                ),
            }
        }
    }
}

fn write_band_values_into_uninit_slice<T: Clone>(
    out: &mut [MaybeUninit<T>],
    values: &[T],
    shape: BandShape,
    layout: BandLayout,
    active_band: &mut ActiveBand,
) {
    match layout {
        BandLayout::Interleaved => {
            if shape.depth <= 1 {
                for pixel in 0..shape.pixel_count {
                    out[pixel * shape.band_count + active_band.band_index]
                        .write(values[pixel].clone());
                    let ActiveBandProgress::Prefix(written_values) = &mut active_band.progress
                    else {
                        unreachable!();
                    };
                    *written_values += 1;
                }
            } else {
                for pixel in 0..shape.pixel_count {
                    let src_base = pixel * shape.depth;
                    let dst_base =
                        (pixel * shape.band_count + active_band.band_index) * shape.depth;
                    for offset in 0..shape.depth {
                        out[dst_base + offset].write(values[src_base + offset].clone());
                        let ActiveBandProgress::Prefix(written_values) = &mut active_band.progress
                        else {
                            unreachable!();
                        };
                        *written_values += 1;
                    }
                }
            }
        }
        BandLayout::Bsq => {
            let band_len = values.len();
            let dst_base = active_band.band_index * band_len;
            for (index, value) in values.iter().enumerate() {
                out[dst_base + index].write(value.clone());
                let ActiveBandProgress::Prefix(written_values) = &mut active_band.progress else {
                    unreachable!();
                };
                *written_values += 1;
            }
        }
    }
}

fn drop_band_prefix<T>(
    out: &mut [MaybeUninit<T>],
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
    order: BandWriteOrder,
    written_values: usize,
) {
    for step_index in 0..written_values {
        let out_index = band_value_index_for_step(shape, band_index, layout, order, step_index);
        unsafe {
            out[out_index].assume_init_drop();
        }
    }
}

fn drop_sparse_band<T>(
    out: &mut [MaybeUninit<T>],
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
    initialized: &[bool],
) {
    for (value_index, is_initialized) in initialized.iter().copied().enumerate() {
        if !is_initialized {
            continue;
        }
        let out_index = band_value_index(shape, band_index, layout, value_index);
        unsafe {
            out[out_index].assume_init_drop();
        }
    }
}

fn band_value_index(
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
    value_index: usize,
) -> usize {
    match layout {
        BandLayout::Interleaved => {
            if shape.depth <= 1 {
                value_index * shape.band_count + band_index
            } else {
                let pixel = value_index / shape.depth;
                let sample = value_index % shape.depth;
                (pixel * shape.band_count + band_index) * shape.depth + sample
            }
        }
        BandLayout::Bsq => band_index * shape.pixel_count * shape.depth.max(1) + value_index,
    }
}

fn logical_index_for_step(shape: BandShape, order: BandWriteOrder, step_index: usize) -> usize {
    match order {
        BandWriteOrder::PixelMajor => step_index,
        BandWriteOrder::DimMajor => {
            let pixel = step_index % shape.pixel_count;
            let dim = step_index / shape.pixel_count;
            pixel * shape.depth + dim
        }
        BandWriteOrder::Arbitrary => {
            unreachable!("arbitrary-order writes cannot be represented as a prefix")
        }
    }
}

fn band_value_index_for_step(
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
    order: BandWriteOrder,
    step_index: usize,
) -> usize {
    let logical_index = logical_index_for_step(shape, order, step_index);
    band_value_index(shape, band_index, layout, logical_index)
}

fn band_value_index_for_pixel(
    shape: BandShape,
    band_index: usize,
    layout: BandLayout,
    pixel: usize,
    dim: usize,
) -> usize {
    match layout {
        BandLayout::Interleaved => ((pixel * shape.band_count + band_index) * shape.depth) + dim,
        BandLayout::Bsq => (band_index * shape.pixel_count + pixel) * shape.depth + dim,
    }
}

fn band_len(pixel_count: usize, depth: usize) -> Result<usize> {
    pixel_count
        .checked_mul(depth.max(1))
        .ok_or_else(|| MaterializeError::new("decoded band length overflows usize"))
}

fn total_len(pixel_count: usize, depth: usize, band_count: usize) -> Result<usize> {
    band_len(pixel_count, depth)?
        .checked_mul(band_count)
        .ok_or_else(|| MaterializeError::new("decoded band set length overflows usize"))
}

fn validate_active_band_complete(shape: BandShape, active_band: &ActiveBand) -> Result<()> {
    let band_len = band_len(shape.pixel_count, shape.depth)?;
    let is_complete = match &active_band.progress {
        ActiveBandProgress::Prefix(written_values) => *written_values == band_len,
        ActiveBandProgress::Sparse(initialized) => {
            initialized.len() == band_len && initialized.iter().all(|initialized| *initialized)
        }
    };
    if is_complete {
        return Ok(());
    }
    Err(MaterializeError::new(format!(
        "band {} was finalized before all decoded values were initialized",
        active_band.band_index
    )))
}

unsafe fn assume_init_vec<T>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
    let len = values.len();
    let cap = values.capacity();
    let ptr = values.as_ptr() as *mut T;
    std::mem::forget(values);
    Vec::from_raw_parts(ptr, len, cap)
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{BandLayout, BandMaterializer, BandWriteOrder, BandWriter};

    #[derive(Debug)]
    struct CloneBomb {
        state: Arc<State>,
    }

    #[derive(Debug)]
    struct State {
        live: AtomicUsize,
        clones: AtomicUsize,
        panic_at: usize,
    }

    impl CloneBomb {
        fn new(state: &Arc<State>) -> Self {
            state.live.fetch_add(1, Ordering::SeqCst);
            Self {
                state: Arc::clone(state),
            }
        }
    }

    impl Clone for CloneBomb {
        fn clone(&self) -> Self {
            let clone_number = self.state.clones.fetch_add(1, Ordering::SeqCst) + 1;
            if clone_number == self.state.panic_at {
                panic!("clone panic for test");
            }
            self.state.live.fetch_add(1, Ordering::SeqCst);
            Self {
                state: Arc::clone(&self.state),
            }
        }
    }

    impl Drop for CloneBomb {
        fn drop(&mut self) {
            self.state.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn drops_fully_and_partially_written_interleaved_bands_on_panic() {
        let state = Arc::new(State {
            live: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at: 10,
        });
        let first_band: Vec<_> = (0..6).map(|_| CloneBomb::new(&state)).collect();
        let second_band: Vec<_> = (0..6).map(|_| CloneBomb::new(&state)).collect();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut materializer = BandMaterializer::new(3, 2, 2, BandLayout::Interleaved).unwrap();
            materializer.copy_band(0, &first_band).unwrap();
            materializer.copy_band(1, &second_band).unwrap();
        }));

        assert!(result.is_err());
        assert_eq!(state.live.load(Ordering::SeqCst), 12);

        drop(first_band);
        drop(second_band);
        assert_eq!(state.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn drops_partially_written_bsq_band_on_panic() {
        let state = Arc::new(State {
            live: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at: 3,
        });
        let values: Vec<_> = (0..4).map(|_| CloneBomb::new(&state)).collect();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut materializer = BandMaterializer::new(4, 1, 1, BandLayout::Bsq).unwrap();
            materializer.copy_band(0, &values).unwrap();
        }));

        assert!(result.is_err());
        assert_eq!(state.live.load(Ordering::SeqCst), 4);

        drop(values);
        assert_eq!(state.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn rejects_finalizing_incomplete_band_writer() {
        let mut materializer =
            BandMaterializer::<u16>::new(2, 2, 1, BandLayout::Interleaved).unwrap();
        let mut writer = materializer.band_writer(0).unwrap();
        writer.set_write_order(BandWriteOrder::DimMajor);
        writer.write(0, 0, 1);
        writer.write(1, 0, 2);
        let err = writer.finish().unwrap_err();
        assert_eq!(
            err.to_string(),
            "band 0 was finalized before all decoded values were initialized"
        );
    }
}
