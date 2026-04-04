use lerc_core::BandLayout;

#[derive(Debug)]
pub(crate) struct BandSink<'a, T> {
    out: &'a mut [T],
    pixel_count: usize,
    depth: usize,
    band_index: usize,
    band_count: usize,
    layout: BandLayout,
}

impl<'a, T: Copy + Default> BandSink<'a, T> {
    pub(crate) fn new(
        out: &'a mut [T],
        pixel_count: usize,
        depth: usize,
        band_index: usize,
        band_count: usize,
        layout: BandLayout,
    ) -> Self {
        Self {
            out,
            pixel_count,
            depth: depth.max(1),
            band_index,
            band_count,
            layout,
        }
    }

    pub(crate) fn fill_default(&mut self) {
        match self.layout {
            BandLayout::Interleaved => {
                for pixel in 0..self.pixel_count {
                    let base = (pixel * self.band_count + self.band_index) * self.depth;
                    self.out[base..base + self.depth].fill(T::default());
                }
            }
            BandLayout::Bsq => {
                let band_len = self.pixel_count * self.depth;
                let base = self.band_index * band_len;
                self.out[base..base + band_len].fill(T::default());
            }
        }
    }

    pub(crate) fn write(&mut self, pixel: usize, dim: usize, value: T) {
        let index = self.index(pixel, dim);
        self.out[index] = value;
    }

    pub(crate) fn read(&self, pixel: usize, dim: usize) -> T {
        self.out[self.index(pixel, dim)]
    }

    fn index(&self, pixel: usize, dim: usize) -> usize {
        match self.layout {
            BandLayout::Interleaved => {
                ((pixel * self.band_count + self.band_index) * self.depth) + dim
            }
            BandLayout::Bsq => (self.band_index * self.pixel_count + pixel) * self.depth + dim,
        }
    }
}
