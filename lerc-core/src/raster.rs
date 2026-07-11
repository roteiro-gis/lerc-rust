use crate::error::{Error, Result};
use crate::types::{BandLayout, DataType, PixelData};

/// Borrowed, dimension-checked view of one pixel-interleaved raster band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterView<'a, T: Sample> {
    width: u32,
    height: u32,
    depth: u32,
    data: &'a [T],
}

impl<'a, T: Sample> RasterView<'a, T> {
    /// Creates a raster view.
    ///
    /// # Errors
    /// Returns an error when depth is zero, dimensions overflow, or `data` has
    /// a different length than `width * height * depth`.
    pub fn new(width: u32, height: u32, depth: u32, data: &'a [T]) -> Result<Self> {
        let expected_len = sample_count_from_dims(width, height, depth)?;
        if data.len() != expected_len {
            return Err(Error::InvalidArgument(
                "raster slice length does not match its dimensions",
            ));
        }
        Ok(Self {
            width,
            height,
            depth,
            data,
        })
    }

    /// Returns the raster width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Returns the raster height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }

    /// Returns the number of samples per pixel.
    pub fn depth(self) -> u32 {
        self.depth
    }

    /// Returns the underlying pixel-interleaved sample slice.
    pub fn data(self) -> &'a [T] {
        self.data
    }

    /// Returns the encoded sample type.
    pub fn data_type(self) -> DataType {
        T::DATA_TYPE
    }

    /// Returns `width * height` using checked arithmetic.
    pub fn pixel_count(self) -> Result<usize> {
        pixel_count_from_dims(self.width, self.height)
    }

    /// Returns the total sample count using checked arithmetic.
    pub fn sample_count(self) -> Result<usize> {
        sample_count_from_dims(self.width, self.height, self.depth)
    }

    /// Returns one sample by flat pixel index and depth index.
    ///
    /// # Panics
    /// Panics if `pixel` or `dim` is outside this view.
    pub fn sample(self, pixel: usize, dim: usize) -> T {
        self.data[sample_index(pixel, self.depth as usize, dim)]
    }
}

/// Borrowed, dimension-checked view of a multi-band raster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandSetView<'a, T: Sample> {
    width: u32,
    height: u32,
    depth: u32,
    band_count: usize,
    layout: BandLayout,
    data: &'a [T],
    pixel_count: usize,
    pixel_stride: usize,
    band_stride: usize,
}

impl<'a, T: Sample> BandSetView<'a, T> {
    /// Creates a band-set view over interleaved or band-sequential samples.
    ///
    /// # Errors
    /// Returns an error for zero bands or depth, overflowing dimensions, or a
    /// sample slice whose length does not match the declared shape.
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        band_count: usize,
        layout: BandLayout,
        data: &'a [T],
    ) -> Result<Self> {
        if band_count == 0 {
            return Err(Error::InvalidArgument(
                "band_count must be greater than zero",
            ));
        }

        let band_sample_count = sample_count_from_dims(width, height, depth)?;
        let pixel_count = pixel_count_from_dims(width, height)?;
        let pixel_stride = (depth as usize)
            .checked_mul(band_count)
            .ok_or(Error::SizeOverflow("interleaved pixel stride"))?;
        let expected_len = band_sample_count
            .checked_mul(band_count)
            .ok_or(Error::SizeOverflow("band set value count"))?;
        if data.len() != expected_len {
            return Err(Error::InvalidArgument(
                "band set slice length does not match its dimensions",
            ));
        }

        Ok(Self {
            width,
            height,
            depth,
            band_count,
            layout,
            data,
            pixel_count,
            pixel_stride,
            band_stride: band_sample_count,
        })
    }

    /// Returns the raster width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Returns the raster height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }

    /// Returns the number of samples per pixel in each band.
    pub fn depth(self) -> u32 {
        self.depth
    }

    /// Returns the number of bands.
    pub fn band_count(self) -> usize {
        self.band_count
    }

    /// Returns the memory layout of the supplied samples.
    pub fn layout(self) -> BandLayout {
        self.layout
    }

    /// Returns the underlying sample slice.
    pub fn data(self) -> &'a [T] {
        self.data
    }

    /// Returns the encoded sample type.
    pub fn data_type(self) -> DataType {
        T::DATA_TYPE
    }

    /// Returns the checked pixel count cached at construction.
    pub fn pixel_count(self) -> Result<usize> {
        Ok(self.pixel_count)
    }

    /// Returns the sample count of one band.
    pub fn band_sample_count(self) -> Result<usize> {
        Ok(self.band_stride)
    }

    /// Returns the total number of samples across every band.
    pub fn value_count(self) -> Result<usize> {
        self.band_sample_count()?
            .checked_mul(self.band_count)
            .ok_or(Error::SizeOverflow("band set value count"))
    }

    /// Returns one sample by band, flat pixel, and depth index.
    ///
    /// # Panics
    /// Panics if any supplied index is outside this view.
    pub fn sample(self, band: usize, pixel: usize, dim: usize) -> T {
        let depth = self.depth as usize;
        let index = match self.layout {
            BandLayout::Interleaved => pixel * self.pixel_stride + band * depth + dim,
            BandLayout::Bsq => band * self.band_stride + pixel * depth + dim,
        };
        self.data[index]
    }
}

/// Borrowed, dimension-checked validity mask; zero is invalid and nonzero is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskView<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> MaskView<'a> {
    /// Creates a validity-mask view.
    ///
    /// # Errors
    /// Returns an error when dimensions overflow or the mask length differs
    /// from `width * height`.
    pub fn new(width: u32, height: u32, data: &'a [u8]) -> Result<Self> {
        let expected_len = pixel_count_from_dims(width, height)?;
        if data.len() != expected_len {
            return Err(Error::InvalidArgument(
                "mask slice length does not match its dimensions",
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns the mask width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Returns the mask height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }

    /// Returns the underlying byte mask.
    pub fn data(self) -> &'a [u8] {
        self.data
    }

    /// Returns the checked number of mask pixels.
    pub fn pixel_count(self) -> Result<usize> {
        pixel_count_from_dims(self.width, self.height)
    }

    /// Counts nonzero, valid mask entries.
    pub fn valid_count(self) -> usize {
        self.data.iter().filter(|&&value| value != 0).count()
    }
}

/// Primitive sample types supported by LERC.
///
/// This trait is sealed and implemented for `i8`, `u8`, `i16`, `u16`, `i32`,
/// `u32`, `f32`, and `f64`.
pub trait Sample: Copy + Default + private::Sealed + 'static {
    /// Corresponding runtime [`DataType`].
    const DATA_TYPE: DataType;
    /// Whether this sample is an integer type.
    const IS_INTEGER: bool;

    /// Converts an `f64` with Rust's primitive numeric-cast semantics.
    fn from_f64(value: f64) -> Self;
    /// Promotes the sample to `f64`.
    fn to_f64(self) -> f64;
    /// Reads one little-endian sample.
    fn read_le(bytes: &[u8]) -> Result<Self>;
    /// Appends one little-endian sample.
    fn write_le(self, out: &mut Vec<u8>);
    /// Wraps a homogeneous buffer in [`PixelData`].
    fn into_pixel_data(values: Vec<Self>) -> PixelData;

    /// Reads an aligned little-endian sample buffer.
    fn read_vec(bytes: &[u8]) -> Result<Vec<Self>> {
        let size = Self::DATA_TYPE.byte_len();
        let chunks = bytes.chunks_exact(size);
        if !chunks.remainder().is_empty() {
            return Err(Error::invalid_blob(
                "typed value payload length is not aligned to its data type",
            ));
        }
        chunks.map(Self::read_le).collect()
    }
}

macro_rules! impl_sample {
    ($ty:ty, $variant:ident, $is_integer:expr) => {
        impl Sample for $ty {
            const DATA_TYPE: DataType = DataType::$variant;
            const IS_INTEGER: bool = $is_integer;

            fn from_f64(value: f64) -> Self {
                value as $ty
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn read_le(bytes: &[u8]) -> Result<Self> {
                let bytes = bytes.try_into().map_err(|_| {
                    Error::invalid_blob("typed scalar byte length does not match its data type")
                })?;
                Ok(<$ty>::from_le_bytes(bytes))
            }

            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn into_pixel_data(values: Vec<Self>) -> PixelData {
                PixelData::$variant(values)
            }
        }
    };
}

impl_sample!(i8, I8, true);
impl_sample!(u8, U8, true);
impl_sample!(i16, I16, true);
impl_sample!(u16, U16, true);
impl_sample!(i32, I32, true);
impl_sample!(u32, U32, true);
impl_sample!(f32, F32, false);
impl_sample!(f64, F64, false);

#[macro_export]
/// Dispatches a runtime [`DataType`](crate::DataType) to a concrete primitive type alias.
macro_rules! dispatch_data_type {
    ($data_type:expr, $sample:ident => $body:block) => {{
        match $data_type {
            $crate::DataType::I8 => {
                type $sample = i8;
                $body
            }
            $crate::DataType::U8 => {
                type $sample = u8;
                $body
            }
            $crate::DataType::I16 => {
                type $sample = i16;
                $body
            }
            $crate::DataType::U16 => {
                type $sample = u16;
                $body
            }
            $crate::DataType::I32 => {
                type $sample = i32;
                $body
            }
            $crate::DataType::U32 => {
                type $sample = u32;
                $body
            }
            $crate::DataType::F32 => {
                type $sample = f32;
                $body
            }
            $crate::DataType::F64 => {
                type $sample = f64;
                $body
            }
        }
    }};
}

/// Computes `width * height` using checked, platform-portable arithmetic.
pub fn pixel_count_from_dims(width: u32, height: u32) -> Result<usize> {
    let width = usize::try_from(width).map_err(|_| Error::SizeOverflow("width as usize"))?;
    let height = usize::try_from(height).map_err(|_| Error::SizeOverflow("height as usize"))?;
    width
        .checked_mul(height)
        .ok_or(Error::SizeOverflow("pixel count"))
}

/// Computes `width * height * depth` using checked arithmetic.
pub fn sample_count_from_dims(width: u32, height: u32, depth: u32) -> Result<usize> {
    if depth == 0 {
        return Err(Error::InvalidArgument("depth must be greater than zero"));
    }
    pixel_count_from_dims(width, height)?
        .checked_mul(depth as usize)
        .ok_or(Error::SizeOverflow("sample count"))
}

/// Decodes little-endian samples and converts them directly to `T`.
pub fn read_values_as<T: Sample>(bytes: &[u8], source_type: DataType) -> Result<Vec<T>> {
    if source_type == T::DATA_TYPE {
        return T::read_vec(bytes);
    }

    let sample_size = source_type.byte_len();
    if bytes.len() % sample_size != 0 {
        return Err(Error::invalid_blob(
            "typed value payload length is not aligned to its data type",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / sample_size);
    crate::dispatch_data_type!(source_type, Source => {
        for chunk in bytes.chunks_exact(sample_size) {
            out.push(T::from_f64(Source::read_le(chunk)?.to_f64()));
        }
    });
    Ok(out)
}

/// Rounds or truncates an `f64` through the specified primitive representation.
pub fn coerce_f64_to_data_type(value: f64, data_type: DataType) -> f64 {
    crate::dispatch_data_type!(data_type, Target => { Target::from_f64(value).to_f64() })
}

/// Converts a decoded value to an output sample while preserving source-type rounding.
pub fn output_value<T: Sample>(value: f64, source_type: DataType) -> T {
    if T::DATA_TYPE == DataType::F64 && source_type != DataType::F64 {
        T::from_f64(coerce_f64_to_data_type(value, source_type))
    } else {
        T::from_f64(value)
    }
}

/// Reads one little-endian scalar and promotes it to `f64`.
pub fn read_scalar(bytes: &[u8], data_type: DataType) -> Result<f64> {
    crate::dispatch_data_type!(data_type, Source => { Ok(Source::read_le(bytes)?.to_f64()) })
}

/// Reads an aligned little-endian sample buffer as promoted `f64` values.
pub fn read_typed_values(bytes: &[u8], data_type: DataType) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(bytes.len() / data_type.byte_len());
    for chunk in bytes.chunks_exact(data_type.byte_len()) {
        out.push(read_scalar(chunk, data_type)?);
    }
    if bytes.len() % data_type.byte_len() != 0 {
        return Err(Error::invalid_blob(
            "typed value payload length is not aligned to its data type",
        ));
    }
    Ok(out)
}

/// Converts and appends one scalar in the requested little-endian representation.
pub fn append_value_as(out: &mut Vec<u8>, value: f64, data_type: DataType) {
    crate::dispatch_data_type!(data_type, Target => {
        Target::from_f64(value).write_le(out);
    });
}

/// Counts valid mask entries in a rectangular row-major block.
///
/// # Panics
/// Panics if the rectangle lies outside `mask` for the supplied row width.
pub fn count_valid_in_block(
    mask: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
    block_width: usize,
    block_height: usize,
) -> usize {
    let mut count = 0usize;
    for row in 0..block_height {
        let row_offset = (y0 + row) * width + x0;
        for col in 0..block_width {
            count += usize::from(mask[row_offset + col] != 0);
        }
    }
    count
}

/// Returns the minimum bit width needed to represent `max_index`.
pub fn bits_required(max_index: usize) -> u8 {
    let mut bits = 0u8;
    let mut value = max_index;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

/// Pads a byte slice to 32-bit alignment and returns little-endian words.
pub fn words_from_padded(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut padded = vec![0u8; bytes.len().div_ceil(4) * 4];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Computes the Fletcher-32 checksum variant used by Lerc2.
pub fn fletcher32(bytes: &[u8]) -> u32 {
    let mut sum1 = 0xffffu32;
    let mut sum2 = 0xffffu32;
    let mut words = bytes.len() / 2;
    let mut index = 0usize;

    while words > 0 {
        let chunk = words.min(359);
        words -= chunk;
        for _ in 0..chunk {
            sum1 += (bytes[index] as u32) << 8;
            index += 1;
            sum2 += sum1 + bytes[index] as u32;
            sum1 += bytes[index] as u32;
            index += 1;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if bytes.len() & 1 != 0 {
        sum1 += (bytes[index] as u32) << 8;
        sum2 += sum1;
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum2 << 16) | (sum1 & 0xffff)
}

/// Computes the pixel-interleaved sample index `pixel * depth + dim`.
pub fn sample_index(pixel: usize, depth: usize, dim: usize) -> usize {
    pixel * depth + dim
}

mod private {
    pub trait Sealed {}

    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
