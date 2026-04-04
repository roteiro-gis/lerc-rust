use crate::error::{Error, Result};
use crate::types::{DataType, PixelData};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterView<'a, T: Sample> {
    width: u32,
    height: u32,
    depth: u32,
    data: &'a [T],
}

impl<'a, T: Sample> RasterView<'a, T> {
    pub fn new(width: u32, height: u32, depth: u32, data: &'a [T]) -> Result<Self> {
        let expected_len = sample_count_from_dims(width, height, depth)?;
        if data.len() != expected_len {
            return Err(Error::InvalidArgument(format!(
                "raster slice length {} does not match width={width}, height={height}, depth={depth}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            depth,
            data,
        })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn depth(self) -> u32 {
        self.depth
    }

    pub fn data(self) -> &'a [T] {
        self.data
    }

    pub fn data_type(self) -> DataType {
        T::data_type()
    }

    pub fn pixel_count(self) -> Result<usize> {
        pixel_count_from_dims(self.width, self.height)
    }

    pub fn sample_count(self) -> Result<usize> {
        sample_count_from_dims(self.width, self.height, self.depth)
    }

    pub fn sample(self, pixel: usize, dim: usize) -> T {
        self.data[sample_index(pixel, self.depth as usize, dim)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskView<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> MaskView<'a> {
    pub fn new(width: u32, height: u32, data: &'a [u8]) -> Result<Self> {
        let expected_len = pixel_count_from_dims(width, height)?;
        if data.len() != expected_len {
            return Err(Error::InvalidArgument(format!(
                "mask slice length {} does not match width={width}, height={height}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn data(self) -> &'a [u8] {
        self.data
    }

    pub fn pixel_count(self) -> Result<usize> {
        pixel_count_from_dims(self.width, self.height)
    }

    pub fn valid_count(self) -> usize {
        self.data.iter().filter(|&&value| value != 0).count()
    }
}

pub trait Sample: Copy + Default + private::Sealed + 'static {
    fn data_type() -> DataType;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn read_vec(bytes: &[u8]) -> Result<Vec<Self>>;
    fn into_pixel_data(values: Vec<Self>) -> PixelData;
    fn append_le_bytes(self, out: &mut Vec<u8>);
}

macro_rules! impl_sample {
    ($ty:ty, $size:expr, $variant:ident) => {
        impl Sample for $ty {
            fn data_type() -> DataType {
                DataType::$variant
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn read_vec(bytes: &[u8]) -> Result<Vec<Self>> {
                let chunks = bytes.chunks_exact($size);
                if !chunks.remainder().is_empty() {
                    return Err(Error::InvalidBlob(
                        "typed value payload length is not aligned to its data type".into(),
                    ));
                }
                Ok(chunks
                    .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().unwrap()))
                    .collect())
            }

            fn into_pixel_data(values: Vec<Self>) -> PixelData {
                PixelData::$variant(values)
            }

            fn append_le_bytes(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl Sample for u8 {
    fn data_type() -> DataType {
        DataType::U8
    }

    fn from_f64(value: f64) -> Self {
        value as u8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn read_vec(bytes: &[u8]) -> Result<Vec<Self>> {
        Ok(bytes.to_vec())
    }

    fn into_pixel_data(values: Vec<Self>) -> PixelData {
        PixelData::U8(values)
    }

    fn append_le_bytes(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

impl Sample for i8 {
    fn data_type() -> DataType {
        DataType::I8
    }

    fn from_f64(value: f64) -> Self {
        value as i8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn read_vec(bytes: &[u8]) -> Result<Vec<Self>> {
        Ok(bytes
            .iter()
            .map(|&byte| i8::from_le_bytes([byte]))
            .collect())
    }

    fn into_pixel_data(values: Vec<Self>) -> PixelData {
        PixelData::I8(values)
    }

    fn append_le_bytes(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
}

impl_sample!(i16, 2, I16);
impl_sample!(u16, 2, U16);
impl_sample!(i32, 4, I32);
impl_sample!(u32, 4, U32);
impl_sample!(f32, 4, F32);
impl_sample!(f64, 8, F64);

pub fn pixel_count_from_dims(width: u32, height: u32) -> Result<usize> {
    let width = usize::try_from(width)
        .map_err(|_| Error::InvalidArgument("width does not fit in usize".into()))?;
    let height = usize::try_from(height)
        .map_err(|_| Error::InvalidArgument("height does not fit in usize".into()))?;
    width
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidArgument("pixel count overflows usize".into()))
}

pub fn sample_count_from_dims(width: u32, height: u32, depth: u32) -> Result<usize> {
    if depth == 0 {
        return Err(Error::InvalidArgument(
            "depth must be greater than zero".into(),
        ));
    }
    pixel_count_from_dims(width, height)?
        .checked_mul(depth as usize)
        .ok_or_else(|| Error::InvalidArgument("sample count overflows usize".into()))
}

pub fn read_values_as<T: Sample>(bytes: &[u8], source_type: DataType) -> Result<Vec<T>> {
    if source_type == T::data_type() {
        return T::read_vec(bytes);
    }

    read_typed_values(bytes, source_type)
        .map(|values| values.into_iter().map(T::from_f64).collect())
}

pub fn coerce_f64_to_data_type(value: f64, data_type: DataType) -> f64 {
    match data_type {
        DataType::I8 => (value as i8) as f64,
        DataType::U8 => (value as u8) as f64,
        DataType::I16 => (value as i16) as f64,
        DataType::U16 => (value as u16) as f64,
        DataType::I32 => (value as i32) as f64,
        DataType::U32 => (value as u32) as f64,
        DataType::F32 => (value as f32) as f64,
        DataType::F64 => value,
    }
}

pub fn output_value<T: Sample>(value: f64, source_type: DataType) -> T {
    if T::data_type() == DataType::F64 && source_type != DataType::F64 {
        T::from_f64(coerce_f64_to_data_type(value, source_type))
    } else {
        T::from_f64(value)
    }
}

pub fn read_scalar(bytes: &[u8], data_type: DataType) -> Result<f64> {
    Ok(match data_type {
        DataType::I8 => i8::from_le_bytes([bytes[0]]) as f64,
        DataType::U8 => bytes[0] as f64,
        DataType::I16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::U16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DataType::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
    })
}

pub fn read_typed_values(bytes: &[u8], data_type: DataType) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(bytes.len() / data_type.byte_len());
    for chunk in bytes.chunks_exact(data_type.byte_len()) {
        out.push(read_scalar(chunk, data_type)?);
    }
    if bytes.len() % data_type.byte_len() != 0 {
        return Err(Error::InvalidBlob(
            "typed value payload length is not aligned to its data type".into(),
        ));
    }
    Ok(out)
}

pub fn append_value_as(out: &mut Vec<u8>, value: f64, data_type: DataType) {
    match data_type {
        DataType::I8 => out.push((value as i8) as u8),
        DataType::U8 => out.push(value as u8),
        DataType::I16 => out.extend_from_slice(&(value as i16).to_le_bytes()),
        DataType::U16 => out.extend_from_slice(&(value as u16).to_le_bytes()),
        DataType::I32 => out.extend_from_slice(&(value as i32).to_le_bytes()),
        DataType::U32 => out.extend_from_slice(&(value as u32).to_le_bytes()),
        DataType::F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
        DataType::F64 => out.extend_from_slice(&value.to_le_bytes()),
    }
}

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

pub fn bits_required(max_index: usize) -> u8 {
    let mut bits = 0u8;
    let mut value = max_index;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

pub fn words_from_padded(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut padded = vec![0u8; bytes.len().div_ceil(4) * 4];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

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
