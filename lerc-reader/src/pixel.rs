use lerc_core::{DataType, Error, PixelData, Result};

pub(crate) trait Sample: Copy + Default {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn read_vec(bytes: &[u8]) -> Result<Vec<Self>>;
    fn into_pixel_data(values: Vec<Self>) -> PixelData;
}

macro_rules! impl_sample {
    ($ty:ty, $size:expr, $variant:ident) => {
        impl Sample for $ty {
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
        }
    };
}

impl Sample for u8 {
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
}

impl Sample for i8 {
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
}

impl_sample!(i16, 2, I16);
impl_sample!(u16, 2, U16);
impl_sample!(i32, 4, I32);
impl_sample!(u32, 4, U32);
impl_sample!(f32, 4, F32);
impl_sample!(f64, 8, F64);

pub(crate) fn read_scalar(bytes: &[u8], data_type: DataType) -> Result<f64> {
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

pub(crate) fn read_typed_values(bytes: &[u8], data_type: DataType) -> Result<Vec<f64>> {
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

pub(crate) fn count_valid_in_block(
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

pub(crate) fn bits_required(max_index: usize) -> u8 {
    let mut bits = 0u8;
    let mut value = max_index;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

pub(crate) fn words_from_padded(bytes: &[u8]) -> Vec<u32> {
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

pub(crate) fn fletcher32(bytes: &[u8]) -> u32 {
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

pub(crate) fn sample_index(pixel: usize, depth: usize, dim: usize) -> usize {
    pixel * depth + dim
}
