use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl DataType {
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Self::I8),
            1 => Ok(Self::U8),
            2 => Ok(Self::I16),
            3 => Ok(Self::U16),
            4 => Ok(Self::I32),
            5 => Ok(Self::U32),
            6 => Ok(Self::F32),
            7 => Ok(Self::F64),
            _ => Err(Error::InvalidBlob(format!(
                "unsupported data type code {code}"
            ))),
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::I8 => 0,
            Self::U8 => 1,
            Self::I16 => 2,
            Self::U16 => 3,
            Self::I32 => 4,
            Self::U32 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
        }
    }

    pub fn byte_len(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PixelData {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl PixelData {
    pub fn data_type(&self) -> DataType {
        match self {
            Self::I8(_) => DataType::I8,
            Self::U8(_) => DataType::U8,
            Self::I16(_) => DataType::I16,
            Self::U16(_) => DataType::U16,
            Self::I32(_) => DataType::I32,
            Self::U32(_) => DataType::U32,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Self::I8(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U8(v) => v.iter().map(|&x| x as f64).collect(),
            Self::I16(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U16(v) => v.iter().map(|&x| x as f64).collect(),
            Self::I32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::F64(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    Lerc2(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlobInfo {
    pub version: Version,
    pub data_type: DataType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub min_values: Option<Vec<f64>>,
    pub max_values: Option<Vec<f64>>,
    pub valid_pixel_count: u32,
    pub micro_block_size: u32,
    pub blob_size: usize,
    pub max_z_error: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl BlobInfo {
    pub fn pixel_count(&self) -> Result<usize> {
        usize::try_from(self.width)
            .ok()
            .and_then(|w| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))
    }

    pub fn sample_count(&self) -> Result<usize> {
        self.pixel_count()?
            .checked_mul(self.depth as usize)
            .ok_or_else(|| Error::InvalidBlob("sample count overflows usize".into()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub info: BlobInfo,
    pub pixels: PixelData,
    pub mask: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedF64 {
    pub info: BlobInfo,
    pub pixels: Vec<f64>,
    pub mask: Option<Vec<u8>>,
}
