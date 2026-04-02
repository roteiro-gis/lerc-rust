use crate::error::{Error, Result};
use ndarray::{ArrayD, IxDyn};

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

    pub fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
            Self::F64 => "f64",
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

    pub fn into_ndarray<T: NdArrayElement>(self, shape: &[usize]) -> Result<ArrayD<T>> {
        ArrayD::from_shape_vec(IxDyn(shape), T::from_pixel_data(self)?).map_err(|e| {
            Error::InvalidBlob(format!("failed to build ndarray from decoded pixels: {e}"))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    Lerc1(u32),
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

    pub fn ndarray_shape(&self) -> Vec<usize> {
        if self.depth <= 1 {
            vec![self.height as usize, self.width as usize]
        } else {
            vec![
                self.height as usize,
                self.width as usize,
                self.depth as usize,
            ]
        }
    }

    pub fn mask_ndarray_shape(&self) -> Vec<usize> {
        vec![self.height as usize, self.width as usize]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub info: BlobInfo,
    pub pixels: PixelData,
    pub mask: Option<Vec<u8>>,
}

impl Decoded {
    pub fn into_ndarray<T: NdArrayElement>(self) -> Result<ArrayD<T>> {
        let shape = self.info.ndarray_shape();
        self.pixels.into_ndarray(&shape)
    }

    pub fn into_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|e| {
                    Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {e}"))
                })
            })
            .transpose()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedF64 {
    pub info: BlobInfo,
    pub pixels: Vec<f64>,
    pub mask: Option<Vec<u8>>,
}

impl DecodedF64 {
    pub fn into_ndarray(self) -> Result<ArrayD<f64>> {
        ArrayD::from_shape_vec(IxDyn(&self.info.ndarray_shape()), self.pixels).map_err(|e| {
            Error::InvalidBlob(format!("failed to build ndarray from decoded pixels: {e}"))
        })
    }

    pub fn into_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|e| {
                    Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {e}"))
                })
            })
            .transpose()
    }
}

pub trait NdArrayElement: Sized {
    fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>>;
}

macro_rules! impl_exact_ndarray_element {
    ($ty:ty, $variant:ident, $name:literal) => {
        impl NdArrayElement for $ty {
            fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>> {
                match pixels {
                    PixelData::$variant(values) => Ok(values),
                    other => Err(Error::InvalidBlob(format!(
                        "cannot decode {} pixels into ndarray<{}>",
                        other.data_type().name(),
                        $name
                    ))),
                }
            }
        }
    };
}

impl_exact_ndarray_element!(i8, I8, "i8");
impl_exact_ndarray_element!(u8, U8, "u8");
impl_exact_ndarray_element!(i16, I16, "i16");
impl_exact_ndarray_element!(u16, U16, "u16");
impl_exact_ndarray_element!(i32, I32, "i32");
impl_exact_ndarray_element!(u32, U32, "u32");
impl_exact_ndarray_element!(f32, F32, "f32");

impl NdArrayElement for f64 {
    fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>> {
        Ok(pixels.to_f64())
    }
}
