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
            Self::I8(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Self::I8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::F32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::F64(values) => values.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandLayout {
    Interleaved,
    Bsq,
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
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| Error::InvalidBlob("pixel count overflows usize".into()))
    }

    pub fn sample_count(&self) -> Result<usize> {
        self.pixel_count()?
            .checked_mul(self.depth as usize)
            .ok_or_else(|| Error::InvalidBlob("sample count overflows usize".into()))
    }

    pub fn raster_shape(&self) -> Vec<usize> {
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

    pub fn mask_shape(&self) -> Vec<usize> {
        vec![self.height as usize, self.width as usize]
    }

    pub fn ndarray_shape(&self) -> Vec<usize> {
        self.raster_shape()
    }

    pub fn mask_ndarray_shape(&self) -> Vec<usize> {
        self.mask_shape()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BandSetInfo {
    pub bands: Vec<BlobInfo>,
}

impl BandSetInfo {
    pub fn new(bands: Vec<BlobInfo>) -> Result<Self> {
        let first = bands
            .first()
            .ok_or_else(|| Error::InvalidBlob("LERC band set is empty".into()))?;
        for band in &bands[1..] {
            if band.width != first.width || band.height != first.height || band.depth != first.depth
            {
                return Err(Error::InvalidBlob(
                    "LERC band set contains mismatched raster dimensions".into(),
                ));
            }
        }
        Ok(Self { bands })
    }

    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    pub fn width(&self) -> u32 {
        self.bands[0].width
    }

    pub fn height(&self) -> u32 {
        self.bands[0].height
    }

    pub fn depth(&self) -> u32 {
        self.bands[0].depth
    }

    pub fn raster_shape(&self) -> Vec<usize> {
        self.raster_shape_for_layout(BandLayout::Interleaved)
    }

    pub fn raster_shape_for_layout(&self, layout: BandLayout) -> Vec<usize> {
        let height = self.height() as usize;
        let width = self.width() as usize;
        let depth = self.depth() as usize;
        let band_count = self.band_count();

        match layout {
            BandLayout::Interleaved => match (band_count, depth) {
                (1, 0 | 1) => vec![height, width],
                (1, depth) => vec![height, width, depth],
                (_, 0 | 1) => vec![height, width, band_count],
                (_, depth) => vec![height, width, band_count, depth],
            },
            BandLayout::Bsq => match (band_count, depth) {
                (1, 0 | 1) => vec![height, width],
                (1, depth) => vec![height, width, depth],
                (_, 0 | 1) => vec![band_count, height, width],
                (_, depth) => vec![band_count, height, width, depth],
            },
        }
    }

    pub fn band_mask_shape(&self) -> Vec<usize> {
        let height = self.height() as usize;
        let width = self.width() as usize;
        match self.band_count() {
            1 => vec![height, width],
            band_count => vec![height, width, band_count],
        }
    }

    pub fn value_count(&self) -> Result<usize> {
        self.bands[0]
            .pixel_count()?
            .checked_mul(self.band_count())
            .and_then(|count| count.checked_mul((self.depth() as usize).max(1)))
            .ok_or_else(|| Error::InvalidBlob("LERC raster size overflows usize".into()))
    }

    pub fn ndarray_shape(&self) -> Vec<usize> {
        self.raster_shape()
    }

    pub fn ndarray_shape_for_layout(&self, layout: BandLayout) -> Vec<usize> {
        self.raster_shape_for_layout(layout)
    }

    pub fn band_mask_ndarray_shape(&self) -> Vec<usize> {
        self.band_mask_shape()
    }
}
