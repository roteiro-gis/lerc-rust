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

    pub fn ndarray_shape(&self) -> Vec<usize> {
        self.ndarray_shape_for_layout(BandLayout::Interleaved)
    }

    pub fn ndarray_shape_for_layout(&self, layout: BandLayout) -> Vec<usize> {
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

    pub fn band_mask_ndarray_shape(&self) -> Vec<usize> {
        let height = self.height() as usize;
        let width = self.width() as usize;
        match self.band_count() {
            1 => vec![height, width],
            band_count => vec![height, width, band_count],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBandSet {
    pub info: BandSetInfo,
    pub bands: Vec<PixelData>,
    pub band_masks: Vec<Option<Vec<u8>>>,
}

impl DecodedBandSet {
    pub fn into_ndarray<T: NdArrayElement>(self) -> Result<ArrayD<T>> {
        self.into_ndarray_with_layout(BandLayout::Interleaved)
    }

    pub fn into_ndarray_with_layout<T: NdArrayElement>(
        self,
        layout: BandLayout,
    ) -> Result<ArrayD<T>> {
        let shape = self.info.ndarray_shape_for_layout(layout);
        let values = self.into_vec_with_layout(layout)?;
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
            Error::InvalidBlob(format!(
                "failed to build ndarray from decoded band set: {e}"
            ))
        })
    }

    pub fn into_vec_with_layout<T: NdArrayElement>(self, layout: BandLayout) -> Result<Vec<T>> {
        if self.bands.len() == 1 {
            return T::from_pixel_data(self.bands.into_iter().next().unwrap());
        }

        let pixel_count = self.info.bands[0].pixel_count()?;
        let depth = self.info.depth() as usize;
        let band_count = self.info.band_count();
        let sample_count = pixel_count
            .checked_mul(band_count)
            .and_then(|n| n.checked_mul(depth.max(1)))
            .ok_or_else(|| Error::InvalidBlob("LERC ndarray size overflows usize".into()))?;
        if sample_count == 0 {
            return Ok(Vec::new());
        }
        let mut bands = self.bands.into_iter();
        let first_band = T::from_pixel_data(
            bands
                .next()
                .ok_or_else(|| Error::InvalidBlob("LERC band set is empty".into()))?,
        )?;
        let seed = first_band.first().cloned().ok_or_else(|| {
            Error::InvalidBlob("decoded non-empty band set produced an empty band".into())
        })?;
        let mut out = vec![seed; sample_count];
        copy_band_values_into_slice(
            &mut out,
            &first_band,
            pixel_count,
            depth,
            0,
            band_count,
            layout,
        )?;

        for (band_index, band) in bands.enumerate() {
            let values = T::from_pixel_data(band)?;
            copy_band_values_into_slice(
                &mut out,
                &values,
                pixel_count,
                depth,
                band_index + 1,
                band_count,
                layout,
            )?;
        }

        Ok(out)
    }

    pub fn copy_into_slice<T: NdArrayElement>(
        self,
        layout: BandLayout,
        out: &mut [T],
    ) -> Result<()> {
        let pixel_count = self.info.bands[0].pixel_count()?;
        let depth = self.info.depth() as usize;
        let band_count = self.info.band_count();
        let expected_len = pixel_count
            .checked_mul(band_count)
            .and_then(|n| n.checked_mul(depth.max(1)))
            .ok_or_else(|| Error::InvalidBlob("LERC ndarray size overflows usize".into()))?;
        if out.len() != expected_len {
            return Err(Error::InvalidBlob(format!(
                "output slice length {} does not match decoded band set length {}",
                out.len(),
                expected_len
            )));
        }

        let bands: Vec<Vec<T>> = self
            .bands
            .into_iter()
            .map(T::from_pixel_data)
            .collect::<Result<_>>()?;
        for (band_index, band) in bands.iter().enumerate() {
            copy_band_values_into_slice(
                out,
                band,
                pixel_count,
                depth,
                band_index,
                band_count,
                layout,
            )?;
        }
        Ok(())
    }

    pub fn into_band_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        if self.band_masks.iter().all(Option::is_none) {
            return Ok(None);
        }

        let pixel_count = self.info.bands[0].pixel_count()?;
        let band_count = self.info.band_count();
        let shape = self.info.band_mask_ndarray_shape();

        if band_count == 1 {
            let mask = self
                .band_masks
                .into_iter()
                .next()
                .flatten()
                .unwrap_or_else(|| vec![1; pixel_count]);
            return ArrayD::from_shape_vec(IxDyn(&shape), mask)
                .map(Some)
                .map_err(|e| {
                    Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {e}"))
                });
        }

        let mut merged = Vec::with_capacity(pixel_count * band_count);
        for pixel in 0..pixel_count {
            for band_mask in &self.band_masks {
                merged.push(band_mask.as_ref().map(|mask| mask[pixel]).unwrap_or(1));
            }
        }

        ArrayD::from_shape_vec(IxDyn(&shape), merged)
            .map(Some)
            .map_err(|e| {
                Error::InvalidBlob(format!(
                    "failed to build ndarray from decoded band mask: {e}"
                ))
            })
    }
}

fn copy_band_values_into_slice<T: Clone>(
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
        .ok_or_else(|| Error::InvalidBlob("LERC ndarray size overflows usize".into()))?;
    if values.len() != band_len {
        return Err(Error::InvalidBlob(
            "LERC band set pixel buffers have inconsistent lengths".into(),
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

pub trait NdArrayElement: Sized + Clone {
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
