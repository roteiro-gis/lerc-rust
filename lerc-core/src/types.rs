use std::any::TypeId;

use crate::error::{Error, Result};
use lerc_band_materialize::{
    copy_band_values_into_slice, BandLayout as MaterializeLayout, BandMaterializer,
};
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

    pub fn value_count(&self) -> Result<usize> {
        self.bands[0]
            .pixel_count()?
            .checked_mul(self.band_count())
            .and_then(|n| n.checked_mul((self.depth() as usize).max(1)))
            .ok_or_else(|| Error::InvalidBlob("LERC ndarray size overflows usize".into()))
    }

    pub fn into_band_mask_ndarray(
        self,
        band_masks: Vec<Option<Vec<u8>>>,
    ) -> Result<Option<ArrayD<u8>>> {
        if band_masks.iter().all(Option::is_none) {
            return Ok(None);
        }

        let pixel_count = self.bands[0].pixel_count()?;
        let band_count = self.band_count();
        let shape = self.band_mask_ndarray_shape();

        if band_count == 1 {
            let mask = band_masks
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
            for band_mask in &band_masks {
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

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBandSet {
    pub info: BandSetInfo,
    pub bands: Vec<PixelData>,
    pub band_masks: Vec<Option<Vec<u8>>>,
}

impl DecodedBandSet {
    pub fn into_ndarray<T: NdArrayElement + 'static>(self) -> Result<ArrayD<T>> {
        self.into_ndarray_with_layout(BandLayout::Interleaved)
    }

    pub fn into_ndarray_with_layout<T: NdArrayElement + 'static>(
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

    pub fn into_vec_with_layout<T: NdArrayElement + 'static>(
        self,
        layout: BandLayout,
    ) -> Result<Vec<T>> {
        if self.bands.len() == 1 {
            return T::from_pixel_data(self.bands.into_iter().next().unwrap());
        }

        let mut materializer = BandMaterializer::new(
            self.info.bands[0].pixel_count()?,
            self.info.depth() as usize,
            self.info.band_count(),
            materialize_layout(layout),
        )
        .map_err(materialize_error)?;
        for (band_index, band) in self.bands.into_iter().enumerate() {
            copy_pixel_data_into_materializer(&mut materializer, band_index, band)?;
        }
        materializer.finish().map_err(materialize_error)
    }

    pub fn copy_into_slice<T: NdArrayElement + 'static>(
        self,
        layout: BandLayout,
        out: &mut [T],
    ) -> Result<()> {
        let pixel_count = self.info.bands[0].pixel_count()?;
        let depth = self.info.depth() as usize;
        let band_count = self.info.band_count();
        let expected_len = self.info.value_count()?;
        if out.len() != expected_len {
            return Err(Error::InvalidBlob(format!(
                "output slice length {} does not match decoded band set length {}",
                out.len(),
                expected_len
            )));
        }

        for (band_index, band) in self.bands.into_iter().enumerate() {
            copy_pixel_data_into_layout_slice(
                out,
                band_index,
                pixel_count,
                depth,
                band_count,
                layout,
                band,
            )?;
        }
        Ok(())
    }

    pub fn into_band_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        self.info.into_band_mask_ndarray(self.band_masks)
    }
}

fn materialize_layout(layout: BandLayout) -> MaterializeLayout {
    match layout {
        BandLayout::Interleaved => MaterializeLayout::Interleaved,
        BandLayout::Bsq => MaterializeLayout::Bsq,
    }
}

fn materialize_error(err: lerc_band_materialize::MaterializeError) -> Error {
    Error::InvalidBlob(err.to_string())
}

fn copy_pixel_data_into_materializer<T: NdArrayElement + 'static>(
    materializer: &mut BandMaterializer<T>,
    band_index: usize,
    band: PixelData,
) -> Result<()> {
    match band {
        PixelData::I8(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::U8(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::I16(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::U16(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::I32(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::U32(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::F32(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
        PixelData::F64(values) => {
            copy_typed_values_into_materializer(materializer, band_index, &values)
        }
    }
}

fn copy_typed_values_into_materializer<T: NdArrayElement + 'static, U: Copy + 'static + IntoF64>(
    materializer: &mut BandMaterializer<T>,
    band_index: usize,
    values: &[U],
) -> Result<()> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        let typed = unsafe { cast_slice::<U, T>(values) };
        return materializer
            .copy_band(band_index, typed)
            .map_err(materialize_error);
    }
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        return materializer
            .copy_band_with(band_index, |index| {
                unsafe_cast::<T, f64>(values[index].into_f64())
            })
            .map_err(materialize_error);
    }
    Err(Error::InvalidBlob(format!(
        "cannot decode {} pixels into ndarray<{}>",
        data_type_name::<U>(),
        std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("unknown"),
    )))
}

fn copy_pixel_data_into_layout_slice<T: NdArrayElement + 'static>(
    out: &mut [T],
    band_index: usize,
    pixel_count: usize,
    depth: usize,
    band_count: usize,
    layout: BandLayout,
    band: PixelData,
) -> Result<()> {
    match band {
        PixelData::I8(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::U8(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::I16(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::U16(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::I32(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::U32(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::F32(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
        PixelData::F64(values) => copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        ),
    }
}

fn copy_typed_values_into_layout_slice<T: NdArrayElement + 'static, U: Copy + 'static + IntoF64>(
    out: &mut [T],
    band_index: usize,
    pixel_count: usize,
    depth: usize,
    band_count: usize,
    layout: BandLayout,
    values: &[U],
) -> Result<()> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        let typed = unsafe { cast_slice::<U, T>(values) };
        return copy_band_values_into_slice(
            out,
            typed,
            pixel_count,
            depth,
            band_index,
            band_count,
            materialize_layout(layout),
        )
        .map_err(materialize_error);
    }
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let band_len = pixel_count
            .checked_mul(depth.max(1))
            .ok_or_else(|| Error::InvalidBlob("decoded band length overflows usize".into()))?;
        if values.len() != band_len {
            return Err(Error::InvalidBlob(
                "decoded band length does not match its metadata".into(),
            ));
        }
        for (value_index, value) in values.iter().copied().enumerate() {
            let out_index = match layout {
                BandLayout::Interleaved => {
                    if depth <= 1 {
                        value_index * band_count + band_index
                    } else {
                        let pixel = value_index / depth;
                        let sample = value_index % depth;
                        (pixel * band_count + band_index) * depth + sample
                    }
                }
                BandLayout::Bsq => band_index * band_len + value_index,
            };
            out[out_index] = unsafe_cast::<T, f64>(value.into_f64());
        }
        return Ok(());
    }
    Err(Error::InvalidBlob(format!(
        "cannot decode {} pixels into ndarray<{}>",
        data_type_name::<U>(),
        std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("unknown"),
    )))
}

unsafe fn cast_slice<U, T>(values: &[U]) -> &[T] {
    &*(values as *const [U] as *const [T])
}

fn unsafe_cast<T, U: Copy>(value: U) -> T {
    unsafe { std::mem::transmute_copy(&value) }
}

trait IntoF64 {
    fn into_f64(self) -> f64;
}

impl IntoF64 for i8 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for u8 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for i16 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for u16 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for i32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for u32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

fn data_type_name<T: 'static>() -> &'static str {
    if TypeId::of::<T>() == TypeId::of::<i8>() {
        "i8"
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        "u8"
    } else if TypeId::of::<T>() == TypeId::of::<i16>() {
        "i16"
    } else if TypeId::of::<T>() == TypeId::of::<u16>() {
        "u16"
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        "i32"
    } else if TypeId::of::<T>() == TypeId::of::<u32>() {
        "u32"
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        "f32"
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        "f64"
    } else {
        "unknown"
    }
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
