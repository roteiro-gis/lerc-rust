use std::any::TypeId;

use lerc_band_materialize::{
    copy_band_values_into_slice, BandLayout as MaterializeLayout, BandMaterializer,
};
use lerc_core::{BandLayout, BandSetInfo, BlobInfo, Error, PixelData, Result};
use ndarray::{ArrayD, IxDyn};

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
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|err| {
                    Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {err}"))
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
        ArrayD::from_shape_vec(IxDyn(&self.info.ndarray_shape()), self.pixels).map_err(|err| {
            Error::InvalidBlob(format!(
                "failed to build ndarray from decoded pixels: {err}"
            ))
        })
    }

    pub fn into_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|err| {
                    Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {err}"))
                })
            })
            .transpose()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBandSet {
    pub info: BandSetInfo,
    pub bands: Vec<PixelData>,
    pub band_masks: Vec<Option<Vec<u8>>>,
}

impl DecodedBandSet {
    pub fn into_ndarray<T: BandElement>(self) -> Result<ArrayD<T>> {
        self.into_ndarray_with_layout(BandLayout::Interleaved)
    }

    pub fn into_ndarray_with_layout<T: BandElement>(self, layout: BandLayout) -> Result<ArrayD<T>> {
        let shape = self.info.ndarray_shape_for_layout(layout);
        let values = self.into_vec_with_layout(layout)?;
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|err| {
            Error::InvalidBlob(format!(
                "failed to build ndarray from decoded band set: {err}"
            ))
        })
    }

    pub fn into_vec_with_layout<T: BandElement>(self, layout: BandLayout) -> Result<Vec<T>> {
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

    pub fn copy_into_slice<T: BandElement>(self, layout: BandLayout, out: &mut [T]) -> Result<()> {
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
        into_band_mask_ndarray(self.info, self.band_masks)
    }
}

pub fn into_band_mask_ndarray(
    info: BandSetInfo,
    band_masks: Vec<Option<Vec<u8>>>,
) -> Result<Option<ArrayD<u8>>> {
    if band_masks.iter().all(Option::is_none) {
        return Ok(None);
    }

    let pixel_count = info.bands[0].pixel_count()?;
    let band_count = info.band_count();
    let shape = info.band_mask_ndarray_shape();

    if band_count == 1 {
        let mask = band_masks
            .into_iter()
            .next()
            .flatten()
            .unwrap_or_else(|| vec![1; pixel_count]);
        return ArrayD::from_shape_vec(IxDyn(&shape), mask)
            .map(Some)
            .map_err(|err| {
                Error::InvalidBlob(format!("failed to build ndarray from decoded mask: {err}"))
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
        .map_err(|err| {
            Error::InvalidBlob(format!(
                "failed to build ndarray from decoded band mask: {err}"
            ))
        })
}

trait SupportedElementValue: Copy + 'static + IntoF64 {
    const KIND: BandElementKind;
}

macro_rules! match_pixel_data_values {
    ($band:expr, |$values:ident| $body:expr) => {
        match $band {
            PixelData::I8($values) => $body,
            PixelData::U8($values) => $body,
            PixelData::I16($values) => $body,
            PixelData::U16($values) => $body,
            PixelData::I32($values) => $body,
            PixelData::U32($values) => $body,
            PixelData::F32($values) => $body,
            PixelData::F64($values) => $body,
        }
    };
}

fn copy_pixel_data_into_materializer<T: BandElement>(
    materializer: &mut BandMaterializer<T>,
    band_index: usize,
    band: PixelData,
) -> Result<()> {
    match_pixel_data_values!(band, |values| {
        copy_typed_values_into_materializer(materializer, band_index, &values)
    })
}

fn copy_typed_values_into_materializer<T: BandElement, U: SupportedElementValue>(
    materializer: &mut BandMaterializer<T>,
    band_index: usize,
    values: &[U],
) -> Result<()> {
    if T::KIND == U::KIND {
        let typed = unsafe { cast_slice::<U, T>(values) };
        return materializer
            .copy_band(band_index, typed)
            .map_err(materialize_error);
    }
    if T::KIND == BandElementKind::F64 {
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

fn copy_pixel_data_into_layout_slice<T: BandElement>(
    out: &mut [T],
    band_index: usize,
    pixel_count: usize,
    depth: usize,
    band_count: usize,
    layout: BandLayout,
    band: PixelData,
) -> Result<()> {
    match_pixel_data_values!(band, |values| {
        copy_typed_values_into_layout_slice(
            out,
            band_index,
            pixel_count,
            depth,
            band_count,
            layout,
            &values,
        )
    })
}

fn copy_typed_values_into_layout_slice<T: BandElement, U: SupportedElementValue>(
    out: &mut [T],
    band_index: usize,
    pixel_count: usize,
    depth: usize,
    band_count: usize,
    layout: BandLayout,
    values: &[U],
) -> Result<()> {
    if T::KIND == U::KIND {
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
    if T::KIND == BandElementKind::F64 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandElementKind {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

pub trait BandElement: NdArrayElement + private::Sealed + Copy + Default + 'static {
    const KIND: BandElementKind;
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

macro_rules! impl_band_element {
    ($ty:ty, $kind:ident) => {
        impl BandElement for $ty {
            const KIND: BandElementKind = BandElementKind::$kind;
        }

        impl SupportedElementValue for $ty {
            const KIND: BandElementKind = BandElementKind::$kind;
        }
    };
}

impl_band_element!(i8, I8);
impl_band_element!(u8, U8);
impl_band_element!(i16, I16);
impl_band_element!(u16, U16);
impl_band_element!(i32, I32);
impl_band_element!(u32, U32);
impl_band_element!(f32, F32);
impl_band_element!(f64, F64);

fn materialize_layout(layout: BandLayout) -> MaterializeLayout {
    match layout {
        BandLayout::Interleaved => MaterializeLayout::Interleaved,
        BandLayout::Bsq => MaterializeLayout::Bsq,
    }
}

fn materialize_error(err: lerc_band_materialize::MaterializeError) -> Error {
    Error::InvalidBlob(err.to_string())
}

trait PixelDataExt {
    fn into_ndarray<T: NdArrayElement>(self, shape: &[usize]) -> Result<ArrayD<T>>;
}

impl PixelDataExt for PixelData {
    fn into_ndarray<T: NdArrayElement>(self, shape: &[usize]) -> Result<ArrayD<T>> {
        ArrayD::from_shape_vec(IxDyn(shape), T::from_pixel_data(self)?).map_err(|err| {
            Error::InvalidBlob(format!(
                "failed to build ndarray from decoded pixels: {err}"
            ))
        })
    }
}
