use std::any::TypeId;

use crate::materialize::{copy_band_values_into_slice, BandMaterializer};
use lerc_core::{BandLayout, BandSetInfo, BlobInfo, Error, PixelData, Result};
#[cfg(feature = "ndarray")]
use ndarray::{ArrayD, IxDyn};

#[cfg(feature = "ndarray")]
use crate::allocation::{checked_mul, default_vec, vec_with_capacity};

/// Native-typed pixels, validated metadata, and an optional validity mask for one blob.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// Validated blob metadata.
    pub info: BlobInfo,
    /// Pixel-interleaved samples in the blob's native numeric type.
    pub pixels: PixelData,
    /// Row-major validity bytes, when the blob has a mask.
    pub mask: Option<Vec<u8>>,
}

impl Decoded {
    #[cfg(feature = "ndarray")]
    /// Borrows this result and copies its pixels into an ndarray.
    ///
    /// # Errors
    /// Returns an error if `T` differs from the native type or the shape is invalid.
    pub fn to_ndarray<T: NdArrayElement>(&self) -> Result<ArrayD<T>> {
        self.pixels.clone().into_ndarray(&self.info.ndarray_shape())
    }

    #[cfg(feature = "ndarray")]
    /// Consumes this result and moves its pixels into an ndarray.
    ///
    /// # Errors
    /// Returns an error if `T` differs from the native type or the shape is invalid.
    pub fn into_ndarray<T: NdArrayElement>(self) -> Result<ArrayD<T>> {
        let shape = self.info.ndarray_shape();
        self.pixels.into_ndarray(&shape)
    }

    #[cfg(feature = "ndarray")]
    /// Consumes this result and moves its optional mask into an ndarray.
    ///
    /// # Errors
    /// Returns an error if decoded metadata and mask length disagree.
    pub fn into_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|err| {
                    Error::invalid_blob(format!("failed to build ndarray from decoded mask: {err}"))
                })
            })
            .transpose()
    }

    #[cfg(feature = "ndarray")]
    /// Copies the optional validity mask into an ndarray.
    ///
    /// # Errors
    /// Returns an error if decoded metadata and mask length disagree.
    pub fn mask_ndarray(&self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .as_ref()
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask.clone()).map_err(|err| {
                    Error::invalid_blob(format!("failed to build ndarray from decoded mask: {err}"))
                })
            })
            .transpose()
    }
}

/// Promoted `f64` pixels, validated metadata, and an optional mask for one blob.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedF64 {
    /// Validated blob metadata.
    pub info: BlobInfo,
    /// Pixel-interleaved samples promoted to `f64`.
    pub pixels: Vec<f64>,
    /// Row-major validity bytes, when present.
    pub mask: Option<Vec<u8>>,
}

impl DecodedF64 {
    #[cfg(feature = "ndarray")]
    /// Copies the promoted pixels into an ndarray.
    ///
    /// # Errors
    /// Returns an error if metadata and sample length disagree.
    pub fn to_ndarray(&self) -> Result<ArrayD<f64>> {
        ArrayD::from_shape_vec(IxDyn(&self.info.ndarray_shape()), self.pixels.clone()).map_err(
            |err| {
                Error::invalid_blob(format!(
                    "failed to build ndarray from decoded pixels: {err}"
                ))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    /// Moves the promoted pixels into an ndarray.
    ///
    /// # Errors
    /// Returns an error if metadata and sample length disagree.
    pub fn into_ndarray(self) -> Result<ArrayD<f64>> {
        ArrayD::from_shape_vec(IxDyn(&self.info.ndarray_shape()), self.pixels).map_err(|err| {
            Error::invalid_blob(format!(
                "failed to build ndarray from decoded pixels: {err}"
            ))
        })
    }

    #[cfg(feature = "ndarray")]
    /// Moves the optional validity mask into an ndarray.
    ///
    /// # Errors
    /// Returns an error if metadata and mask length disagree.
    pub fn into_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|err| {
                    Error::invalid_blob(format!("failed to build ndarray from decoded mask: {err}"))
                })
            })
            .transpose()
    }

    #[cfg(feature = "ndarray")]
    /// Copies the optional validity mask into an ndarray.
    ///
    /// # Errors
    /// Returns an error if metadata and mask length disagree.
    pub fn mask_ndarray(&self) -> Result<Option<ArrayD<u8>>> {
        let shape = self.info.mask_ndarray_shape();
        self.mask
            .as_ref()
            .map(|mask| {
                ArrayD::from_shape_vec(IxDyn(&shape), mask.clone()).map_err(|err| {
                    Error::invalid_blob(format!("failed to build ndarray from decoded mask: {err}"))
                })
            })
            .transpose()
    }
}

/// Native-typed bands, per-band masks, and shared band-set metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBandSet {
    /// Validated metadata for every band.
    pub info: BandSetInfo,
    /// Native-typed pixel buffers in band order.
    pub bands: Vec<PixelData>,
    /// Logical mask for each band, including inherited masks.
    pub band_masks: Vec<Option<Vec<u8>>>,
}

impl DecodedBandSet {
    #[cfg(feature = "ndarray")]
    /// Copies the band set into an interleaved ndarray.
    ///
    /// # Errors
    /// Returns an error for incompatible types or inconsistent shapes.
    pub fn to_ndarray<T: BandElement>(&self) -> Result<ArrayD<T>> {
        self.clone().into_ndarray()
    }

    #[cfg(feature = "ndarray")]
    /// Moves the band set into an interleaved ndarray.
    ///
    /// # Errors
    /// Returns an error for incompatible types or inconsistent shapes.
    pub fn into_ndarray<T: BandElement>(self) -> Result<ArrayD<T>> {
        self.into_ndarray_with_layout(BandLayout::Interleaved)
    }

    #[cfg(feature = "ndarray")]
    /// Moves the band set into an ndarray using the requested layout.
    ///
    /// # Errors
    /// Returns an error for incompatible types or inconsistent shapes.
    pub fn into_ndarray_with_layout<T: BandElement>(self, layout: BandLayout) -> Result<ArrayD<T>> {
        let shape = self.info.ndarray_shape_for_layout(layout);
        let values = self.into_vec_with_layout(layout)?;
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|err| {
            Error::invalid_blob(format!(
                "failed to build ndarray from decoded band set: {err}"
            ))
        })
    }

    /// Materializes every band as one homogeneous vector in `layout`.
    ///
    /// # Errors
    /// Returns an error when band types cannot convert to `T` or metadata is inconsistent.
    pub fn into_vec_with_layout<T: BandElement>(self, layout: BandLayout) -> Result<Vec<T>> {
        if self.bands.len() == 1 {
            let band = self
                .bands
                .into_iter()
                .next()
                .ok_or(Error::Internal("single-band decode lost its only band"))?;
            return T::from_pixel_data(band);
        }

        let mut materializer = BandMaterializer::new(
            self.info.bands[0].pixel_count()?,
            self.info.depth() as usize,
            self.info.band_count(),
            layout,
        )?;
        for (band_index, band) in self.bands.into_iter().enumerate() {
            copy_pixel_data_into_materializer(&mut materializer, band_index, band)?;
        }
        materializer.finish()
    }

    /// Copies every band into an exactly sized caller-provided slice.
    ///
    /// # Errors
    /// Returns an error for the wrong output length, incompatible types, or bad metadata.
    pub fn copy_into_slice<T: BandElement>(self, layout: BandLayout, out: &mut [T]) -> Result<()> {
        let pixel_count = self.info.bands[0].pixel_count()?;
        let depth = self.info.depth() as usize;
        let band_count = self.info.band_count();
        let expected_len = self.info.value_count()?;
        if out.len() != expected_len {
            return Err(Error::InvalidArgument(
                "output slice length does not match decoded band set length",
            ));
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

    #[cfg(feature = "ndarray")]
    /// Moves per-band masks into an ndarray.
    ///
    /// # Errors
    /// Returns an error if mask metadata and lengths disagree.
    pub fn into_band_mask_ndarray(self) -> Result<Option<ArrayD<u8>>> {
        band_masks_into_ndarray(self.info, self.band_masks)
    }

    #[cfg(feature = "ndarray")]
    /// Copies per-band masks into an ndarray.
    ///
    /// # Errors
    /// Returns an error if mask metadata and lengths disagree.
    pub fn band_mask_ndarray(&self) -> Result<Option<ArrayD<u8>>> {
        band_masks_into_ndarray(self.info.clone(), self.band_masks.clone())
    }
}

#[cfg(feature = "ndarray")]
pub(crate) fn band_masks_into_ndarray(
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
            .map(Ok)
            .unwrap_or_else(|| default_vec(pixel_count, "decoded band mask"))?;
        return ArrayD::from_shape_vec(IxDyn(&shape), mask)
            .map(Some)
            .map_err(|err| {
                Error::invalid_blob(format!("failed to build ndarray from decoded mask: {err}"))
            });
    }

    let merged_len = checked_mul(pixel_count, band_count, "decoded band mask length")?;
    let mut merged = vec_with_capacity(merged_len, "decoded band mask")?;
    for pixel in 0..pixel_count {
        for band_mask in &band_masks {
            merged.push(band_mask.as_ref().map(|mask| mask[pixel]).unwrap_or(1));
        }
    }

    ArrayD::from_shape_vec(IxDyn(&shape), merged)
        .map(Some)
        .map_err(|err| {
            Error::invalid_blob(format!(
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
        // SAFETY: equal BandElementKind values mean T and U are the same
        // supported primitive type, so the slice layout and alignment are
        // unchanged.
        let typed = unsafe { cast_slice::<U, T>(values) };
        return materializer.copy_band(band_index, typed);
    }
    if T::KIND == BandElementKind::F64 {
        return materializer.copy_band_with(band_index, |index| {
            // SAFETY: this branch is only entered when T is f64.
            unsafe_cast::<T, f64>(values[index].into_f64())
        });
    }
    Err(Error::invalid_blob(format!(
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
        // SAFETY: equal BandElementKind values mean T and U are the same
        // supported primitive type, so the slice layout and alignment are
        // unchanged.
        let typed = unsafe { cast_slice::<U, T>(values) };
        return copy_band_values_into_slice(
            out,
            typed,
            pixel_count,
            depth,
            band_index,
            band_count,
            layout,
        );
    }
    if T::KIND == BandElementKind::F64 {
        let band_len = pixel_count
            .checked_mul(depth.max(1))
            .ok_or(Error::SizeOverflow("decoded band length"))?;
        if values.len() != band_len {
            return Err(Error::invalid_blob(
                "decoded band length does not match its metadata",
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
            // SAFETY: this branch is only entered when T is f64.
            out[out_index] = unsafe_cast::<T, f64>(value.into_f64());
        }
        return Ok(());
    }
    Err(Error::invalid_blob(format!(
        "cannot decode {} pixels into ndarray<{}>",
        data_type_name::<U>(),
        std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("unknown"),
    )))
}

unsafe fn cast_slice<U, T>(values: &[U]) -> &[T] {
    debug_assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
    debug_assert_eq!(std::mem::align_of::<U>(), std::mem::align_of::<T>());
    // SAFETY: callers must guarantee U and T are the same primitive element type.
    unsafe { &*(values as *const [U] as *const [T]) }
}

fn unsafe_cast<T, U: Copy>(value: U) -> T {
    debug_assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
    debug_assert_eq!(std::mem::align_of::<U>(), std::mem::align_of::<T>());
    // SAFETY: callers only use this helper for same-size primitive casts where
    // T is known by branch guards to match the source value representation.
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

/// Runtime discriminator for supported homogeneous band output types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandElementKind {
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Single-precision float.
    F32,
    /// Double-precision float.
    F64,
}

/// Sealed primitive types accepted by homogeneous band-set decode APIs.
pub trait BandElement: private::Sealed + Copy + Default + Send + Sync + 'static {
    /// Runtime kind corresponding to this primitive.
    const KIND: BandElementKind;
    /// Converts a native decoded buffer when its representation is compatible.
    fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>>;
}

#[cfg(feature = "ndarray")]
/// Sealed primitive types accepted by single-blob ndarray decode APIs.
pub trait NdArrayElement: BandElement {}

macro_rules! impl_exact_band_element {
    ($ty:ty, $variant:ident, $name:literal) => {
        impl BandElement for $ty {
            const KIND: BandElementKind = BandElementKind::$variant;

            fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>> {
                match pixels {
                    PixelData::$variant(values) => Ok(values),
                    other => Err(Error::invalid_blob(format!(
                        "cannot decode {} pixels into ndarray<{}>",
                        other.data_type().name(),
                        $name
                    ))),
                }
            }
        }
    };
}

impl_exact_band_element!(i8, I8, "i8");
impl_exact_band_element!(u8, U8, "u8");
impl_exact_band_element!(i16, I16, "i16");
impl_exact_band_element!(u16, U16, "u16");
impl_exact_band_element!(i32, I32, "i32");
impl_exact_band_element!(u32, U32, "u32");
impl_exact_band_element!(f32, F32, "f32");

impl BandElement for f64 {
    const KIND: BandElementKind = BandElementKind::F64;

    fn from_pixel_data(pixels: PixelData) -> Result<Vec<Self>> {
        Ok(pixels.to_f64())
    }
}

#[cfg(feature = "ndarray")]
macro_rules! impl_ndarray_element {
    ($($ty:ty),+ $(,)?) => { $(impl NdArrayElement for $ty {})+ };
}

#[cfg(feature = "ndarray")]
impl_ndarray_element!(i8, u8, i16, u16, i32, u32, f32, f64);

macro_rules! impl_band_element {
    ($ty:ty, $kind:ident) => {
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

#[cfg(feature = "ndarray")]
trait PixelDataExt {
    fn into_ndarray<T: NdArrayElement>(self, shape: &[usize]) -> Result<ArrayD<T>>;
}

#[cfg(feature = "ndarray")]
impl PixelDataExt for PixelData {
    fn into_ndarray<T: NdArrayElement>(self, shape: &[usize]) -> Result<ArrayD<T>> {
        ArrayD::from_shape_vec(IxDyn(shape), T::from_pixel_data(self)?).map_err(|err| {
            Error::invalid_blob(format!(
                "failed to build ndarray from decoded pixels: {err}"
            ))
        })
    }
}
