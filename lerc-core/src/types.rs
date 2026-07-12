use crate::error::{Error, Result};

/// Primitive numeric representations supported by the LERC format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
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
    /// IEEE-754 single-precision float.
    F32,
    /// IEEE-754 double-precision float.
    F64,
}

impl DataType {
    /// Returns whether this is an integer representation.
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::U8 | Self::I16 | Self::U16 | Self::I32 | Self::U32
        )
    }

    /// Converts a LERC numeric type code.
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
            _ => Err(Error::invalid_blob(format!(
                "unsupported data type code {code}"
            ))),
        }
    }

    /// Returns the LERC numeric type code.
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

    /// Returns the encoded byte width of one sample.
    pub fn byte_len(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Returns the canonical Rust-style type name.
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

/// Owned homogeneous pixel samples with their runtime numeric type.
#[derive(Debug, Clone, PartialEq)]
pub enum PixelData {
    /// Signed 8-bit samples.
    I8(Vec<i8>),
    /// Unsigned 8-bit samples.
    U8(Vec<u8>),
    /// Signed 16-bit samples.
    I16(Vec<i16>),
    /// Unsigned 16-bit samples.
    U16(Vec<u16>),
    /// Signed 32-bit samples.
    I32(Vec<i32>),
    /// Unsigned 32-bit samples.
    U32(Vec<u32>),
    /// Single-precision samples.
    F32(Vec<f32>),
    /// Double-precision samples.
    F64(Vec<f64>),
}

impl PixelData {
    /// Returns the numeric type of this buffer.
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

    /// Returns the number of samples.
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

    /// Returns whether the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Promotes all samples to a new `f64` buffer.
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

/// Memory layout for a multi-band raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandLayout {
    /// Pixels are contiguous, with band values inside each pixel.
    Interleaved,
    /// Each complete band occupies one contiguous plane (band sequential).
    Bsq,
}

/// LERC format family and encoded version number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// Legacy CntZImage/Lerc1 version.
    Lerc1(u32),
    /// Lerc2 version.
    Lerc2(u32),
}

/// How a decoded blob represents pixel validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskEncoding {
    /// Every pixel is valid and no mask is needed.
    None,
    /// This blob stores explicit mask bytes.
    Explicit,
    /// This blob inherits a mask supplied externally or by a preceding band.
    External,
    /// Every pixel is invalid, represented without stored mask bytes.
    ImplicitAllInvalid,
}

impl MaskEncoding {
    /// Returns the number of stored or implicit masks represented by this encoding.
    pub fn mask_count(self) -> u32 {
        match self {
            Self::None | Self::External => 0,
            Self::Explicit | Self::ImplicitAllInvalid => 1,
        }
    }

    /// Returns whether decoding produces a logical mask.
    pub fn has_mask(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns whether mask bytes must come from outside this blob.
    pub fn uses_external_mask(self) -> bool {
        matches!(self, Self::External)
    }

    /// Returns whether the blob stores an explicit mask payload.
    pub fn stores_mask_bytes(self) -> bool {
        matches!(self, Self::Explicit)
    }
}

/// Validated metadata for one LERC blob.
#[derive(Debug, Clone, PartialEq)]
pub struct BlobInfo {
    /// Format family and version.
    pub version: Version,
    /// Numeric representation stored by the blob.
    pub data_type: DataType,
    /// Raster width in pixels.
    pub width: u32,
    /// Raster height in pixels.
    pub height: u32,
    /// Samples per pixel.
    pub depth: u32,
    /// Optional per-depth minima when the format carries them.
    pub min_values: Option<Vec<f64>>,
    /// Optional per-depth maxima when the format carries them.
    pub max_values: Option<Vec<f64>>,
    /// Number of pixels marked valid.
    pub valid_pixel_count: u32,
    /// Encoded square micro-block size, or zero for Lerc1.
    pub micro_block_size: u32,
    /// Exact encoded length of this blob in bytes.
    pub blob_size: usize,
    /// Maximum absolute reconstruction error declared by the blob.
    pub max_z_error: f64,
    /// Minimum encoded valid sample value.
    pub z_min: f64,
    /// Maximum encoded valid sample value.
    pub z_max: f64,
    /// Validity-mask representation.
    pub mask_encoding: MaskEncoding,
    /// Original no-data sentinel when Lerc2 v6 carries one.
    pub no_data_value: Option<f64>,
}

impl BlobInfo {
    /// Returns `width * height` using checked arithmetic.
    pub fn pixel_count(&self) -> Result<usize> {
        usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(Error::SizeOverflow("blob pixel count"))
    }

    /// Returns `width * height * depth` using checked arithmetic.
    pub fn sample_count(&self) -> Result<usize> {
        self.pixel_count()?
            .checked_mul(self.depth as usize)
            .ok_or(Error::SizeOverflow("blob sample count"))
    }

    /// Returns the number of logical masks represented by this blob.
    pub fn mask_count(&self) -> u32 {
        self.mask_encoding.mask_count()
    }

    /// Returns whether the decoded blob has a validity mask.
    pub fn has_mask(&self) -> bool {
        self.mask_encoding.has_mask()
    }

    /// Returns whether the blob requires an externally supplied mask.
    pub fn uses_external_mask(&self) -> bool {
        self.mask_encoding.uses_external_mask()
    }

    /// Returns whether the blob carries an active no-data sentinel.
    pub fn uses_no_data_value(&self) -> bool {
        self.no_data_value.is_some()
    }

    /// Returns the natural row-major raster shape.
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

    /// Returns the row-major mask shape `[height, width]`.
    pub fn mask_shape(&self) -> Vec<usize> {
        vec![self.height as usize, self.width as usize]
    }

    /// Alias for [`Self::raster_shape`] used by ndarray integrations.
    pub fn ndarray_shape(&self) -> Vec<usize> {
        self.raster_shape()
    }

    /// Alias for [`Self::mask_shape`] used by ndarray integrations.
    pub fn mask_ndarray_shape(&self) -> Vec<usize> {
        self.mask_shape()
    }
}

/// Validated metadata for a concatenated set of same-shaped LERC bands.
#[derive(Debug, Clone, PartialEq)]
pub struct BandSetInfo {
    /// Per-band blob metadata in encoded order.
    pub bands: Vec<BlobInfo>,
}

impl BandSetInfo {
    /// Validates and constructs band-set metadata.
    ///
    /// # Errors
    /// Returns an error for an empty set or mismatched band dimensions/depths.
    pub fn new(bands: Vec<BlobInfo>) -> Result<Self> {
        let first = bands
            .first()
            .ok_or_else(|| Error::invalid_blob("LERC band set is empty"))?;
        for band in &bands[1..] {
            if band.width != first.width || band.height != first.height || band.depth != first.depth
            {
                return Err(Error::invalid_blob(
                    "LERC band set contains mismatched raster dimensions",
                ));
            }
        }
        Ok(Self { bands })
    }

    /// Returns the number of bands.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Returns the shared raster width.
    pub fn width(&self) -> u32 {
        self.bands[0].width
    }

    /// Returns the shared raster height.
    pub fn height(&self) -> u32 {
        self.bands[0].height
    }

    /// Returns the shared samples-per-pixel depth.
    pub fn depth(&self) -> u32 {
        self.bands[0].depth
    }

    /// Returns the number of masks required to represent the band set.
    pub fn mask_count(&self) -> usize {
        let first_valid_pixel_count = self.bands[0].valid_pixel_count;
        let first_mask_count = self.bands[0].mask_count() as usize;
        let has_distinct_band_masks = self.bands[1..].iter().any(|band| {
            band.mask_encoding.stores_mask_bytes()
                || band.valid_pixel_count != first_valid_pixel_count
        });

        if has_distinct_band_masks {
            self.band_count()
        } else {
            first_mask_count
        }
    }

    /// Returns whether any band uses a no-data sentinel.
    pub fn uses_no_data_value(&self) -> bool {
        self.bands.iter().any(BlobInfo::uses_no_data_value)
    }

    /// Returns the natural interleaved raster shape.
    pub fn raster_shape(&self) -> Vec<usize> {
        self.raster_shape_for_layout(BandLayout::Interleaved)
    }

    /// Returns the raster shape for the requested band layout.
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

    /// Returns `[band, height, width]` when masks differ, otherwise `[height, width]`.
    pub fn band_mask_shape(&self) -> Vec<usize> {
        let height = self.height() as usize;
        let width = self.width() as usize;
        match self.band_count() {
            1 => vec![height, width],
            band_count => vec![height, width, band_count],
        }
    }

    /// Returns the checked sample count across all bands.
    pub fn value_count(&self) -> Result<usize> {
        self.bands[0]
            .pixel_count()?
            .checked_mul(self.band_count())
            .and_then(|count| count.checked_mul((self.depth() as usize).max(1)))
            .ok_or(Error::SizeOverflow("band set value count"))
    }

    /// Alias for [`Self::raster_shape`] used by ndarray integrations.
    pub fn ndarray_shape(&self) -> Vec<usize> {
        self.raster_shape()
    }

    /// Alias for [`Self::raster_shape_for_layout`] used by ndarray integrations.
    pub fn ndarray_shape_for_layout(&self, layout: BandLayout) -> Vec<usize> {
        self.raster_shape_for_layout(layout)
    }

    /// Alias for [`Self::band_mask_shape`] used by ndarray integrations.
    pub fn band_mask_ndarray_shape(&self) -> Vec<usize> {
        self.band_mask_shape()
    }
}
