use lerc_core::{Error, Result};

/// Controls the accuracy and representation of a newly encoded LERC blob.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Maximum absolute reconstruction error for valid samples.
    pub max_z_error: f64,
    /// Width and height of the square micro-blocks used by tiled encoding.
    pub micro_block_size: u32,
    /// Optional per-sample no-data sentinel for rasters whose depth is greater than one.
    pub no_data_value: Option<f64>,
    /// How the encoder chooses among valid encodings.
    pub profile: EncoderProfile,
}

/// Selects how the encoder chooses among encodings that all honour
/// `max_z_error`.
///
/// Both profiles write standard Lerc2 that any conforming decoder reads, and
/// both keep every valid sample within `max_z_error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EncoderProfile {
    /// The smallest output this crate can produce. This may differ from other
    /// encoders in block layout and, for multidimensional integer data, use
    /// Lerc2 v5 difference tiles.
    #[default]
    Compact,
    /// Reproduces the Lerc2 blobs GDAL writes for `COMPRESS=LERC`,
    /// `LERC_DEFLATE` and `LERC_ZSTD` (GDAL's default `LERC_VERSION=2.4`).
    ///
    /// GDAL encodes through libLerc's codec v4, and this profile makes the
    /// same choices: the error-bound raise for decimal-grid data, block
    /// offsets, constant/raw/simple/lookup-table block modes with libLerc's
    /// lookup-table trigger, the doubled micro-block retry, and the Huffman
    /// and one-sweep fallbacks. The output decodes to the same values as
    /// GDAL's and, for the cases covered by the test suite, is byte-identical.
    ///
    /// The profile always writes Lerc2 v4 with 8x8 base micro-blocks and
    /// rejects `no_data_value`, which v4 cannot store. A block whose
    /// quantized values would break `max_z_error` through `f64` rounding,
    /// which libLerc does not check, is stored raw instead.
    Gdal,
}

impl EncodeOptions {
    /// Creates lossless options with canonical 8-by-8 micro-blocks.
    pub const fn new() -> Self {
        Self {
            max_z_error: 0.0,
            micro_block_size: 8,
            no_data_value: None,
            profile: EncoderProfile::Compact,
        }
    }

    /// Sets the requested maximum absolute reconstruction error.
    pub const fn with_max_z_error(mut self, max_z_error: f64) -> Self {
        self.max_z_error = max_z_error;
        self
    }

    /// Sets the square micro-block size. Supported values are 2 through 32.
    pub const fn with_micro_block_size(mut self, micro_block_size: u32) -> Self {
        self.micro_block_size = micro_block_size;
        self
    }

    /// Sets the no-data sentinel used by multidimensional rasters.
    pub const fn with_no_data_value(mut self, no_data_value: f64) -> Self {
        self.no_data_value = Some(no_data_value);
        self
    }

    /// Selects the encoder profile. See [`EncoderProfile`].
    pub const fn with_profile(mut self, profile: EncoderProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Removes a previously configured no-data sentinel.
    pub const fn without_no_data_value(mut self) -> Self {
        self.no_data_value = None;
        self
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

pub(super) fn validate(options: EncodeOptions) -> Result<()> {
    if !options.max_z_error.is_finite() || options.max_z_error < 0.0 {
        return Err(Error::InvalidArgument(
            "max_z_error must be finite and non-negative",
        ));
    }
    if !(2..=32).contains(&options.micro_block_size) {
        return Err(Error::InvalidArgument(
            "micro_block_size must be in the range 2..=32",
        ));
    }
    if options
        .no_data_value
        .is_some_and(|no_data_value| !no_data_value.is_finite())
    {
        return Err(Error::InvalidArgument(
            "no_data_value must be finite when provided",
        ));
    }
    if options.profile == EncoderProfile::Gdal {
        if options.micro_block_size != GDAL_MICRO_BLOCK_SIZE {
            return Err(Error::InvalidArgument(
                "the GDAL encoder profile uses 8x8 micro-blocks",
            ));
        }
        if options.no_data_value.is_some() {
            return Err(Error::InvalidArgument(
                "the GDAL encoder profile writes Lerc2 v4, which cannot store no_data_value",
            ));
        }
    }
    Ok(())
}

/// Base micro-block size of libLerc, and therefore of GDAL.
const GDAL_MICRO_BLOCK_SIZE: u32 = 8;
