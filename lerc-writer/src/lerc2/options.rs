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
    /// Whether reconstructed values are checked against `max_z_error`.
    pub error_bound: ErrorBound,
}

/// Whether the encoder checks every reconstructed value against the requested
/// `max_z_error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ErrorBound {
    /// Every reconstructed value is checked against the requested
    /// `max_z_error`, with a slack of `max_z_error * 1e-12 + 1e-12` for the
    /// check's own rounding. A block whose reconstruction
    /// `offset + q * 2 * max_z_error` would leave it, through `f64` rounding
    /// or after a raise of the header bound to a coarser grid, is stored raw,
    /// as is a constant block whose range exceeds it. Decoders may then still
    /// round the value to the output type.
    #[default]
    Checked,
    /// No check: blocks are bit-stuffed or stored as constants wherever
    /// libLerc would, and the encoder enforces no error bound after
    /// quantization. The decoded error is then libLerc's: up to the header's
    /// bound (the requested `max_z_error`, or the coarser grid it was raised
    /// to) plus about one ulp of the block's largest magnitude, the `f64`
    /// spacing there. Near 1e4 that is 1.8e-12; at 2^47 it is 0.03125, so
    /// `2^47 + k * 0.1` at a requested 0.001 decodes up to 0.03125 off, 31
    /// times the request, exactly as libLerc decodes its own blob. With
    /// [`EncoderProfile::Gdal`], the blobs are then GDAL's for these blocks
    /// too, typically Float64 data on a decimal grid next to a far-off value
    /// such as a `-9999` nodata sentinel. Requires [`EncoderProfile::Gdal`].
    Unchecked,
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
    /// lookup-table trigger, the doubled micro-block retry, the one-sweep
    /// fallback, libLerc's Huffman tables for lossless 8-bit data (code
    /// lengths from its heap, canonical codes, wrap-around symbol ranges, code
    /// table version 4), and its bit-mask padding. The output decodes to the
    /// same values as GDAL's and, for the cases covered by the test suite, is
    /// byte-identical, with these exceptions. Huffman code lengths of equally
    /// frequent symbols follow `std::priority_queue` as libstdc++ and the MSVC
    /// STL implement it; libc++ breaks ties otherwise. A Huffman blob's last 4
    /// bytes, which libLerc reserves for its decoder's read-ahead without
    /// writing them, are zeros here, as libLerc 4.1.0 and later leave them;
    /// libLerc 4.0.x leaves whatever the output buffer held.
    ///
    /// The profile always writes Lerc2 v4 with 8x8 base micro-blocks and
    /// rejects `no_data_value`, which v4 cannot store. A block whose
    /// quantized values would break `max_z_error` through `f64` rounding,
    /// which libLerc does not check, is stored raw instead, unless
    /// [`ErrorBound::Unchecked`] is selected.
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
            error_bound: ErrorBound::Checked,
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

    /// Selects whether reconstructed values are checked against
    /// `max_z_error`. See [`ErrorBound`].
    pub const fn with_error_bound(mut self, error_bound: ErrorBound) -> Self {
        self.error_bound = error_bound;
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
    if options.error_bound == ErrorBound::Unchecked && options.profile != EncoderProfile::Gdal {
        return Err(Error::InvalidArgument(
            "ErrorBound::Unchecked requires the GDAL encoder profile",
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
