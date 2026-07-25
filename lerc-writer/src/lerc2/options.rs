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
}

impl EncodeOptions {
    /// Creates lossless options with canonical 8-by-8 micro-blocks.
    pub const fn new() -> Self {
        Self {
            max_z_error: 0.0,
            micro_block_size: 8,
            no_data_value: None,
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
    Ok(())
}
