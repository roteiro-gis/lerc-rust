use std::borrow::Cow;
use std::fmt;

/// Result type used by all LERC crates.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned while validating, decoding, or encoding LERC data.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Input ended before a required field or payload was complete.
    Truncated {
        /// Byte offset at which the incomplete read began.
        offset: usize,
        /// Number of bytes required by the read.
        needed: usize,
        /// Number of bytes available from `offset`.
        available: usize,
    },
    /// A caller-provided output buffer cannot hold the result.
    OutputTooSmall {
        /// Minimum number of bytes required.
        needed: usize,
        /// Number of bytes supplied by the caller.
        available: usize,
    },
    /// The input does not start with a supported LERC signature.
    InvalidMagic,
    /// The blob uses a format version this implementation does not support.
    UnsupportedVersion(u32),
    /// The blob requests a recognized but unsupported format feature.
    UnsupportedFeature(&'static str),
    /// A fixed header field violates the LERC format constraints.
    InvalidHeader(&'static str),
    /// A public API argument is invalid.
    InvalidArgument(&'static str),
    /// Encoded data is structurally inconsistent or corrupt.
    InvalidBlob(Cow<'static, str>),
    /// Checked size arithmetic overflowed.
    SizeOverflow(&'static str),
    /// An internal codec invariant was violated.
    Internal(&'static str),
    /// Reading from an external stream failed.
    Io {
        /// Portable category of the underlying I/O failure.
        kind: std::io::ErrorKind,
        /// Contextualized error message.
        message: Cow<'static, str>,
    },
    /// A Lerc2 checksum did not match the encoded payload.
    ChecksumMismatch {
        /// Checksum stored in the blob header.
        expected: u32,
        /// Checksum computed from the payload.
        actual: u32,
    },
}

impl Error {
    /// Constructs a corrupt-blob error from static or owned context.
    pub fn invalid_blob(reason: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidBlob(reason.into())
    }

    /// Converts an I/O error while retaining its portable kind and operation context.
    pub fn io(context: &'static str, error: std::io::Error) -> Self {
        Self::Io {
            kind: error.kind(),
            message: format!("{context}: {error}").into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated input at offset {offset}: need {needed} bytes, have {available}"
            ),
            Self::OutputTooSmall { needed, available } => write!(
                f,
                "output buffer too small: need {needed} bytes, have {available}"
            ),
            Self::InvalidMagic => write!(f, "invalid LERC magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported LERC version {version}")
            }
            Self::UnsupportedFeature(feature) => write!(f, "unsupported LERC feature: {feature}"),
            Self::InvalidHeader(reason) => write!(f, "invalid LERC header: {reason}"),
            Self::InvalidArgument(reason) => write!(f, "invalid argument: {reason}"),
            Self::InvalidBlob(reason) => write!(f, "invalid LERC blob: {reason}"),
            Self::SizeOverflow(context) => write!(f, "size overflow while computing {context}"),
            Self::Internal(reason) => write!(f, "internal codec invariant failed: {reason}"),
            Self::Io { message, .. } => write!(f, "I/O error: {message}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "LERC checksum mismatch: expected {expected:#010x}, got {actual:#010x}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
