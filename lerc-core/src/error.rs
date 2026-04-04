use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Truncated {
        offset: usize,
        needed: usize,
        available: usize,
    },
    OutputTooSmall {
        needed: usize,
        available: usize,
    },
    InvalidMagic,
    UnsupportedVersion(u32),
    UnsupportedFeature(&'static str),
    InvalidHeader(&'static str),
    InvalidArgument(String),
    InvalidBlob(String),
    ChecksumMismatch {
        expected: u32,
        actual: u32,
    },
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
