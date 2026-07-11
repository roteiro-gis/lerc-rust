use std::borrow::Cow;

use lerc_core::{Error, MaskView, Result};

#[derive(Debug, Clone, Copy)]
pub(super) enum MaskKind<'a> {
    None,
    Explicit(&'a [u8]),
    External(&'a [u8]),
}

impl<'a> MaskKind<'a> {
    pub(super) fn data(self) -> Option<&'a [u8]> {
        match self {
            Self::None => None,
            Self::Explicit(mask) | Self::External(mask) => Some(mask),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskStorage {
    None,
    Explicit,
    External,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedMask<'a> {
    data: Option<Cow<'a, [u8]>>,
    storage: MaskStorage,
    derived: bool,
    payload: Vec<u8>,
}

impl<'a> PreparedMask<'a> {
    pub(super) fn from_kind(mask: MaskKind<'a>) -> Self {
        match mask {
            MaskKind::None => Self {
                data: None,
                storage: MaskStorage::None,
                derived: false,
                payload: Vec::new(),
            },
            MaskKind::Explicit(mask) => Self {
                data: Some(Cow::Borrowed(mask)),
                storage: MaskStorage::Explicit,
                derived: false,
                payload: Vec::new(),
            },
            MaskKind::External(mask) => Self {
                data: Some(Cow::Borrowed(mask)),
                storage: MaskStorage::External,
                derived: false,
                payload: Vec::new(),
            },
        }
    }

    pub(super) fn data(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    pub(super) fn is_derived(&self) -> bool {
        self.derived
    }

    pub(super) fn derive(&mut self, pixel_count: usize) -> Result<&mut [u8]> {
        if !self.derived {
            let owned = self
                .data
                .as_deref()
                .map(<[u8]>::to_vec)
                .unwrap_or_else(|| vec![1; pixel_count]);
            self.data = Some(Cow::Owned(owned));
            self.storage = MaskStorage::Explicit;
            self.derived = true;
        }
        self.data
            .as_mut()
            .map(|data| Cow::to_mut(data).as_mut_slice())
            .ok_or(Error::Internal("derived mask has no data"))
    }

    pub(super) fn make_explicit(&mut self, pixel_count: usize) {
        if self.data.is_none() {
            self.data = Some(Cow::Owned(vec![1; pixel_count]));
        }
        self.storage = MaskStorage::Explicit;
    }

    pub(super) fn prepare_payload(
        &mut self,
        pixel_count: usize,
        valid_pixel_count: usize,
    ) -> Result<()> {
        let kind = match self.storage {
            MaskStorage::None => MaskKind::None,
            MaskStorage::Explicit => MaskKind::Explicit(
                self.data
                    .as_deref()
                    .ok_or(Error::Internal("explicit mask has no data"))?,
            ),
            MaskStorage::External => MaskKind::External(
                self.data
                    .as_deref()
                    .ok_or(Error::Internal("external mask has no data"))?,
            ),
        };
        self.payload = encode_mask_rle(kind, pixel_count, valid_pixel_count)?;
        Ok(())
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.payload
    }
}

pub(super) fn shared_mask_for_band(mask: Option<&[u8]>, band_index: usize) -> MaskKind<'_> {
    match mask {
        Some(mask) if band_index == 0 => MaskKind::Explicit(mask),
        Some(mask) => MaskKind::External(mask),
        None => MaskKind::None,
    }
}

pub(super) fn validate_dimensions(
    width: u32,
    height: u32,
    mask: Option<MaskView<'_>>,
) -> Result<()> {
    if mask.is_some_and(|mask| mask.width() != width || mask.height() != height) {
        return Err(Error::InvalidArgument(
            "mask dimensions must match the raster dimensions",
        ));
    }
    Ok(())
}

pub(super) fn validate_slice(mask: Option<&[u8]>, pixel_count: usize) -> Result<()> {
    if mask.is_some_and(|mask| mask.len() != pixel_count) {
        return Err(Error::InvalidArgument(
            "mask length does not match the raster dimensions",
        ));
    }
    Ok(())
}

pub(super) fn pixel_is_valid(mask: Option<&[u8]>, pixel: usize) -> bool {
    match mask {
        Some(mask) => mask[pixel] != 0,
        None => true,
    }
}

fn encode_mask_rle(
    mask: MaskKind<'_>,
    pixel_count: usize,
    valid_pixel_count: usize,
) -> Result<Vec<u8>> {
    if valid_pixel_count == 0 || valid_pixel_count == pixel_count {
        return Ok(Vec::new());
    }
    let MaskKind::Explicit(mask) = mask else {
        return Ok(Vec::new());
    };

    let bitset = pack_mask_bitset(mask, pixel_count)?;
    let mut out = Vec::new();
    emit_mask_rle_segments(&bitset, |segment| {
        match segment {
            MaskRleSegment::Literal(bytes) => {
                out.extend_from_slice(&(bytes.len() as i16).to_le_bytes());
                out.extend_from_slice(bytes);
            }
            MaskRleSegment::Repeat { value, count } => {
                out.extend_from_slice(&(-(count as i16)).to_le_bytes());
                out.push(value);
            }
        }
        Ok(())
    })?;
    out.extend_from_slice(&i16::MIN.to_le_bytes());
    Ok(out)
}

fn pack_mask_bitset(mask: &[u8], pixel_count: usize) -> Result<Vec<u8>> {
    validate_slice(Some(mask), pixel_count)?;
    let mut bitset = vec![0u8; pixel_count.div_ceil(8)];
    for (index, &value) in mask.iter().enumerate() {
        if value != 0 {
            bitset[index >> 3] |= 1 << (7 - (index & 7));
        }
    }
    Ok(bitset)
}

fn emit_mask_literal_chunks<F>(bytes: &[u8], emit_literal: &mut F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    let mut offset = 0usize;
    while offset < bytes.len() {
        let chunk = (bytes.len() - offset).min(i16::MAX as usize);
        emit_literal(&bytes[offset..offset + chunk])?;
        offset += chunk;
    }
    Ok(())
}

enum MaskRleSegment<'a> {
    Literal(&'a [u8]),
    Repeat { value: u8, count: usize },
}

fn emit_mask_rle_segments<F>(bitset: &[u8], mut emit: F) -> Result<()>
where
    F: FnMut(MaskRleSegment<'_>) -> Result<()>,
{
    let mut literal_start = 0usize;
    let mut offset = 0usize;

    while offset < bitset.len() {
        let value = bitset[offset];
        let mut run_end = offset + 1;
        while run_end < bitset.len() && bitset[run_end] == value {
            run_end += 1;
        }

        let run_len = run_end - offset;
        let repeat_threshold = if literal_start == offset && run_end == bitset.len() {
            2
        } else if literal_start == offset || run_end == bitset.len() {
            4
        } else {
            6
        };

        if run_len >= repeat_threshold {
            emit_mask_literal_chunks(&bitset[literal_start..offset], &mut |bytes| {
                emit(MaskRleSegment::Literal(bytes))
            })?;
            let mut emitted = 0usize;
            while emitted < run_len {
                let chunk = (run_len - emitted).min(i16::MAX as usize);
                emit(MaskRleSegment::Repeat {
                    value,
                    count: chunk,
                })?;
                emitted += chunk;
            }
            literal_start = run_end;
        }
        offset = run_end;
    }

    emit_mask_literal_chunks(&bitset[literal_start..], &mut |bytes| {
        emit(MaskRleSegment::Literal(bytes))
    })
}
