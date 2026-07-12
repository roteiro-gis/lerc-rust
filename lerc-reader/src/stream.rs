use std::io::{ErrorKind, Read};

use lerc_core::{Error, Result};

use crate::{lerc1, lerc2};

const MAGIC_LEN: usize = 10;
const LERC1_PREFIX_LEN: usize = 34;
const LERC1_SECTION_HEADER_LEN: usize = 16;
const MAX_STREAM_BLOB_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn read_next_blob<R: Read + ?Sized>(
    reader: &mut R,
    lerc1_shared_mask: Option<&[u8]>,
) -> Result<Option<Vec<u8>>> {
    let Some(mut blob) = read_magic(reader)? else {
        return Ok(None);
    };

    if lerc2::is_lerc2(&blob) {
        let version = read_i32_at(&blob, 6)?;
        if !(1..=6).contains(&version) {
            return Err(Error::UnsupportedVersion(version.max(0) as u32));
        }
        let blob_size_offset = 26 + usize::from(version >= 3) * 4 + usize::from(version >= 4) * 4;
        let prefix_len = blob_size_offset + 4;
        read_exact_to(reader, &mut blob, prefix_len)?;
        let declared_size = read_i32_at(&blob, blob_size_offset)?;
        if declared_size <= 0 {
            return Err(Error::InvalidHeader("non-positive blob size"));
        }
        let declared_size = declared_size as usize;
        validate_stream_size(declared_size)?;
        if declared_size < prefix_len {
            return Err(Error::InvalidHeader(
                "blob size is smaller than the fixed Lerc2 prefix",
            ));
        }
        read_exact_to(reader, &mut blob, declared_size)?;
        return Ok(Some(blob));
    }

    if !lerc1::is_lerc1(&blob) {
        return Err(Error::InvalidMagic);
    }

    read_exact_to(
        reader,
        &mut blob,
        LERC1_PREFIX_LEN + LERC1_SECTION_HEADER_LEN,
    )?;
    let first_payload_len = read_u32_at(&blob, LERC1_PREFIX_LEN + 8)? as usize;
    let first_end = checked_stream_end(blob.len(), first_payload_len)?;
    read_exact_to(reader, &mut blob, first_end)?;

    let inline_result = lerc1::parse(&blob, None);
    if let Ok(parsed) = inline_result.as_ref() {
        ensure_lerc1_declared_size(parsed.info.blob_size, blob.len())?;
        return Ok(Some(blob));
    }

    let external_result = lerc1_shared_mask.map(|mask| lerc1::parse(&blob, Some(mask)));
    if let Some(Ok(parsed)) = external_result.as_ref() {
        ensure_lerc1_declared_size(parsed.info.blob_size, blob.len())?;
        return Ok(Some(blob));
    }

    let needs_pixel_section = matches!(inline_result, Err(Error::Truncated { .. }))
        || matches!(external_result, Some(Err(Error::Truncated { .. })));
    if !needs_pixel_section {
        return match inline_result {
            Err(error) => Err(error),
            Ok(_) => Err(Error::Internal(
                "successful inline Lerc1 stream parse was not returned",
            )),
        };
    }

    let pixel_header_end = checked_stream_end(blob.len(), LERC1_SECTION_HEADER_LEN)?;
    read_exact_to(reader, &mut blob, pixel_header_end)?;
    let pixel_payload_len = read_u32_at(&blob, pixel_header_end - 8)? as usize;
    let declared_end = checked_stream_end(blob.len(), pixel_payload_len)?;
    read_exact_to(reader, &mut blob, declared_end)?;

    let parsed = lerc1::parse(&blob, lerc1_shared_mask)?;
    ensure_lerc1_declared_size(parsed.info.blob_size, declared_end)?;
    Ok(Some(blob))
}

fn read_magic<R: Read + ?Sized>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let mut blob = vec![0u8; MAGIC_LEN];
    let mut filled = 0usize;
    while filled < MAGIC_LEN {
        match reader.read(&mut blob[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => {
                return Err(Error::Truncated {
                    offset: filled,
                    needed: MAGIC_LEN - filled,
                    available: 0,
                })
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(Error::io("reading the LERC magic", error)),
        }
    }
    Ok(Some(blob))
}

fn read_exact_to<R: Read + ?Sized>(
    reader: &mut R,
    blob: &mut Vec<u8>,
    target_len: usize,
) -> Result<()> {
    validate_stream_size(target_len)?;
    if target_len < blob.len() {
        return Err(Error::Internal("stream target moved behind buffered input"));
    }
    blob.try_reserve(target_len - blob.len())
        .map_err(|_| Error::invalid_blob("unable to reserve the declared stream blob size"))?;
    let mut filled = blob.len();
    blob.resize(target_len, 0);
    while filled < target_len {
        match reader.read(&mut blob[filled..]) {
            Ok(0) => {
                blob.truncate(filled);
                return Err(Error::Truncated {
                    offset: filled,
                    needed: target_len - filled,
                    available: 0,
                });
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                blob.truncate(filled);
                return Err(Error::io("reading a LERC blob", error));
            }
        }
    }
    Ok(())
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or(Error::SizeOverflow("stream header field end"))?;
    let field = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invalid_blob("stream header field is outside the buffered prefix"))?;
    let field = <[u8; 4]>::try_from(field)
        .map_err(|_| Error::Internal("stream header field has the wrong width"))?;
    Ok(u32::from_le_bytes(field))
}

fn read_i32_at(bytes: &[u8], offset: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        read_u32_at(bytes, offset)?.to_le_bytes(),
    ))
}

fn checked_stream_end(offset: usize, len: usize) -> Result<usize> {
    let end = offset
        .checked_add(len)
        .ok_or(Error::SizeOverflow("stream blob length"))?;
    validate_stream_size(end)?;
    Ok(end)
}

fn validate_stream_size(size: usize) -> Result<()> {
    if size > MAX_STREAM_BLOB_BYTES {
        return Err(Error::invalid_blob(
            "declared stream blob size exceeds the decoder limit",
        ));
    }
    Ok(())
}

fn ensure_lerc1_declared_size(parsed_size: usize, declared_size: usize) -> Result<()> {
    if parsed_size != declared_size {
        return Err(Error::invalid_blob(
            "Lerc1 section length does not match the parsed blob length",
        ));
    }
    Ok(())
}
