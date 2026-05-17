use std::cmp;

use crate::allocation::default_vec;
use crate::bitstuff::decode_bits;
use crate::io::Cursor;
use crate::pixel::{output_value, sample_index, words_from_padded, Sample};
use lerc_band_materialize::{BandWriteOrder, BandWriter};
use lerc_core::{BlobInfo, Error, PixelData, Result, Version};

const HUFFMAN_LUT_BITS_MAX: u8 = 12;
const HUFFMAN_SYMBOL_COUNT: usize = 256;
const HUFFMAN_CODE_BITS_MAX: u8 = 32;

#[derive(Debug, Clone)]
struct HuffmanEntry {
    bit_len: u8,
    value: u32,
}

#[derive(Debug, Default)]
struct HuffmanNode {
    value: Option<u32>,
    left: Option<Box<HuffmanNode>>,
    right: Option<Box<HuffmanNode>>,
}

#[derive(Debug)]
struct HuffmanInfo {
    quick_lut: Vec<Option<HuffmanEntry>>,
    quick_bits: u8,
    max_bits: u8,
    tree: HuffmanNode,
    stuffed_data: Vec<u32>,
    src_ptr: usize,
    bit_pos: u8,
}

#[derive(Debug)]
struct HuffmanStream<'a> {
    words: &'a [u32],
    src_ptr: usize,
    bit_pos: u8,
}

pub(crate) fn decode_huffman<T: Sample>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    mask: Option<&[u8]>,
    delta_encode: bool,
) -> Result<PixelData> {
    let width = info.width as usize;
    let height = info.height as usize;
    let depth = info.depth as usize;
    let sample_count = info.sample_count()?;
    let mut result: Vec<T> = default_vec(sample_count, "Lerc2 Huffman output")?;

    let huffman = read_huffman_tree(cursor, info)?;
    let mut stream = HuffmanStream {
        words: &huffman.stuffed_data,
        src_ptr: huffman.src_ptr,
        bit_pos: huffman.bit_pos,
    };
    if stream.bit_pos > 0 {
        stream.src_ptr += 1;
        stream.bit_pos = 0;
    }

    let offset = if info.data_type == lerc_core::DataType::I8 {
        128.0
    } else {
        0.0
    };

    if depth < 2 || delta_encode {
        for dim in 0..depth {
            let mut prev_value = 0.0;
            for row in 0..height {
                for col in 0..width {
                    let pixel = row * width + col;
                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                        let value = read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                        let decoded = if delta_encode {
                            let mut delta = value;
                            if col > 0 && mask.map(|m| m[pixel - 1] != 0).unwrap_or(true) {
                                delta += prev_value;
                            } else if row > 0 && mask.map(|m| m[pixel - width] != 0).unwrap_or(true)
                            {
                                delta += result[sample_index(pixel - width, depth, dim)].to_f64();
                            } else {
                                delta += prev_value;
                            }
                            let wrapped = ((delta as i64) & 0xFF) as f64;
                            prev_value = wrapped;
                            wrapped
                        } else {
                            value
                        };
                        result[sample_index(pixel, depth, dim)] =
                            output_value::<T>(decoded, info.data_type);
                    }
                }
            }
        }
    } else {
        for row in 0..height {
            for col in 0..width {
                let pixel = row * width + col;
                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                    for dim in 0..depth {
                        let value = read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                        result[sample_index(pixel, depth, dim)] =
                            output_value::<T>(value, info.data_type);
                    }
                }
            }
        }
    }

    Ok(T::into_pixel_data(result))
}

pub(crate) fn decode_huffman_into<T: Sample, W: BandWriter<T>>(
    cursor: &mut Cursor<'_>,
    info: &BlobInfo,
    mask: Option<&[u8]>,
    delta_encode: bool,
    out: &mut W,
) -> Result<()> {
    let width = info.width as usize;
    let height = info.height as usize;
    let depth = info.depth as usize;

    let huffman = read_huffman_tree(cursor, info)?;
    let mut stream = HuffmanStream {
        words: &huffman.stuffed_data,
        src_ptr: huffman.src_ptr,
        bit_pos: huffman.bit_pos,
    };
    if stream.bit_pos > 0 {
        stream.src_ptr += 1;
        stream.bit_pos = 0;
    }

    let offset = if info.data_type == lerc_core::DataType::I8 {
        128.0
    } else {
        0.0
    };

    if depth < 2 || delta_encode {
        if depth > 1 && delta_encode {
            out.set_write_order(BandWriteOrder::DimMajor);
        }
        for dim in 0..depth {
            let mut prev_value = 0.0;
            for row in 0..height {
                for col in 0..width {
                    let pixel = row * width + col;
                    if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                        let value = read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                        let decoded = if delta_encode {
                            let mut delta = value;
                            if col > 0 && mask.map(|m| m[pixel - 1] != 0).unwrap_or(true) {
                                delta += prev_value;
                            } else if row > 0 && mask.map(|m| m[pixel - width] != 0).unwrap_or(true)
                            {
                                delta += out.read(pixel - width, dim).to_f64();
                            } else {
                                delta += prev_value;
                            }
                            let wrapped = ((delta as i64) & 0xFF) as f64;
                            prev_value = wrapped;
                            wrapped
                        } else {
                            value
                        };
                        out.write(pixel, dim, output_value::<T>(decoded, info.data_type));
                    }
                }
            }
        }
    } else {
        for row in 0..height {
            for col in 0..width {
                let pixel = row * width + col;
                if mask.map(|m| m[pixel] != 0).unwrap_or(true) {
                    for dim in 0..depth {
                        let value = read_huffman_symbol(&mut stream, &huffman)? as f64 - offset;
                        out.write(pixel, dim, output_value::<T>(value, info.data_type));
                    }
                }
            }
        }
    }

    Ok(())
}

fn read_huffman_tree(cursor: &mut Cursor<'_>, info: &BlobInfo) -> Result<HuffmanInfo> {
    let version = read_i32(cursor.read_bytes(4)?)?;
    if version < 2 {
        return Err(Error::UnsupportedFeature("Huffman version < 2"));
    }
    let size = read_i32(cursor.read_bytes(4)?)?;
    let i0 = read_i32(cursor.read_bytes(4)?)?;
    let i1 = read_i32(cursor.read_bytes(4)?)?;
    if size <= 0 || i0 < 0 || i1 <= i0 {
        return Err(Error::InvalidBlob("invalid Huffman table header".into()));
    }

    let size = usize::try_from(size)
        .map_err(|_| Error::InvalidBlob("invalid Huffman table header".into()))?;
    let i0 = usize::try_from(i0)
        .map_err(|_| Error::InvalidBlob("invalid Huffman table header".into()))?;
    let i1 = usize::try_from(i1)
        .map_err(|_| Error::InvalidBlob("invalid Huffman table header".into()))?;
    let code_length_count = i1
        .checked_sub(i0)
        .ok_or_else(|| Error::InvalidBlob("invalid Huffman table header".into()))?;

    if size == 0
        || size > HUFFMAN_SYMBOL_COUNT
        || i0 >= size
        || code_length_count == 0
        || code_length_count > size
    {
        return Err(Error::InvalidBlob("invalid Huffman table header".into()));
    }

    let mut code_lengths = vec![0.0; code_length_count];
    let _ = decode_bits(
        cursor,
        match info.version {
            Version::Lerc2(version) => version,
            _ => unreachable!("Lerc2 Huffman decode called with a non-Lerc2 blob"),
        },
        &mut code_lengths,
        None,
        info.max_z_error,
        info.z_max,
    )?;

    let mut code_table: Vec<Option<(u8, u32)>> = vec![None; size];
    for i in i0..i1 {
        let j = if i < size { i } else { i - size };
        let bit_len = code_lengths[i - i0];
        if !bit_len.is_finite()
            || bit_len.fract() != 0.0
            || bit_len < 0.0
            || bit_len > f64::from(HUFFMAN_CODE_BITS_MAX)
        {
            return Err(Error::InvalidBlob("invalid Huffman code length".into()));
        }
        code_table[j] = Some((bit_len as u8, 0));
    }

    let stuffed_data = words_from_padded(cursor.read_bytes(cursor.remaining())?);
    let mut stream = HuffmanStream {
        words: &stuffed_data,
        src_ptr: 0,
        bit_pos: 0,
    };

    for entry in code_table.iter_mut().flatten() {
        if entry.0 > 0 {
            entry.1 = stream.peek_bits(entry.0 as usize)?;
            stream.advance(entry.0 as usize)?;
        }
    }

    let mut max_bits = 0u8;
    for (bit_len, _) in code_table.iter().flatten() {
        max_bits = max_bits.max(*bit_len);
    }
    let quick_bits = cmp::min(max_bits, HUFFMAN_LUT_BITS_MAX);
    let mut quick_lut = vec![None; 1usize << quick_bits];
    let mut tree = HuffmanNode::default();

    for (symbol, entry) in code_table.into_iter().enumerate() {
        let Some((bit_len, code)) = entry else {
            continue;
        };
        if bit_len == 0 {
            continue;
        }
        if bit_len <= quick_bits {
            let base = code << (quick_bits - bit_len);
            let num_entries = 1usize << (quick_bits - bit_len);
            for extra in 0..num_entries {
                quick_lut[(base as usize) | extra] = Some(HuffmanEntry {
                    bit_len,
                    value: symbol as u32,
                });
            }
        } else {
            insert_huffman_code(&mut tree, code, bit_len, symbol as u32)?;
        }
    }

    let src_ptr = stream.src_ptr;
    let bit_pos = stream.bit_pos;

    Ok(HuffmanInfo {
        quick_lut,
        quick_bits,
        max_bits,
        tree,
        stuffed_data,
        src_ptr,
        bit_pos,
    })
}

fn insert_huffman_code(root: &mut HuffmanNode, code: u32, bit_len: u8, value: u32) -> Result<()> {
    let mut node = root;
    for shift in (0..bit_len).rev() {
        let bit = (code >> shift) & 1;
        if shift == 0 {
            let child = if bit == 0 {
                node.left
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            } else {
                node.right
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            };
            child.value = Some(value);
        } else {
            node = if bit == 0 {
                node.left
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            } else {
                node.right
                    .get_or_insert_with(|| Box::new(HuffmanNode::default()))
            };
        }
    }
    Ok(())
}

fn read_huffman_symbol(stream: &mut HuffmanStream<'_>, info: &HuffmanInfo) -> Result<u32> {
    if let Some(entry) =
        info.quick_lut[stream.peek_bits(info.quick_bits as usize)? as usize].as_ref()
    {
        stream.advance(entry.bit_len as usize)?;
        return Ok(entry.value);
    }

    let value = stream.peek_bits(info.max_bits as usize)?;
    let mut node = &info.tree;
    for used_bits in 0..info.max_bits {
        let bit = (value >> (info.max_bits - used_bits - 1)) & 1;
        node = if bit == 0 {
            node.left.as_deref().ok_or_else(|| {
                Error::InvalidBlob("Huffman decode walked a missing left node".into())
            })?
        } else {
            node.right.as_deref().ok_or_else(|| {
                Error::InvalidBlob("Huffman decode walked a missing right node".into())
            })?
        };

        if node.left.is_none() && node.right.is_none() {
            let symbol = node
                .value
                .ok_or_else(|| Error::InvalidBlob("Huffman leaf is missing a symbol".into()))?;
            stream.advance((used_bits + 1) as usize)?;
            return Ok(symbol);
        }
    }

    Err(Error::InvalidBlob(
        "Huffman symbol exceeded the configured code width".into(),
    ))
}

impl<'a> HuffmanStream<'a> {
    fn peek_bits(&self, num_bits: usize) -> Result<u32> {
        if num_bits == 0 {
            return Ok(0);
        }
        if num_bits > usize::from(HUFFMAN_CODE_BITS_MAX) {
            return Err(Error::InvalidBlob("Huffman bit width exceeds u32".into()));
        }
        if self.src_ptr >= self.words.len() {
            return Err(Error::Truncated {
                offset: self.src_ptr * 4,
                needed: 4,
                available: 0,
            });
        }
        let word = self.words[self.src_ptr];
        let mut value = word.wrapping_shl(self.bit_pos as u32) >> (32 - num_bits as u32);
        if 32 - (self.bit_pos as usize) < num_bits {
            let next = self.words.get(self.src_ptr + 1).copied().unwrap_or(0);
            value |= next >> (64 - self.bit_pos as usize - num_bits);
        }
        Ok(value)
    }

    fn advance(&mut self, bits: usize) -> Result<()> {
        let total = self.bit_pos as usize + bits;
        self.src_ptr = self
            .src_ptr
            .checked_add(total / 32)
            .ok_or_else(|| Error::InvalidBlob("Huffman bitstream pointer overflow".into()))?;
        self.bit_pos = (total % 32) as u8;
        Ok(())
    }
}

fn read_i32(bytes: &[u8]) -> Result<i32> {
    Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
}
