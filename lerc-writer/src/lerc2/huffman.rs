use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lerc_core::{DataType, Error, Result, Sample};

use super::{bitstuff::write_raw_block, pixel_is_valid, ByteSink, RasterAnalysis, RasterSource};

#[derive(Debug, Clone)]
pub(super) struct HuffmanPlan {
    mode: HuffmanMode,
    alphabet: HuffmanAlphabet,
    table_bytes: Vec<u8>,
    codes: Vec<Option<HuffmanCode>>,
    pub(super) data_len: usize,
}

#[derive(Debug, Clone)]
pub(super) struct HuffmanHistograms {
    alphabet: HuffmanAlphabet,
    plain: [u64; 256],
    delta: [u64; 256],
}

pub(super) struct HistogramBuilder {
    alphabet: HuffmanAlphabet,
    width: usize,
    depth: usize,
    previous_values: Vec<i32>,
    previous_row: Vec<i32>,
    plain: [u64; 256],
    delta: [u64; 256],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HuffmanMode {
    Delta = 1,
    Plain = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HuffmanAlphabet {
    Signed,
    Unsigned,
}

impl HuffmanAlphabet {
    fn for_data_type(data_type: DataType) -> Option<Self> {
        match data_type {
            DataType::I8 => Some(Self::Signed),
            DataType::U8 => Some(Self::Unsigned),
            _ => None,
        }
    }

    fn sample_value(self, value: f64) -> i32 {
        match self {
            Self::Signed => value as i8 as i32,
            Self::Unsigned => value as u8 as i32,
        }
    }

    fn symbol(self, value: i32) -> usize {
        match self {
            Self::Signed => (value + 128) as usize,
            Self::Unsigned => value as usize,
        }
    }

    fn delta_symbol(self, delta: i32) -> usize {
        match self {
            Self::Signed => ((delta + 128) & 0xFF) as usize,
            Self::Unsigned => ((delta & 0xFF) as u8) as usize,
        }
    }
}

impl HistogramBuilder {
    pub(super) fn new(data_type: DataType, width: usize, depth: usize) -> Result<Option<Self>> {
        let Some(alphabet) = HuffmanAlphabet::for_data_type(data_type) else {
            return Ok(None);
        };
        let row_len = width
            .checked_mul(depth)
            .ok_or(Error::SizeOverflow("Huffman predictor row length"))?;
        Ok(Some(Self {
            alphabet,
            width,
            depth,
            previous_values: vec![0; depth],
            previous_row: vec![0; row_len],
            plain: [0; 256],
            delta: [0; 256],
        }))
    }

    pub(super) fn observe(
        &mut self,
        pixel: usize,
        mask: Option<&[u8]>,
        values: &[f64],
    ) -> Result<()> {
        if values.len() != self.depth {
            return Err(Error::Internal(
                "Huffman histogram sample depth changed during analysis",
            ));
        }
        let col = pixel % self.width;
        let row = pixel / self.width;
        let row_base = col
            .checked_mul(self.depth)
            .ok_or(Error::SizeOverflow("Huffman predictor row offset"))?;
        for (dim, &sample) in values.iter().enumerate() {
            let value = self.alphabet.sample_value(sample);
            let symbol = self.alphabet.symbol(value);
            self.plain[symbol] = self.plain[symbol]
                .checked_add(1)
                .ok_or(Error::SizeOverflow("Huffman frequency count"))?;

            let predictor = if col > 0 && pixel_is_valid(mask, pixel - 1) {
                self.previous_values[dim]
            } else if row > 0 && pixel_is_valid(mask, pixel - self.width) {
                self.previous_row[row_base + dim]
            } else {
                self.previous_values[dim]
            };
            let symbol = self.alphabet.delta_symbol(value - predictor);
            self.delta[symbol] = self.delta[symbol]
                .checked_add(1)
                .ok_or(Error::SizeOverflow("Huffman frequency count"))?;
            self.previous_values[dim] = value;
            self.previous_row[row_base + dim] = value;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> HuffmanHistograms {
        HuffmanHistograms {
            alphabet: self.alphabet,
            plain: self.plain,
            delta: self.delta,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuffmanCode {
    bit_len: u8,
    bits: u32,
}

#[derive(Debug, Clone)]
enum HuffmanNodeKind {
    Leaf(u16),
    Branch { left: usize, right: usize },
}

#[derive(Debug, Clone)]
struct HuffmanNode {
    kind: HuffmanNodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct HuffmanHeapEntry {
    freq: u64,
    min_symbol: u16,
    node_index: usize,
}

#[derive(Debug, Default)]
struct MsbBitWriter {
    words: Vec<u32>,
    current: u32,
    bits_used: u8,
}

impl MsbBitWriter {
    fn push_bits(&mut self, bits: u32, bit_len: u8) {
        let mut remaining = bit_len as usize;
        while remaining > 0 {
            let space = 32usize - self.bits_used as usize;
            let take = remaining.min(space);
            let shift_out = remaining - take;
            let chunk_mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            let chunk = (bits >> shift_out) & chunk_mask;
            self.current |= chunk << (space - take);
            self.bits_used += take as u8;
            remaining -= take;
            if self.bits_used == 32 {
                self.words.push(self.current);
                self.current = 0;
                self.bits_used = 0;
            }
        }
    }

    fn into_aligned_bytes(mut self) -> Vec<u8> {
        if self.bits_used != 0 {
            self.words.push(self.current);
        }
        words_to_le_bytes(&self.words)
    }

    fn into_bytes_with_trailing_word(mut self) -> Vec<u8> {
        if self.bits_used != 0 {
            self.words.push(self.current);
        }
        self.words.push(0);
        words_to_le_bytes(&self.words)
    }
}

pub(super) fn supports(data_type: DataType, max_z_error: f64) -> bool {
    HuffmanAlphabet::for_data_type(data_type).is_some() && (max_z_error - 0.5).abs() < 1e-5
}

pub(super) fn build_plan(analysis: &RasterAnalysis) -> Result<Option<HuffmanPlan>> {
    if !supports(analysis.data_type, analysis.max_z_error) {
        return Ok(None);
    }
    let Some(histograms) = analysis.huffman_histograms.as_ref() else {
        return Ok(None);
    };
    let plain = build_candidate_from_hist(&histograms.plain)?;
    let delta = build_candidate_from_hist(&histograms.delta)?;
    let selected = match (plain, delta) {
        (Some(plain), Some(delta)) if plain.data_len <= delta.data_len => {
            Some((HuffmanMode::Plain, plain))
        }
        (Some(_), Some(delta)) => Some((HuffmanMode::Delta, delta)),
        (Some(plain), None) => Some((HuffmanMode::Plain, plain)),
        (None, Some(delta)) => Some((HuffmanMode::Delta, delta)),
        (None, None) => None,
    };

    Ok(selected.map(|(mode, candidate)| HuffmanPlan {
        mode,
        alphabet: histograms.alphabet,
        table_bytes: candidate.table_bytes,
        codes: candidate.codes,
        data_len: candidate.data_len,
    }))
}

pub(super) fn write_body<T: Sample, R: RasterSource<T>>(
    sink: &mut impl ByteSink,
    raster: R,
    mask: Option<&[u8]>,
    plan: &HuffmanPlan,
) -> Result<()> {
    sink.push(0)?;
    sink.push(plan.mode as u8)?;
    sink.extend_from_slice(&plan.table_bytes)?;

    let width = raster.width() as usize;
    let height = raster.height() as usize;
    let depth = raster.depth() as usize;
    let mut writer = MsbBitWriter::default();

    match plan.mode {
        HuffmanMode::Delta => {
            for dim in 0..depth {
                let mut prev_value = 0i32;
                for row in 0..height {
                    for col in 0..width {
                        let pixel = row * width + col;
                        if !pixel_is_valid(mask, pixel) {
                            continue;
                        }
                        let value = plan
                            .alphabet
                            .sample_value(raster.sample(pixel, dim).to_f64());
                        let predictor = if col > 0 && pixel_is_valid(mask, pixel - 1) {
                            prev_value
                        } else if row > 0 && pixel_is_valid(mask, pixel - width) {
                            plan.alphabet
                                .sample_value(raster.sample(pixel - width, dim).to_f64())
                        } else {
                            prev_value
                        };
                        let symbol = plan.alphabet.delta_symbol(value - predictor);
                        let code = plan.codes[symbol]
                            .ok_or(Error::Internal("missing Huffman delta symbol"))?;
                        writer.push_bits(code.bits, code.bit_len);
                        prev_value = value;
                    }
                }
            }
        }
        HuffmanMode::Plain => {
            let pixel_count = raster.pixel_count()?;
            for pixel in 0..pixel_count {
                if !pixel_is_valid(mask, pixel) {
                    continue;
                }
                for dim in 0..depth {
                    let value = plan
                        .alphabet
                        .sample_value(raster.sample(pixel, dim).to_f64());
                    let symbol = plan.alphabet.symbol(value);
                    let code =
                        plan.codes[symbol].ok_or(Error::Internal("missing Huffman symbol"))?;
                    writer.push_bits(code.bits, code.bit_len);
                }
            }
        }
    }

    sink.extend_from_slice(&writer.into_bytes_with_trailing_word())
}

struct HuffmanCandidate {
    table_bytes: Vec<u8>,
    codes: Vec<Option<HuffmanCode>>,
    data_len: usize,
}

fn build_candidate_from_hist(hist: &[u64; 256]) -> Result<Option<HuffmanCandidate>> {
    let Some(codes) = build_codes(hist)? else {
        return Ok(None);
    };
    let table_bytes = build_table_bytes(&codes)?;
    let payload_bits = hist
        .iter()
        .zip(&codes)
        .try_fold(0usize, |acc, (&count, code)| {
            let Some(code) = code else {
                return Ok(acc);
            };
            let count = usize::try_from(count)
                .map_err(|_| Error::SizeOverflow("Huffman symbol frequency as usize"))?;
            let symbol_bits = count
                .checked_mul(code.bit_len as usize)
                .ok_or(Error::SizeOverflow("Huffman symbol payload bit count"))?;
            acc.checked_add(symbol_bits)
                .ok_or(Error::SizeOverflow("Huffman payload bit count"))
        })?;
    let payload_data_bytes = payload_bits
        .div_ceil(32)
        .checked_add(1)
        .and_then(|words| words.checked_mul(4))
        .ok_or(Error::SizeOverflow("Huffman payload byte count"))?;
    let payload_bytes = payload_data_bytes
        .checked_add(table_bytes.len())
        .ok_or(Error::SizeOverflow("Huffman payload byte count"))?;

    Ok(Some(HuffmanCandidate {
        table_bytes,
        codes,
        data_len: payload_bytes,
    }))
}

fn build_codes(hist: &[u64; 256]) -> Result<Option<Vec<Option<HuffmanCode>>>> {
    let mut nodes = Vec::<HuffmanNode>::new();
    let mut heap = BinaryHeap::new();

    for (symbol, &freq) in hist.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            kind: HuffmanNodeKind::Leaf(symbol as u16),
        });
        heap.push(Reverse(HuffmanHeapEntry {
            freq,
            min_symbol: symbol as u16,
            node_index,
        }));
    }
    if heap.is_empty() {
        return Ok(None);
    }

    while heap.len() > 1 {
        let Reverse(left) = heap
            .pop()
            .ok_or(Error::Internal("Huffman heap lost its left node"))?;
        let Reverse(right) = heap
            .pop()
            .ok_or(Error::Internal("Huffman heap lost its right node"))?;
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            kind: HuffmanNodeKind::Branch {
                left: left.node_index,
                right: right.node_index,
            },
        });
        heap.push(Reverse(HuffmanHeapEntry {
            freq: left
                .freq
                .checked_add(right.freq)
                .ok_or(Error::SizeOverflow("Huffman frequency count"))?,
            min_symbol: left.min_symbol.min(right.min_symbol),
            node_index,
        }));
    }

    let root = heap
        .pop()
        .ok_or(Error::Internal("Huffman heap lost its root node"))?
        .0
        .node_index;
    let mut codes = vec![None; 256];
    if assign_codes(&nodes, root, 0, 0, &mut codes).is_err() {
        return Ok(None);
    }
    Ok(Some(codes))
}

fn assign_codes(
    nodes: &[HuffmanNode],
    node_index: usize,
    bits: u32,
    bit_len: u8,
    codes: &mut [Option<HuffmanCode>],
) -> Result<()> {
    let node = nodes
        .get(node_index)
        .ok_or(Error::Internal("Huffman node index is out of bounds"))?;
    match node.kind {
        HuffmanNodeKind::Leaf(symbol) => {
            let bit_len = bit_len.max(1);
            if bit_len > 31 {
                return Err(Error::Internal("Huffman code length exceeds Lerc2 limits"));
            }
            codes[symbol as usize] = Some(HuffmanCode { bit_len, bits });
        }
        HuffmanNodeKind::Branch { left, right } => {
            if bit_len >= 31 {
                return Err(Error::Internal("Huffman code length exceeds Lerc2 limits"));
            }
            assign_codes(nodes, left, bits << 1, bit_len + 1, codes)?;
            assign_codes(nodes, right, (bits << 1) | 1, bit_len + 1, codes)?;
        }
    }
    Ok(())
}

fn build_table_bytes(codes: &[Option<HuffmanCode>]) -> Result<Vec<u8>> {
    let first_symbol = codes
        .iter()
        .position(Option::is_some)
        .ok_or(Error::Internal("Huffman code table cannot be empty"))?;
    let last_symbol = codes
        .iter()
        .rposition(Option::is_some)
        .ok_or(Error::Internal("Huffman code table cannot be empty"))?
        + 1;
    let code_lengths: Vec<u32> = codes[first_symbol..last_symbol]
        .iter()
        .map(|code| code.map_or(0, |code| code.bit_len as u32))
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&(codes.len() as i32).to_le_bytes());
    out.extend_from_slice(&(first_symbol as i32).to_le_bytes());
    out.extend_from_slice(&(last_symbol as i32).to_le_bytes());
    write_raw_block(&mut out, &code_lengths)?;

    let mut writer = MsbBitWriter::default();
    for code in codes[first_symbol..last_symbol].iter().flatten() {
        writer.push_bits(code.bits, code.bit_len);
    }
    out.extend_from_slice(&writer.into_aligned_bytes());
    Ok(out)
}

fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}
