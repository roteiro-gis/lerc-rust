use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lerc_core::{DataType, Error, Result, Sample};

use super::options::EncoderProfile;
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
    profile: EncoderProfile,
    plain: [u64; 256],
    delta: [u64; 256],
}

pub(super) struct HistogramBuilder {
    alphabet: HuffmanAlphabet,
    profile: EncoderProfile,
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
    pub(super) fn new(
        data_type: DataType,
        width: usize,
        depth: usize,
        profile: EncoderProfile,
    ) -> Result<Option<Self>> {
        let Some(alphabet) = HuffmanAlphabet::for_data_type(data_type) else {
            return Ok(None);
        };
        let row_len = width
            .checked_mul(depth)
            .ok_or(Error::SizeOverflow("Huffman predictor row length"))?;
        Ok(Some(Self {
            alphabet,
            profile,
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
            profile: self.profile,
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
    let build = |hist: &[u64; 256]| match histograms.profile {
        EncoderProfile::Compact => build_candidate_from_hist(hist),
        EncoderProfile::Gdal => build_liblerc_candidate(hist),
    };
    let plain = build(&histograms.plain)?;
    let delta = build(&histograms.delta)?;
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
    candidate(hist, table_bytes, codes)
}

/// The candidate's size: the code table, then the bit-stuffed codes of every
/// sample in 32-bit words, plus the word libLerc's decode lookup table may read
/// ahead (`Huffman::ComputeCompressedSize`).
fn candidate(
    hist: &[u64; 256],
    table_bytes: Vec<u8>,
    codes: Vec<Option<HuffmanCode>>,
) -> Result<Option<HuffmanCandidate>> {
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
    let data_len = payload_bits
        .div_ceil(32)
        .checked_add(1)
        .and_then(|words| words.checked_mul(4))
        .and_then(|len| len.checked_add(table_bytes.len()))
        .ok_or(Error::SizeOverflow("Huffman payload byte count"))?;
    Ok(Some(HuffmanCandidate {
        table_bytes,
        codes,
        data_len,
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

/// Builds the Huffman candidate libLerc builds for `hist` (`Huffman::ComputeCodes`,
/// `ConvertCodesToCanonical` and `WriteCodeTable` in Esri LERC 4, Apache-2.0),
/// for the GDAL profile: the code lengths of its heap-built tree, canonical
/// codes, the shortest wrap-around symbol range, and code table version 4.
/// `None` where libLerc declines Huffman coding: fewer than two used symbols,
/// or a code longer than 32 bits.
fn build_liblerc_candidate(hist: &[u64; 256]) -> Result<Option<HuffmanCandidate>> {
    let Some(lengths) = liblerc_code_lengths(hist)? else {
        return Ok(None);
    };
    let codes = canonical_codes(&lengths);
    let Some((first, end)) = liblerc_symbol_range(&lengths) else {
        return Ok(None);
    };
    let table_bytes = build_wrapped_table_bytes(&codes, first, end)?;
    candidate(hist, table_bytes, codes)
}

/// A node of libLerc's Huffman heap: its weight is the negated frequency, so
/// `std::priority_queue<Node, std::vector<Node>, std::less<Node>>` pops the
/// least frequent first.
#[derive(Debug, Clone, Copy)]
struct HeapNode {
    weight: i64,
    node: usize,
}

/// `std::priority_queue`'s heap over a vector, as libstdc++ and the MSVC STL
/// implement `push_heap` and `pop_heap`. Equal weights leave the heap in the
/// order these algorithms produce, which decides the code lengths of
/// equally frequent symbols; other standard libraries may order ties otherwise.
struct StdHeap {
    nodes: Vec<HeapNode>,
}

impl StdHeap {
    fn less(a: HeapNode, b: HeapNode) -> bool {
        a.weight < b.weight
    }

    /// Moves `value` up from `hole` (`std::__push_heap`).
    fn sift_up(&mut self, mut hole: usize, value: HeapNode) {
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if !Self::less(self.nodes[parent], value) {
                break;
            }
            self.nodes[hole] = self.nodes[parent];
            hole = parent;
        }
        self.nodes[hole] = value;
    }

    fn push(&mut self, value: HeapNode) {
        self.nodes.push(value);
        let hole = self.nodes.len() - 1;
        self.sift_up(hole, value);
    }

    /// Removes the top (`std::pop_heap` then `pop_back`): the hole at the top
    /// moves down to a leaf along the larger children, and the last element
    /// moves up from there (`std::__adjust_heap`).
    fn pop(&mut self) -> Option<HeapNode> {
        let top = *self.nodes.first()?;
        let value = self.nodes.pop()?;
        let len = self.nodes.len();
        if len == 0 {
            return Some(top);
        }
        let mut hole = 0usize;
        let mut child = 0usize;
        while child < (len - 1) / 2 {
            child = 2 * (child + 1);
            if Self::less(self.nodes[child], self.nodes[child - 1]) {
                child -= 1;
            }
            self.nodes[hole] = self.nodes[child];
            hole = child;
        }
        if len % 2 == 0 && child == (len - 2) / 2 {
            child = 2 * (child + 1);
            self.nodes[hole] = self.nodes[child - 1];
            hole = child - 1;
        }
        self.sift_up(hole, value);
        Some(top)
    }
}

/// A node of libLerc's Huffman tree: a symbol, or child 0 and child 1.
enum TreeNode {
    Leaf(u16),
    Pair(usize, usize),
}

/// The code length of each symbol of libLerc's Huffman tree for `hist`, or
/// `None` where libLerc gives up (fewer than two used symbols, or a code
/// longer than 32 bits).
///
/// Counts are `u64` and weights `i64`; libLerc's are `int`, so above about
/// 2^31 samples in one histogram libLerc's arithmetic overflows and the two
/// can differ. Lerc2 tiles never get there.
fn liblerc_code_lengths(hist: &[u64; 256]) -> Result<Option<[u8; 256]>> {
    let mut tree: Vec<TreeNode> = Vec::with_capacity(511);
    let mut heap = StdHeap {
        nodes: Vec::with_capacity(256),
    };
    for (symbol, &count) in hist.iter().enumerate() {
        if count > 0 {
            let weight = i64::try_from(count)
                .map_err(|_| Error::SizeOverflow("Huffman symbol frequency as i64"))?;
            tree.push(TreeNode::Leaf(symbol as u16));
            heap.push(HeapNode {
                weight: -weight,
                node: tree.len() - 1,
            });
        }
    }
    if heap.nodes.len() < 2 {
        return Ok(None);
    }
    while heap.nodes.len() > 1 {
        let (Some(child0), Some(child1)) = (heap.pop(), heap.pop()) else {
            return Err(Error::Internal("Huffman heap lost a node"));
        };
        let weight = child0
            .weight
            .checked_add(child1.weight)
            .ok_or(Error::SizeOverflow("Huffman tree weight"))?;
        tree.push(TreeNode::Pair(child0.node, child1.node));
        heap.push(HeapNode {
            weight,
            node: tree.len() - 1,
        });
    }
    let root = heap
        .pop()
        .ok_or(Error::Internal("Huffman heap lost its root"))?
        .node;
    let mut lengths = [0u8; 256];
    let mut stack = vec![(root, 0u8)];
    while let Some((node, depth)) = stack.pop() {
        match tree.get(node) {
            Some(TreeNode::Leaf(symbol)) => lengths[*symbol as usize] = depth,
            Some(TreeNode::Pair(child0, child1)) => {
                // libLerc's `TreeToLUT` refuses codes longer than 32 bits.
                if depth == 32 {
                    return Ok(None);
                }
                stack.push((*child1, depth + 1));
                stack.push((*child0, depth + 1));
            }
            None => return Err(Error::Internal("Huffman tree node is out of bounds")),
        }
    }
    Ok(Some(lengths))
}

/// libLerc's canonical codes for `lengths` (`ConvertCodesToCanonical`):
/// symbols ordered by decreasing code length, then increasing symbol, the
/// longest code being 0.
fn canonical_codes(lengths: &[u8; 256]) -> Vec<Option<HuffmanCode>> {
    let mut order: Vec<usize> = (0..lengths.len()).filter(|&s| lengths[s] > 0).collect();
    order.sort_unstable_by(|&a, &b| lengths[b].cmp(&lengths[a]).then(a.cmp(&b)));
    let mut codes = vec![None; lengths.len()];
    let Some(&longest) = order.first() else {
        return codes;
    };
    let mut bit_len = lengths[longest];
    let mut code = 0u32;
    for symbol in order {
        let delta = bit_len - lengths[symbol];
        // `bit_len` is at most 32, so a shift of 32 empties the code as
        // libLerc's 32-bit shift would.
        code = code.checked_shr(u32::from(delta)).unwrap_or(0);
        bit_len -= delta;
        codes[symbol] = Some(HuffmanCode {
            bit_len,
            bits: code,
        });
        code = code.wrapping_add(1);
    }
    codes
}

/// The symbol range libLerc's code table covers (`Huffman::GetRange`): from
/// the first to the last used symbol, or, when shorter, the range that wraps
/// around the end of the alphabet past its longest run of unused symbols.
/// `end` may exceed the alphabet size; symbols wrap modulo it.
fn liblerc_symbol_range(lengths: &[u8; 256]) -> Option<(usize, usize)> {
    let size = lengths.len();
    let first = lengths.iter().position(|&len| len > 0)?;
    let last = lengths.iter().rposition(|&len| len > 0)?;
    let (mut start, mut end) = (first, last + 1);
    let (mut gap_start, mut gap_len) = (0usize, 0usize);
    let mut j = 0usize;
    while j < size {
        while j < size && lengths[j] > 0 {
            j += 1;
        }
        let k0 = j;
        while j < size && lengths[j] == 0 {
            j += 1;
        }
        if j - k0 > gap_len {
            (gap_start, gap_len) = (k0, j - k0);
        }
    }
    if size - gap_len < end - start {
        start = gap_start + gap_len;
        end = gap_start + size;
    }
    (start < end).then_some((start, end))
}

/// libLerc's code table (`Huffman::WriteCodeTable`): version 4, the alphabet
/// size, the symbol range `[start, end)` wrapping modulo the size, the code
/// lengths over the range, bit-stuffed, then the codes.
fn build_wrapped_table_bytes(
    codes: &[Option<HuffmanCode>],
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    let size = codes.len();
    let symbol = |i: usize| if i < size { i } else { i - size };
    let code_lengths: Vec<u32> = (start..end)
        .map(|i| codes[symbol(i)].map_or(0, |code| u32::from(code.bit_len)))
        .collect();
    let as_i32 = |value: usize| {
        i32::try_from(value).map_err(|_| Error::SizeOverflow("Huffman code table field"))
    };
    let mut out = Vec::new();
    out.extend_from_slice(&4i32.to_le_bytes());
    out.extend_from_slice(&as_i32(size)?.to_le_bytes());
    out.extend_from_slice(&as_i32(start)?.to_le_bytes());
    out.extend_from_slice(&as_i32(end)?.to_le_bytes());
    write_raw_block(&mut out, &code_lengths)?;

    let mut writer = MsbBitWriter::default();
    for code in (start..end).filter_map(|i| codes[symbol(i)]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fibonacci_histogram(symbols: usize) -> [u64; 256] {
        let mut hist = [0u64; 256];
        let (mut a, mut b) = (1u64, 1u64);
        for slot in hist.iter_mut().take(symbols) {
            *slot = a;
            (a, b) = (b, a + b);
        }
        hist
    }

    #[test]
    fn liblerc_codes_stop_at_32_bits() {
        let lengths = liblerc_code_lengths(&fibonacci_histogram(33))
            .unwrap()
            .unwrap();
        assert_eq!(lengths.iter().copied().max(), Some(32));
        assert_eq!(
            liblerc_code_lengths(&fibonacci_histogram(34)).unwrap(),
            None
        );
    }

    #[test]
    fn liblerc_declines_a_single_symbol() {
        let mut hist = [0u64; 256];
        hist[9] = 100;
        assert_eq!(liblerc_code_lengths(&hist).unwrap(), None);
        assert!(build_liblerc_candidate(&hist).unwrap().is_none());
    }

    #[test]
    fn canonical_codes_order_by_length_then_symbol() {
        let mut lengths = [0u8; 256];
        lengths[3] = 1;
        lengths[1] = 2;
        lengths[7] = 3;
        lengths[200] = 3;
        let codes = canonical_codes(&lengths);
        let code = |s: usize| codes[s].map(|c| (c.bit_len, c.bits));
        // Longest first, from 0: 7 -> 000, 200 -> 001; then 1 -> 01, 3 -> 1.
        assert_eq!(code(7), Some((3, 0b000)));
        assert_eq!(code(200), Some((3, 0b001)));
        assert_eq!(code(1), Some((2, 0b01)));
        assert_eq!(code(3), Some((1, 0b1)));
    }

    #[test]
    fn liblerc_range_wraps_past_the_longest_gap() {
        let mut lengths = [0u8; 256];
        for symbol in [254, 255, 0, 1, 2] {
            lengths[symbol] = 3;
        }
        assert_eq!(liblerc_symbol_range(&lengths), Some((254, 259)));
        let mut lengths = [0u8; 256];
        lengths[10] = 1;
        lengths[20] = 1;
        assert_eq!(liblerc_symbol_range(&lengths), Some((10, 21)));
        assert_eq!(liblerc_symbol_range(&[0u8; 256]), None);
    }

    /// The tied frequencies of `liblerc-v4-u8-huffman-ties.lerc2`. libLerc
    /// 4.2.0 and rasterio 1.5.2's libLerc (both MSVC STL) write these lengths;
    /// libc++'s pop order would give symbols 0 and 18 lengths 3 and 2.
    #[test]
    fn liblerc_code_lengths_pin_the_tie_order() {
        let counts = [320u64, 120, 320, 320, 320, 240, 40, 160, 80];
        let mut hist = [0u64; 256];
        for (i, &count) in counts.iter().enumerate() {
            hist[i * 9] = count;
        }
        let lengths = liblerc_code_lengths(&hist).unwrap().unwrap();
        let used: Vec<u8> = (0..counts.len()).map(|i| lengths[i * 9]).collect();
        assert_eq!(used, [2, 4, 3, 3, 3, 3, 5, 3, 5]);
    }

    /// Equal weights leave the heap in libstdc++'s and the MSVC STL's order.
    #[test]
    fn std_heap_pops_ties_in_std_order() {
        let mut heap = StdHeap { nodes: Vec::new() };
        for node in 0..7 {
            heap.push(HeapNode { weight: -1, node });
        }
        let mut order = Vec::new();
        while let Some(top) = heap.pop() {
            order.push(top.node);
        }
        // libc++'s pop order would be [0, 1, 3, 6, 5, 4, 2].
        assert_eq!(order, [0, 2, 5, 6, 4, 1, 3]);
    }

    #[test]
    fn std_heap_pops_in_weight_order() {
        let mut heap = StdHeap { nodes: Vec::new() };
        for (node, weight) in [-5i64, -1, -3, -1, -2, -8, -1].into_iter().enumerate() {
            heap.push(HeapNode { weight, node });
        }
        let mut weights = Vec::new();
        while let Some(top) = heap.pop() {
            weights.push(top.weight);
        }
        assert_eq!(weights, [-1, -1, -1, -2, -3, -5, -8]);
    }
}
