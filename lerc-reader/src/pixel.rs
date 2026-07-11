pub(crate) use lerc_core::{
    bits_required, count_valid_in_block, fletcher32, output_value, read_scalar, read_typed_values,
    read_values_as, words_from_padded, Sample,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct AllValid;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaskValidity<'a>(&'a [u8]);

impl<'a> MaskValidity<'a> {
    pub(crate) fn new(mask: &'a [u8]) -> Self {
        Self(mask)
    }
}

pub(crate) trait Validity: Copy {
    fn is_valid(self, pixel: usize) -> bool;

    fn count_in_block(
        self,
        width: usize,
        x: usize,
        y: usize,
        block_width: usize,
        block_height: usize,
    ) -> usize;
}

impl Validity for AllValid {
    #[inline]
    fn is_valid(self, _pixel: usize) -> bool {
        true
    }

    #[inline]
    fn count_in_block(
        self,
        _width: usize,
        _x: usize,
        _y: usize,
        block_width: usize,
        block_height: usize,
    ) -> usize {
        block_width * block_height
    }
}

impl Validity for MaskValidity<'_> {
    #[inline]
    fn is_valid(self, pixel: usize) -> bool {
        self.0[pixel] != 0
    }

    #[inline]
    fn count_in_block(
        self,
        width: usize,
        x: usize,
        y: usize,
        block_width: usize,
        block_height: usize,
    ) -> usize {
        count_valid_in_block(self.0, width, x, y, block_width, block_height)
    }
}
