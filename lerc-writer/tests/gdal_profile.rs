#![allow(missing_docs)]

//! The GDAL encoder profile against Lerc2 blobs written by GDAL itself.

use lerc_core::{Error, MaskView, RasterView};
use lerc_writer::{encode, EncodeOptions, EncoderProfile};

#[path = "../../test-support/surfaces.rs"]
mod surfaces;

use surfaces::Surface;

// Single-tile GeoTIFFs written by GDAL 3.12.2 (rasterio 1.5.2, bundled
// libLerc) with `COMPRESS=LERC` and the surface's `MAX_Z_ERROR`; NaN marks
// invalid pixels. The blobs are the raw TIFF tile payloads.
const GDAL_MASKED_DEPTH: &[u8] =
    include_bytes!("../../testdata/interoperability/gdal-3.12-f32-masked-depth.lerc2");
const GDAL_CENTIMETRE_GRID: &[u8] =
    include_bytes!("../../testdata/interoperability/gdal-3.12-f32-centimetre-grid.lerc2");

fn encode_surface(surface: &Surface, profile: EncoderProfile) -> Vec<u8> {
    let raster = RasterView::new(
        surface.width as u32,
        surface.height as u32,
        1,
        &surface.pixels,
    )
    .unwrap();
    let mask = surface
        .mask
        .as_deref()
        .map(|mask| MaskView::new(surface.width as u32, surface.height as u32, mask).unwrap());
    encode(
        raster,
        mask,
        EncodeOptions::new()
            .with_max_z_error(surface.max_z_error)
            .with_profile(profile),
    )
    .unwrap()
}

fn gdal_cases() -> [(Surface, &'static [u8]); 2] {
    [
        (surfaces::gdal_fixture_depth(), GDAL_MASKED_DEPTH),
        (surfaces::gdal_fixture_elevation(), GDAL_CENTIMETRE_GRID),
    ]
}

#[test]
fn gdal_profile_reproduces_gdal_written_blobs() {
    for (surface, gdal_blob) in gdal_cases() {
        let blob = encode_surface(&surface, EncoderProfile::Gdal);
        assert!(
            blob == gdal_blob,
            "{}: {} bytes differ from GDAL's {} bytes",
            surface.name,
            blob.len(),
            gdal_blob.len()
        );
    }
}

#[test]
fn compact_profile_decodes_to_gdal_values() {
    // Both profiles quantize identically; the compact one may only pick
    // smaller block encodings for the same values.
    for (surface, gdal_blob) in gdal_cases() {
        let blob = encode_surface(&surface, EncoderProfile::Compact);
        assert!(blob.len() <= gdal_blob.len(), "{}", surface.name);
        let ours = lerc_reader::decode(&blob).unwrap();
        let gdal = lerc_reader::decode(gdal_blob).unwrap();
        assert_eq!(ours.pixels, gdal.pixels, "{}", surface.name);
        assert_eq!(ours.mask, gdal.mask, "{}", surface.name);
    }
}

#[test]
fn gdal_profile_rejects_settings_lerc2_v4_cannot_reproduce() {
    let pixels = [1.0f32, 2.0, 3.0, 4.0];
    let raster = RasterView::new(2, 2, 1, &pixels).unwrap();
    let gdal = EncodeOptions::new().with_profile(EncoderProfile::Gdal);
    assert!(matches!(
        encode(raster, None, gdal.with_micro_block_size(16)),
        Err(Error::InvalidArgument(_))
    ));

    let depth_two = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let raster = RasterView::new(2, 2, 2, &depth_two).unwrap();
    assert!(matches!(
        encode(raster, None, gdal.with_no_data_value(-1.0)),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn gdal_profile_never_writes_difference_tiles() {
    // Compact encoding uses Lerc2 v5 difference tiles for this lossless
    // multidimensional integer raster; Lerc2 v4 has none.
    let pixels: Vec<u16> = (0..64u16).flat_map(|value| [value, value + 1]).collect();
    let raster = RasterView::new(8, 8, 2, &pixels).unwrap();
    let options = EncodeOptions::new().with_max_z_error(0.5);
    let compact = encode(raster, None, options).unwrap();
    let gdal = encode(raster, None, options.with_profile(EncoderProfile::Gdal)).unwrap();
    assert_eq!(
        lerc_reader::get_blob_info(&compact).unwrap().version,
        lerc_core::Version::Lerc2(5)
    );
    assert_eq!(
        lerc_reader::get_blob_info(&gdal).unwrap().version,
        lerc_core::Version::Lerc2(4)
    );
    assert_eq!(
        lerc_reader::decode(&compact).unwrap().pixels,
        lerc_reader::decode(&gdal).unwrap().pixels
    );
}
