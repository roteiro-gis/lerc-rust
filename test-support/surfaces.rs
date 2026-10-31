#![allow(dead_code)]

//! Deterministic synthetic `f32` surfaces for lossy encoder tests.

pub struct Surface {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
    pub mask: Option<Vec<u8>>,
    pub max_z_error: f64,
}

fn grid(width: usize, height: usize, value: impl Fn(f64, f64) -> f64) -> Vec<f32> {
    (0..height)
        .flat_map(|row| (0..width).map(move |col| (col, row)))
        .map(|(col, row)| value(col as f64, row as f64) as f32)
        .collect()
}

/// Each 8x8 block spans 0.009, below the 0.01 bound, so every block can be
/// stored as a constant.
pub fn near_constant_blocks() -> Surface {
    Surface {
        name: "near-constant-blocks",
        width: 64,
        height: 64,
        pixels: grid(64, 64, |col, row| {
            100.123_456_7 + 0.003 * ((col * 7.0 + row * 13.0) % 4.0) + 2.0 * (row / 8.0).floor()
        }),
        mask: None,
        max_z_error: 0.01,
    }
}

/// Three widely spaced levels per block: a lookup table beats simple
/// bit-stuffing.
pub fn few_levels() -> Surface {
    let levels = [10.0, 13.0, 410.0];
    Surface {
        name: "few-levels",
        width: 32,
        height: 32,
        pixels: grid(32, 32, |col, row| {
            levels[((col + 2.0 * row) as usize) % levels.len()] + 0.1
        }),
        mask: None,
        max_z_error: 0.5,
    }
}

/// Centimetre data encoded with a millimetre bound.
pub fn centimetre_grid() -> Surface {
    Surface {
        name: "centimetre-grid",
        width: 48,
        height: 48,
        pixels: grid(48, 48, |col, row| {
            (100.0 * (20.0 + 3.0 * (col / 9.0).sin() + 2.0 * (row / 7.0).cos())).round() / 100.0
        }),
        mask: None,
        max_z_error: 0.001,
    }
}

/// The same smooth surface without the decimal grid.
pub fn smooth() -> Surface {
    Surface {
        name: "smooth",
        width: 48,
        height: 48,
        pixels: grid(48, 48, |col, row| {
            20.0 + 3.0 * (col / 9.0).sin() + 2.0 * (row / 7.0).cos()
        }),
        mask: None,
        max_z_error: 0.001,
    }
}

/// Terraces that are flat over 16x16 cells, so 8x8 block headers dominate.
pub fn terraces() -> Surface {
    Surface {
        name: "terraces",
        width: 128,
        height: 128,
        pixels: grid(128, 128, |col, row| {
            50.123_456_7 + 0.37 * (col / 16.0).floor() + 0.23 * (row / 16.0).floor()
        }),
        mask: None,
        max_z_error: 0.01,
    }
}

/// An inundation-depth-like surface: masked outside the wet area, smooth
/// inside, with stepped pools.
pub fn masked_depth() -> Surface {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 192;
    let depth = |col: f64, row: f64| {
        let x = col / WIDTH as f64;
        let y = row / HEIGHT as f64;
        3.0 + 8.0 * (4.0 * x).sin() * (3.0 * y).cos() + 0.3 * (6.0 * y).floor()
    };
    Surface {
        name: "masked-depth",
        width: WIDTH,
        height: HEIGHT,
        pixels: grid(WIDTH, HEIGHT, |col, row| depth(col, row).max(0.0)),
        mask: Some(
            (0..WIDTH * HEIGHT)
                .map(|pixel| u8::from(depth((pixel % WIDTH) as f64, (pixel / WIDTH) as f64) >= 0.5))
                .collect(),
        ),
        max_z_error: 0.01,
    }
}

pub fn all() -> Vec<Surface> {
    vec![
        near_constant_blocks(),
        few_levels(),
        centimetre_grid(),
        smooth(),
        terraces(),
        masked_depth(),
    ]
}

/// Polynomial-only surfaces (no transcendental functions, so the `f32`
/// values are identical on every platform) used with the vendored
/// GDAL-written fixtures.
pub fn gdal_fixture_depth() -> Surface {
    const WIDTH: usize = 160;
    const HEIGHT: usize = 112;
    let depth = |col: f64, row: f64| {
        let x = col / WIDTH as f64;
        let y = row / HEIGHT as f64;
        let bowl = 9.0 * x * (1.0 - x) * (1.5 - y) * (0.5 + y);
        bowl + 0.25 * (5.0 * y).floor() - 0.6 + 0.004 * ((col * 3.0 + row * 7.0) % 3.0)
    };
    Surface {
        name: "gdal-fixture-depth",
        width: WIDTH,
        height: HEIGHT,
        pixels: grid(WIDTH, HEIGHT, |col, row| depth(col, row).max(0.0)),
        mask: Some(
            (0..WIDTH * HEIGHT)
                .map(|pixel| u8::from(depth((pixel % WIDTH) as f64, (pixel / WIDTH) as f64) > 0.0))
                .collect(),
        ),
        max_z_error: 0.01,
    }
}

/// Centimetre-rounded elevations, all valid, encoded with a millimetre bound.
pub fn gdal_fixture_elevation() -> Surface {
    const WIDTH: usize = 160;
    const HEIGHT: usize = 112;
    Surface {
        name: "gdal-fixture-elevation",
        width: WIDTH,
        height: HEIGHT,
        pixels: grid(WIDTH, HEIGHT, |col, row| {
            let x = col / WIDTH as f64;
            let y = row / HEIGHT as f64;
            (100.0 * (210.0 + 14.0 * x * x - 9.0 * x * y + 6.0 * y * y * y)).round() / 100.0
        }),
        mask: None,
        max_z_error: 0.001,
    }
}
