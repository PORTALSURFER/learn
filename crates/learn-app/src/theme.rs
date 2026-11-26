use gpui::{Rgba, rgb, rgba};

// Sleek Monochrome + Signal Red Sci-Fi Palette
// Strictly carbon blacks, titanium greys, stark white, and laser red.

pub fn transparent() -> Rgba {
    rgba(0x00000000)
}

pub fn pure_white() -> Rgba {
    rgb(0xFAFAFC)
}

pub fn bg_canvas() -> Rgba {
    rgb(0x050506)
}

pub fn bg_sidebar() -> Rgba {
    rgb(0x09090B)
}

pub fn bg_card() -> Rgba {
    rgb(0x101014)
}

pub fn bg_card_hover() -> Rgba {
    rgb(0x18181E)
}

pub fn bg_input() -> Rgba {
    rgb(0x08080A)
}

pub fn border_subtle() -> Rgba {
    rgb(0x23232A)
}

pub fn border_focus() -> Rgba {
    rgb(0xE11D2E)
}

pub fn text_primary() -> Rgba {
    rgb(0xF4F4F6)
}

pub fn text_secondary() -> Rgba {
    rgb(0xA1A1AA)
}

pub fn text_muted() -> Rgba {
    rgb(0x63636E)
}

// Primary Signal Red accent
pub fn accent() -> Rgba {
    rgb(0xE11D2E)
}

pub fn accent_hover() -> Rgba {
    rgb(0xFF2B3E)
}

pub fn accent_soft() -> Rgba {
    rgb(0x240B0F)
}

pub fn accent_text() -> Rgba {
    rgb(0xFF5C6B)
}

// Verified / Mastered / Success -> Stark Platinum White on Graphite
pub fn emerald() -> Rgba {
    rgb(0xF4F4F6)
}

pub fn emerald_soft() -> Rgba {
    rgb(0x18181E)
}

pub fn emerald_text() -> Rgba {
    rgb(0xFFFFFF)
}

// Caution / Review / Guessing -> Smoked Silver-Red Telemetry
pub fn amber() -> Rgba {
    rgb(0xA1A1AA)
}

pub fn amber_soft() -> Rgba {
    rgb(0x1B1719)
}

pub fn amber_text() -> Rgba {
    rgb(0xE4E4E7)
}

// Alert / Error / Misconception -> Deep Crimson & Laser Red
pub fn rose() -> Rgba {
    rgb(0xFF2B3E)
}

pub fn rose_soft() -> Rgba {
    rgb(0x2A0A10)
}

pub fn rose_text() -> Rgba {
    rgb(0xFF8A94)
}

// Technical / Structural Highlights -> Crisp Steel & White
pub fn sky_soft() -> Rgba {
    rgb(0x16161C)
}

pub fn sky_text() -> Rgba {
    rgb(0xD4D4D8)
}
