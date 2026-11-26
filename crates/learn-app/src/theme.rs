use gpui::{Rgba, rgb, rgba};

// Modern Monochrome + Laser Red Terminal Palette
// Pure carbon blacks, steel hairline borders, stark white, and signal red.

pub fn mono_font() -> &'static str {
    "Menlo"
}

pub fn ascii_bar(ratio: f64, width: usize) -> String {
    let clamped = ratio.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

pub fn transparent() -> Rgba {
    rgba(0x00000000)
}

pub fn pure_white() -> Rgba {
    rgb(0xFAFAFC)
}

pub fn bg_canvas() -> Rgba {
    rgb(0x040405)
}

pub fn bg_sidebar() -> Rgba {
    rgb(0x08080A)
}

pub fn bg_card() -> Rgba {
    rgb(0x0D0D11)
}

pub fn bg_card_hover() -> Rgba {
    rgb(0x16161C)
}

pub fn bg_input() -> Rgba {
    rgb(0x060608)
}

pub fn border_subtle() -> Rgba {
    rgb(0x24242C)
}

pub fn border_focus() -> Rgba {
    rgb(0xE11D2E)
}

pub fn text_primary() -> Rgba {
    rgb(0xF4F4F6)
}

pub fn text_secondary() -> Rgba {
    rgb(0x9A9AA6)
}

pub fn text_muted() -> Rgba {
    rgb(0x5C5C68)
}

// Primary Signal Red accent
pub fn accent() -> Rgba {
    rgb(0xE11D2E)
}

pub fn accent_hover() -> Rgba {
    rgb(0xFF2B3E)
}

pub fn accent_soft() -> Rgba {
    rgb(0x21090D)
}

pub fn accent_text() -> Rgba {
    rgb(0xFF5C6B)
}

// Verified / Mastered / Success -> Stark Platinum White on Graphite
pub fn emerald() -> Rgba {
    rgb(0xF4F4F6)
}

pub fn emerald_soft() -> Rgba {
    rgb(0x15151B)
}

pub fn emerald_text() -> Rgba {
    rgb(0xFFFFFF)
}

// Caution / Review / Guessing -> Steel Silver on Dark Charcoal
pub fn amber() -> Rgba {
    rgb(0x9A9AA6)
}

pub fn amber_soft() -> Rgba {
    rgb(0x171416)
}

pub fn amber_text() -> Rgba {
    rgb(0xE4E4E7)
}

// Alert / Error / Misconception -> Deep Crimson & Laser Red
pub fn rose() -> Rgba {
    rgb(0xFF2B3E)
}

pub fn rose_soft() -> Rgba {
    rgb(0x26080E)
}

pub fn rose_text() -> Rgba {
    rgb(0xFF8A94)
}

// Technical / Structural Highlights -> Crisp Steel & White
pub fn sky_soft() -> Rgba {
    rgb(0x14141A)
}

pub fn sky_text() -> Rgba {
    rgb(0xD4D4D8)
}
