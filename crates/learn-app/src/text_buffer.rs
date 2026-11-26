use std::ops::Range;

/// Pure UTF-8 safe single-line text buffer supporting native selection,
/// word/line navigation, and macOS/standard deletion shortcuts.
#[derive(Debug, Clone, Default)]
pub struct TextBuffer {
    text: String,
    cursor: usize,
    anchor: usize,
}

impl TextBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn from_string(text: String) -> Self {
        let end = text.len();
        Self {
            text,
            cursor: end,
            anchor: end,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn set_text(&mut self, text: String) {
        let end = text.len();
        self.text = text;
        self.cursor = end;
        self.anchor = end;
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.anchor = 0;
    }

    pub fn selected_range(&self) -> Range<usize> {
        if self.anchor <= self.cursor {
            self.anchor..self.cursor
        } else {
            self.cursor..self.anchor
        }
    }

    pub fn has_selection(&self) -> bool {
        self.anchor != self.cursor
    }

    pub fn selected_text(&self) -> &str {
        let range = self.selected_range();
        &self.text[range]
    }

    pub fn select_all(&mut self) {
        self.anchor = 0;
        self.cursor = self.text.len();
    }

    pub fn collapse_to_end(&mut self) {
        let end = self.text.len();
        self.cursor = end;
        self.anchor = end;
    }

    pub fn insert_str(&mut self, input: &str) {
        if self.has_selection() {
            let range = self.selected_range();
            self.text.replace_range(range.clone(), input);
            let next = range.start + input.len();
            self.cursor = next;
            self.anchor = next;
        } else {
            self.text.insert_str(self.cursor, input);
            self.cursor += input.len();
            self.anchor = self.cursor;
        }
    }

    pub fn delete_selection(&mut self) -> bool {
        if !self.has_selection() {
            return false;
        }
        let range = self.selected_range();
        self.text.replace_range(range.clone(), "");
        self.cursor = range.start;
        self.anchor = range.start;
        true
    }

    pub fn backspace(&mut self) {
        if self.delete_selection() {
            return;
        }
        let prev = self.prev_char_boundary(self.cursor);
        if prev < self.cursor {
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
            self.anchor = prev;
        }
    }

    pub fn delete_forward(&mut self) {
        if self.delete_selection() {
            return;
        }
        let next = self.next_char_boundary(self.cursor);
        if next > self.cursor {
            self.text.replace_range(self.cursor..next, "");
            self.anchor = self.cursor;
        }
    }

    pub fn delete_word_backward(&mut self) {
        if self.delete_selection() {
            return;
        }
        let prev = self.prev_word_boundary(self.cursor);
        if prev < self.cursor {
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
            self.anchor = prev;
        }
    }

    pub fn delete_to_start(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor > 0 {
            self.text.replace_range(0..self.cursor, "");
            self.cursor = 0;
            self.anchor = 0;
        }
    }

    pub fn delete_to_end(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor < self.text.len() {
            self.text.replace_range(self.cursor..self.text.len(), "");
            self.anchor = self.cursor;
        }
    }

    pub fn move_left(&mut self, extend_selection: bool) {
        if !extend_selection && self.has_selection() {
            let start = self.selected_range().start;
            self.cursor = start;
            self.anchor = start;
            return;
        }
        let prev = self.prev_char_boundary(self.cursor);
        self.cursor = prev;
        if !extend_selection {
            self.anchor = prev;
        }
    }

    pub fn move_right(&mut self, extend_selection: bool) {
        if !extend_selection && self.has_selection() {
            let end = self.selected_range().end;
            self.cursor = end;
            self.anchor = end;
            return;
        }
        let next = self.next_char_boundary(self.cursor);
        self.cursor = next;
        if !extend_selection {
            self.anchor = next;
        }
    }

    pub fn move_word_left(&mut self, extend_selection: bool) {
        let prev = self.prev_word_boundary(self.cursor);
        self.cursor = prev;
        if !extend_selection {
            self.anchor = prev;
        }
    }

    pub fn move_word_right(&mut self, extend_selection: bool) {
        let next = self.next_word_boundary(self.cursor);
        self.cursor = next;
        if !extend_selection {
            self.anchor = next;
        }
    }

    pub fn move_to_start(&mut self, extend_selection: bool) {
        self.cursor = 0;
        if !extend_selection {
            self.anchor = 0;
        }
    }

    pub fn move_to_end(&mut self, extend_selection: bool) {
        let end = self.text.len();
        self.cursor = end;
        if !extend_selection {
            self.anchor = end;
        }
    }

    /// Splits visible text (plain or bullet-masked) into `(before, selected, after)` parts
    /// so the UI can highlight the active selection and position the caret accurately.
    pub fn display_parts(&self, masked: bool) -> (String, String, String) {
        let range = self.selected_range();
        let before_raw = &self.text[..range.start];
        let sel_raw = &self.text[range.start..range.end];
        let after_raw = &self.text[range.end..];

        if masked {
            (
                "•".repeat(before_raw.chars().count()),
                "•".repeat(sel_raw.chars().count()),
                "•".repeat(after_raw.chars().count()),
            )
        } else {
            (
                before_raw.to_string(),
                sel_raw.to_string(),
                after_raw.to_string(),
            )
        }
    }

    fn prev_char_boundary(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        self.text[..pos]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn next_char_boundary(&self, pos: usize) -> usize {
        if pos >= self.text.len() {
            return self.text.len();
        }
        let mut iter = self.text[pos..].char_indices();
        let _ = iter.next();
        iter.next()
            .map(|(rel, _)| pos + rel)
            .unwrap_or(self.text.len())
    }

    fn prev_word_boundary(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let mut chars: Vec<(usize, char)> = self.text[..pos].char_indices().collect();
        while let Some((_, ch)) = chars.last() {
            if !ch.is_alphanumeric() {
                chars.pop();
            } else {
                break;
            }
        }
        while let Some((_, ch)) = chars.last() {
            if ch.is_alphanumeric() {
                chars.pop();
            } else {
                break;
            }
        }
        chars
            .last()
            .map(|(idx, ch)| idx + ch.len_utf8())
            .unwrap_or(0)
    }

    fn next_word_boundary(&self, pos: usize) -> usize {
        if pos >= self.text.len() {
            return self.text.len();
        }
        let slice = &self.text[pos..];
        let mut iter = slice.char_indices().peekable();
        while let Some(&(_, ch)) = iter.peek() {
            if !ch.is_alphanumeric() {
                iter.next();
            } else {
                break;
            }
        }
        while let Some(&(_, ch)) = iter.peek() {
            if ch.is_alphanumeric() {
                iter.next();
            } else {
                break;
            }
        }
        iter.peek()
            .map(|(rel, _)| pos + rel)
            .unwrap_or(self.text.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_all_and_replace_clears_accidental_paste() {
        let mut buf = TextBuffer::from_string("huge accidental paste blob".into());
        buf.select_all();
        assert_eq!(buf.selected_text(), "huge accidental paste blob");
        buf.backspace();
        assert_eq!(buf.as_str(), "");

        buf.insert_str("alex");
        buf.select_all();
        buf.insert_str("jordan");
        assert_eq!(buf.as_str(), "jordan");
        assert!(!buf.has_selection());
    }

    #[test]
    fn word_and_line_deletion_and_utf8_cursor_navigation() {
        let mut buf = TextBuffer::from_string("hypnotic techno café".into());
        buf.delete_word_backward();
        assert_eq!(buf.as_str(), "hypnotic techno ");

        buf.move_word_left(true);
        assert_eq!(buf.selected_text(), "techno ");
        buf.insert_str("dub ");
        assert_eq!(buf.as_str(), "hypnotic dub ");

        buf.delete_to_start();
        assert_eq!(buf.as_str(), "");
    }

    #[test]
    fn display_parts_handles_masked_selection() {
        let mut buf = TextBuffer::from_string("secret".into());
        buf.move_left(true);
        buf.move_left(true);
        let (before, sel, after) = buf.display_parts(true);
        assert_eq!(before, "••••");
        assert_eq!(sel, "••");
        assert_eq!(after, "");
    }
}
