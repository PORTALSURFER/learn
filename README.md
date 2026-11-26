# Learn · Native Rust + GPUI Adaptive Learning App

A local-first desktop app built in **Rust + GPUI** that trains you on **any topic** through live AI-generated 4-choice questions, spaced repetition, and a compressed persistent **Knowledge Map** stored in SQLite.

## How It Works

1. **Local User Profiles**: Sign up or sign in locally (`~/.learn-iocore/learn.db`). All learned concepts, misconceptions, mastery scores, and streaks persist across restarts.
2. **Compressed Knowledge Map (`learn-core::store`)**: Instead of hoarding massive raw Q&A logs, each topic is modeled as compact concept nodes (`depth 1–4`, `mastery`, SM-2 `ease_factor` & `interval_hours`, max 3 concise `misconceptions`, max 4 `recent_angles`), and answered questions automatically prune to a rolling window of 10 per topic.
3. **Deterministic Learning Scheduler (`learn-core::scheduler`)**: Prioritizes:
   - `review_mistake`: Re-tests missed concepts from a fresh angle to resolve tracked misconceptions.
   - `reinforce`: Spaced-repetition checks on mastered concepts.
   - `introduce`: Interleaves unseen concepts from fundamentals upward.
   - `dig_deeper`: Automatically deepens and expands the Knowledge Map once fundamentals reach high confidence.
4. **Local Subscription CLI & API Runners (`learn-core::runner`)**:
   - **Codex CLI** (`codex exec --ephemeral --skip-git-repo-check -s read-only --output-schema ...`)
   - **Antigravity CLI** (`agy -p ...`)
   - **Gemini CLI** (`gemini -p ...`)
   - **Claude CLI** (`claude -p ... --output-format text`)
   - **Direct Gemini API / OpenAI API** (via `GEMINI_API_KEY` / `OPENAI_API_KEY` or saved in Settings)
   - **Instant Offline Demo** fallback for immediate testing
5. **Background Prefetching**: While you read explanations after answering, the next tailored question is prefetched off the UI thread so clicking **Next Question** (`Enter`) feels instant.

## Quick Start

```bash
# Run the fast (< 1s) boundary unit tests for learn-core
cargo test -p learn-core

# Launch the native GPUI desktop app
cargo run -p learn-app
```

## Keyboard Shortcuts

- **Quiz Arena**: `1`–`4` (or `A`–`D`) to pick an answer, `Enter` or `Space` for next question.
- **Forms**: `Tab` switches Auth fields, `Cmd+V` pastes into focused inputs, `Enter` submits.
- **App**: `Cmd+Q` quits cleanly.

## Extending to `learn.iocore.dev` Later

All persistence, scheduling, and AI runner logic lives cleanly inside `crates/learn-core` with zero UI dependencies—making it straightforward to add a cloud sync endpoint or hosted web/MCP backend for `learn.iocore.dev` on top of the exact same crate.
