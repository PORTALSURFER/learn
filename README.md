# Learn · Native Rust + GPUI Cognitive Mesh Learning App

A local-first desktop app built in **Rust + GPUI** that trains you on **any topic** through live AI-generated 4-choice questions, a neuro-symbolic **Cognitive Mesh** stored in SQLite, and a bounded background question buffer.

## Architecture

- **`crates/learn-core`** (zero UI dependencies, `< 1s` boundary tests, ready to reuse for a future `learn.iocore.dev` online/sync backend):
  - **Cognitive Mesh (`mesh.rs`)**: Tracks `(μ mastery, σ uncertainty, S stability)` per concept, signed typed edges (`Requires` +w, `Bridges` +w, `Confuses` -w), predictive-coding **tension seams**, and **Shadow Misconception Nodes** with Bayesian-lite posterior updates and automatic pruning.
  - **Zero-extra-LLM local belief propagation**: Distractors are pre-tagged at generation time with `trapKind`, `trapTarget`, and `whyWrong`, so clicking an answer runs 1-hop damped credit/doubt backpropagation, associative stability refresh, confusion-edge weighting, and shadow-node updates in `< 1ms`.
  - **~200-Token Cognitive Slice**: Compiles `macro_map`, `frontier`, `tension_seams`, `active_hypotheses`, `confusion_pairs`, and `strong_bridges` (cross-topic mastered analogies) for every live AI prompt.
  - **Clarifying Question Gap Capture**: Ask any question mid-quiz (e.g., *"What is a motif?"*) to get a spoiler-free explanation tied to your topic and automatically add the missing prerequisite concept to your Cognitive Mesh for upcoming practice.
  - **Bounded Adaptive Buffer (`DEFAULT_BUFFER_TARGET = 3`)**: Keeps up to 3 ready questions queued ahead on distinct scheduled concepts, invalidates stale queued questions for concepts you just answered, and refills off the UI thread for zero-wait transitions.
  - **Local Subscription CLI & API Runners (`runner.rs`)**: Auto-detects `codex` (`codex exec --ephemeral --skip-git-repo-check -s read-only --output-schema`), `agy`, `gemini`, `claude`, plus optional direct Gemini/OpenAI API keys and an instant offline Demo mode.
- **`crates/learn-app`** (native macOS GPUI desktop UI):
  - Local user profiles persisted in `~/.learn-iocore/learn.db` (SQLite WAL mode).
  - Live Quiz Arena with plain-English **Goal & Context**, keyboard shortcuts (`1`–`4` / `A`–`D`, `Enter` / `Space`), tailored **"Why your pick didn't fit"** feedback, Cognitive Mesh update notes, and the bottom **Ask Tutor** clarifying-question bar.
  - Visual **Cognitive Mesh Map** displaying Depth 1–4 concept tiers, `μ ± σ` & stability `S`, High-Tension Seams, Active Shadow Misconception Hypotheses, Inhibitory Confusion Links, and Cross-Topic Analogy Bridges.

## Quick Start

```bash
# Run all boundary unit tests (< 1s)
cargo test --workspace

# Launch the native GPUI desktop app
cargo run -p learn-app
```

## Keyboard Shortcuts

- **Quiz Arena**: `1`–`4` (or `A`–`D`) to pick an answer, `Enter` or `Space` for next question.
- **Forms**: `Tab` switches Auth fields, `Cmd+V` pastes into focused inputs, `Enter` submits.
- **Window**: `Cmd+W` / red traffic light or `Cmd+Q` quits cleanly.
