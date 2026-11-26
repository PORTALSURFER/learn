use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::models::{
    ConceptInput, DetectedRunner, QuestionChoice, QuestionIntent, RunnerKind, SchedulerDirective,
    SkillLevel,
};

#[derive(Debug, Clone)]
pub struct GeneratedQuestion {
    pub prompt: String,
    pub choices: Vec<QuestionChoice>,
    pub correct_choice_id: String,
    pub explanation: String,
    pub angle_summary: String,
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub kind: RunnerKind,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub openai_api_key: String,
    pub openai_model: String,
}

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub trait CommandExecutor: Send + Sync {
    fn run(&self, program: &str, args: &[String], stdin: Option<&str>) -> Result<CommandOutput>;
}

pub struct SystemCommandExecutor;

impl CommandExecutor for SystemCommandExecutor {
    fn run(&self, program: &str, args: &[String], stdin_data: Option<&str>) -> Result<CommandOutput> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to launch '{program}' — is it installed on your PATH?"))?;

        if let Some(input) = stdin_data {
            if let Some(mut pipe) = child.stdin.take() {
                pipe.write_all(input.as_bytes())?;
            }
        } else {
            drop(child.stdin.take());
        }

        let output = child.wait_with_output()?;
        Ok(CommandOutput {
            exit_code: output.status.code().unwrap_or(1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

pub fn concept_map_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["concepts"],
        "properties": {
            "concepts": {
                "type": "array",
                "minItems": 4,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["title", "subtopic", "summary", "depth"],
                    "properties": {
                        "title": { "type": "string" },
                        "subtopic": { "type": "string" },
                        "summary": { "type": "string" },
                        "depth": { "type": "integer", "minimum": 1, "maximum": 4 }
                    }
                }
            }
        }
    })
}

pub fn question_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["prompt", "correctAnswer", "distractors", "explanation", "angleSummary"],
        "properties": {
            "prompt": { "type": "string" },
            "correctAnswer": { "type": "string" },
            "distractors": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": { "type": "string" }
            },
            "explanation": { "type": "string" },
            "angleSummary": { "type": "string" }
        }
    })
}

pub fn extract_json_value(raw: &str) -> Result<Value> {
    let trimmed = raw.trim();
    let mut candidates: Vec<&str> = Vec::new();

    if let Some(start) = trimmed.find("```") {
        let rest = &trimmed[start + 3..];
        let after_lang = rest
            .strip_prefix("json")
            .or_else(|| rest.strip_prefix("JSON"))
            .unwrap_or(rest);
        if let Some(end) = after_lang.find("```") {
            candidates.push(after_lang[..end].trim());
        }
    }
    candidates.push(trimmed);

    if let (Some(open), Some(close)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if close > open {
            candidates.push(&trimmed[open..=close]);
        }
    }

    for candidate in candidates {
        if let Ok(val) = serde_json::from_str::<Value>(candidate) {
            if val.is_object() {
                return Ok(val);
            }
        }
    }

    let snippet: String = trimmed.chars().take(180).collect();
    Err(anyhow!("AI did not return valid JSON object. Output snippet: {snippet}"))
}

#[derive(Deserialize)]
struct RawConceptMap {
    concepts: Vec<RawConcept>,
}

#[derive(Deserialize)]
struct RawConcept {
    title: String,
    #[serde(default)]
    subtopic: String,
    #[serde(default)]
    summary: String,
    #[serde(default = "default_depth")]
    depth: u8,
}

fn default_depth() -> u8 {
    1
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawQuestion {
    prompt: String,
    correct_answer: String,
    distractors: Vec<String>,
    explanation: String,
    #[serde(default)]
    angle_summary: String,
}

pub fn parse_concept_map(val: Value) -> Result<Vec<ConceptInput>> {
    let parsed: RawConceptMap = serde_json::from_value(val).context("invalid concept map JSON")?;
    let mut out = Vec::new();
    for item in parsed.concepts {
        let title = item.title.trim();
        if title.is_empty() {
            continue;
        }
        out.push(ConceptInput {
            title: title.chars().take(100).collect(),
            subtopic: if item.subtopic.trim().is_empty() {
                "Core".to_string()
            } else {
                item.subtopic.trim().chars().take(60).collect()
            },
            summary: item.summary.trim().chars().take(220).collect(),
            depth: item.depth.clamp(1, 4),
        });
    }
    if out.is_empty() {
        return Err(anyhow!("AI returned an empty concept map."));
    }
    Ok(out)
}

pub fn parse_and_shuffle_question(val: Value) -> Result<GeneratedQuestion> {
    let raw: RawQuestion = serde_json::from_value(val).context("invalid question JSON")?;
    let prompt = raw.prompt.trim().to_string();
    let correct = raw.correct_answer.trim().to_string();
    let explanation = raw.explanation.trim().to_string();
    if prompt.is_empty() || correct.is_empty() || explanation.is_empty() {
        return Err(anyhow!("Question JSON missing prompt, correctAnswer, or explanation."));
    }

    let mut seen = HashSet::new();
    seen.insert(correct.to_lowercase());
    let mut distractors = Vec::new();
    for d in raw.distractors {
        let t = d.trim().to_string();
        if !t.is_empty() && seen.insert(t.to_lowercase()) {
            distractors.push(t);
        }
    }
    if distractors.len() < 3 {
        return Err(anyhow!("Question JSON must provide 3 distinct distractors."));
    }

    let mut pool = vec![
        (correct, true),
        (distractors[0].clone(), false),
        (distractors[1].clone(), false),
        (distractors[2].clone(), false),
    ];
    let mut rng = rand::thread_rng();
    pool.shuffle(&mut rng);

    let labels = ["A", "B", "C", "D"];
    let mut correct_choice_id = "A".to_string();
    let mut choices = Vec::with_capacity(4);
    for (idx, (text, is_right)) in pool.into_iter().enumerate() {
        let id = labels[idx].to_string();
        if is_right {
            correct_choice_id = id.clone();
        }
        choices.push(QuestionChoice { id, text });
    }

    let angle_summary = if raw.angle_summary.trim().is_empty() {
        prompt.chars().take(90).collect()
    } else {
        raw.angle_summary.trim().chars().take(120).collect()
    };

    Ok(GeneratedQuestion {
        prompt,
        choices,
        correct_choice_id,
        explanation,
        angle_summary,
    })
}

pub fn build_concept_map_prompt(
    topic_title: &str,
    level: SkillLevel,
    existing_titles: &[String],
    min_depth: u8,
) -> String {
    let scope_note = if existing_titles.is_empty() {
        format!(
            "Create 8 foundational-to-intermediate concepts (depths 1 to 3) covering the essential mental models of \"{topic_title}\" for a {} learner.",
            level.as_str()
        )
    } else {
        format!(
            "Already covered concepts (DO NOT repeat these; generate 6 deeper next-level concepts at depth {min_depth}-4): {}.",
            existing_titles.join(", ")
        )
    };

    format!(
        "You are an expert adaptive tutor building a compact Knowledge Map for \"{topic_title}\".\n\
         {scope_note}\n\
         Return ONLY a valid JSON object matching:\n\
         {{\"concepts\":[{{\"title\":\"...\",\"subtopic\":\"...\",\"summary\":\"1-sentence core mental model\",\"depth\":1}}]}}\n\
         Keep titles concise (2-5 words), group subtopics into 2-4 clean themes, and make every summary crisp and accurate."
    )
}

pub fn build_question_prompt(directive: &SchedulerDirective) -> String {
    let intent_guidance = match directive.intent {
        QuestionIntent::Introduce => {
            "Introduce this concept with an intuitive, concrete 4-choice question that teaches the core distinction."
        }
        QuestionIntent::ReviewMistake => {
            "The learner previously missed this concept. Ask about the SAME concept from a fresh angle/scenario that directly clears their misconception without reusing old wording."
        }
        QuestionIntent::Reinforce => {
            "Reinforce retention by testing this concept in a slightly different practical scenario than before."
        }
        QuestionIntent::DigDeeper => {
            "The learner has strong fundamentals. Ask a deeper trade-off, synthesis, or edge-case question at the target depth."
        }
    };

    let mut lines = vec![
        format!(
            "You are a live adaptive tutor for \"{}\" (learner level: {}).",
            directive.compressed_context.topic_title,
            directive.compressed_context.level.as_str()
        ),
        format!(
            "Target concept: \"{}\" (subtopic: {}, summary: {}).",
            directive.concept.title, directive.concept.subtopic, directive.concept.summary
        ),
        format!(
            "Intent: {} (target depth {}/4). {intent_guidance}",
            directive.intent.as_str(),
            directive.target_depth
        ),
    ];

    if !directive.concept.misconceptions.is_empty() {
        lines.push(format!(
            "Misconceptions to resolve: {}",
            directive.concept.misconceptions.join(" | ")
        ));
    }
    if !directive.compressed_context.recent_angles.is_empty() {
        lines.push(format!(
            "Avoid repeating these recent angles: {}",
            directive.compressed_context.recent_angles.join(" | ")
        ));
    }
    if !directive.compressed_context.mastered_concepts.is_empty() {
        lines.push(format!(
            "Already mastered concepts: {}",
            directive.compressed_context.mastered_concepts.join(", ")
        ));
    }

    lines.push("Rules:".into());
    lines.push("- Exactly 1 unambiguously correct answer and 3 plausible, similar-length distractors (never 'All of the above' or 'None of the above').".into());
    lines.push("- Provide a crisp 2-sentence explanation of why the correct answer works.".into());
    lines.push("- Provide a 4-8 word angleSummary describing what angle this question tested.".into());
    lines.push("Return ONLY valid JSON: {\"prompt\":\"...\",\"correctAnswer\":\"...\",\"distractors\":[\"...\",\"...\",\"...\"],\"explanation\":\"...\",\"angleSummary\":\"...\"}".into());
    lines.join("\n")
}

pub struct AiRunner {
    executor: Box<dyn CommandExecutor>,
}

impl Default for AiRunner {
    fn default() -> Self {
        Self {
            executor: Box::new(SystemCommandExecutor),
        }
    }
}

impl AiRunner {
    pub fn with_executor(executor: Box<dyn CommandExecutor>) -> Self {
        Self { executor }
    }

    pub fn detect_runners(&self, gemini_api_key: &str, openai_api_key: &str) -> Vec<DetectedRunner> {
        let cli_defs = [
            (
                RunnerKind::Codex,
                "Codex CLI (OpenAI subscription)",
                "codex",
                "Uses your local `codex exec` in read-only ephemeral mode with strict JSON schema output.",
            ),
            (
                RunnerKind::Agy,
                "Antigravity CLI (Gemini subscription)",
                "agy",
                "Uses your locally authenticated `agy` CLI to generate concepts and questions.",
            ),
            (
                RunnerKind::GeminiCli,
                "Gemini CLI (Google account)",
                "gemini",
                "Uses local `gemini -p` non-interactively under your Google CLI login.",
            ),
            (
                RunnerKind::ClaudeCli,
                "Claude CLI (Anthropic subscription)",
                "claude",
                "Uses local `claude -p` non-interactively under your Claude CLI login.",
            ),
        ];

        let mut out = Vec::new();
        for (kind, label, cmd, desc) in cli_defs {
            let probe = self.executor.run(cmd, &["--version".into()], None);
            let (available, version) = match probe {
                Ok(res) if res.exit_code == 0 => {
                    let raw = if !res.stdout.trim().is_empty() {
                        res.stdout
                    } else {
                        res.stderr
                    };
                    let first = raw.lines().next().unwrap_or("").trim().to_string();
                    (true, if first.is_empty() { None } else { Some(first) })
                }
                _ => (false, None),
            };
            out.push(DetectedRunner {
                kind,
                label: label.into(),
                is_cli: true,
                available,
                command: Some(cmd.into()),
                version,
                description: desc.into(),
            });
        }

        let env_gem = std::env::var("GEMINI_API_KEY").unwrap_or_default();
        let has_gem = !gemini_api_key.trim().is_empty() || !env_gem.trim().is_empty();
        out.push(DetectedRunner {
            kind: RunnerKind::GeminiApi,
            label: "Gemini API (Direct)".into(),
            is_cli: false,
            available: has_gem,
            command: None,
            version: None,
            description: if has_gem {
                "Ready via configured GEMINI_API_KEY.".into()
            } else {
                "Set GEMINI_API_KEY env var or enter one in Settings.".into()
            },
        });

        let env_oai = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let has_oai = !openai_api_key.trim().is_empty() || !env_oai.trim().is_empty();
        out.push(DetectedRunner {
            kind: RunnerKind::OpenaiApi,
            label: "OpenAI API (Direct)".into(),
            is_cli: false,
            available: has_oai,
            command: None,
            version: None,
            description: if has_oai {
                "Ready via configured OPENAI_API_KEY.".into()
            } else {
                "Set OPENAI_API_KEY env var or enter one in Settings.".into()
            },
        });

        out.push(DetectedRunner {
            kind: RunnerKind::Demo,
            label: "Instant Offline Demo".into(),
            is_cli: false,
            available: true,
            command: None,
            version: None,
            description: "Zero-latency built-in fallback for instant testing without an AI backend.".into(),
        });

        out
    }

    pub fn pick_best_default_runner(&self) -> RunnerKind {
        let detected = self.detect_runners("", "");
        for preferred in [
            RunnerKind::Codex,
            RunnerKind::Agy,
            RunnerKind::GeminiCli,
            RunnerKind::ClaudeCli,
            RunnerKind::GeminiApi,
            RunnerKind::OpenaiApi,
        ] {
            if detected.iter().any(|r| r.kind == preferred && r.available) {
                return preferred;
            }
        }
        RunnerKind::Demo
    }

    pub fn generate_concept_map(
        &self,
        topic_title: &str,
        level: SkillLevel,
        cfg: &RunnerConfig,
        existing_titles: &[String],
        min_depth: u8,
    ) -> Result<Vec<ConceptInput>> {
        if cfg.kind == RunnerKind::Demo {
            return Ok(demo_concepts(topic_title, existing_titles, min_depth));
        }
        let prompt = build_concept_map_prompt(topic_title, level, existing_titles, min_depth);
        let json_val = self.run_json_prompt(&prompt, &concept_map_schema(), cfg)?;
        parse_concept_map(json_val)
    }

    pub fn generate_question(
        &self,
        directive: &SchedulerDirective,
        cfg: &RunnerConfig,
    ) -> Result<GeneratedQuestion> {
        if cfg.kind == RunnerKind::Demo {
            return parse_and_shuffle_question(demo_question(directive));
        }
        let prompt = build_question_prompt(directive);
        let json_val = self.run_json_prompt(&prompt, &question_schema(), cfg)?;
        parse_and_shuffle_question(json_val)
    }

    fn run_json_prompt(&self, prompt: &str, schema: &Value, cfg: &RunnerConfig) -> Result<Value> {
        match cfg.kind {
            RunnerKind::Codex => self.run_codex_cli(prompt, schema),
            RunnerKind::Agy => self.run_simple_cli("agy", &["-p".into(), prompt.into()]),
            RunnerKind::GeminiCli => self.run_simple_cli("gemini", &["-p".into(), prompt.into()]),
            RunnerKind::ClaudeCli => self.run_simple_cli(
                "claude",
                &["-p".into(), prompt.into(), "--output-format".into(), "text".into()],
            ),
            RunnerKind::GeminiApi => self.run_gemini_api(prompt, cfg),
            RunnerKind::OpenaiApi => self.run_openai_api(prompt, cfg),
            RunnerKind::Demo => Err(anyhow!("Demo runner handled directly")),
        }
    }

    fn run_codex_cli(&self, prompt: &str, schema: &Value) -> Result<Value> {
        let dir = std::env::temp_dir().join(format!("learn-codex-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let schema_path = dir.join("schema.json");
        let out_path = dir.join("out.json");
        std::fs::write(&schema_path, serde_json::to_string_pretty(schema)?)?;

        let args = vec![
            "exec".into(),
            "--ephemeral".into(),
            "--skip-git-repo-check".into(),
            "-s".into(),
            "read-only".into(),
            "--output-schema".into(),
            schema_path.to_string_lossy().into_owned(),
            "-o".into(),
            out_path.to_string_lossy().into_owned(),
            "-".into(),
        ];

        let run_result = self.executor.run("codex", &args, Some(prompt));
        let parsed = match run_result {
            Ok(res) if res.exit_code == 0 => {
                let raw = std::fs::read_to_string(&out_path).unwrap_or(res.stdout);
                extract_json_value(&raw)
            }
            Ok(res) => {
                let msg = if !res.stderr.trim().is_empty() {
                    res.stderr
                } else {
                    res.stdout
                };
                Err(anyhow!(
                    "codex exec failed (exit {}): {}",
                    res.exit_code,
                    msg.chars().take(280).collect::<String>()
                ))
            }
            Err(err) => Err(err),
        };

        let _ = std::fs::remove_dir_all(&dir);
        parsed
    }

    fn run_simple_cli(&self, cmd: &str, args: &[String]) -> Result<Value> {
        let res = self.executor.run(cmd, args, None)?;
        if res.exit_code != 0 {
            let msg = if !res.stderr.trim().is_empty() {
                res.stderr
            } else {
                res.stdout
            };
            return Err(anyhow!(
                "{cmd} failed (exit {}): {}",
                res.exit_code,
                msg.chars().take(280).collect::<String>()
            ));
        }
        extract_json_value(&res.stdout)
    }

    fn run_gemini_api(&self, prompt: &str, cfg: &RunnerConfig) -> Result<Value> {
        let key = if !cfg.gemini_api_key.trim().is_empty() {
            cfg.gemini_api_key.trim().to_string()
        } else {
            std::env::var("GEMINI_API_KEY").unwrap_or_default()
        };
        if key.is_empty() {
            return Err(anyhow!("GEMINI_API_KEY is not configured."));
        }
        let model = if cfg.gemini_model.trim().is_empty() {
            "gemini-2.5-flash"
        } else {
            cfg.gemini_model.trim()
        };
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        );
        let resp: Value = ureq::post(&url)
            .set("x-goog-api-key", &key)
            .send_json(json!({
                "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.7
                }
            }))
            .map_err(|e| anyhow!("Gemini API request failed: {e}"))?
            .into_json()?;
        let text = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("");
        extract_json_value(text)
    }

    fn run_openai_api(&self, prompt: &str, cfg: &RunnerConfig) -> Result<Value> {
        let key = if !cfg.openai_api_key.trim().is_empty() {
            cfg.openai_api_key.trim().to_string()
        } else {
            std::env::var("OPENAI_API_KEY").unwrap_or_default()
        };
        if key.is_empty() {
            return Err(anyhow!("OPENAI_API_KEY is not configured."));
        }
        let model = if cfg.openai_model.trim().is_empty() {
            "gpt-4o-mini"
        } else {
            cfg.openai_model.trim()
        };
        let resp: Value = ureq::post("https://api.openai.com/v1/chat/completions")
            .set("Authorization", &format!("Bearer {key}"))
            .send_json(json!({
                "model": model,
                "response_format": { "type": "json_object" },
                "temperature": 0.7,
                "messages": [
                    { "role": "system", "content": "You are an adaptive tutor. Output strictly valid JSON." },
                    { "role": "user", "content": prompt }
                ]
            }))
            .map_err(|e| anyhow!("OpenAI API request failed: {e}"))?
            .into_json()?;
        let text = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        extract_json_value(text)
    }
}

fn demo_concepts(topic: &str, existing: &[String], min_depth: u8) -> Vec<ConceptInput> {
    let t = topic.trim();
    let templates = vec![
        ConceptInput {
            title: format!("{t} Mental Model"),
            subtopic: "Foundations".into(),
            summary: format!("The core purpose and unifying abstraction behind {t}."),
            depth: 1,
        },
        ConceptInput {
            title: "Core Primitives & Terms".into(),
            subtopic: "Foundations".into(),
            summary: format!("Key building blocks and terminology required to reason about {t}."),
            depth: 1,
        },
        ConceptInput {
            title: "Step-by-Step Lifecycle".into(),
            subtopic: "Mechanics".into(),
            summary: format!("How inputs transform into outputs across a typical {t} flow."),
            depth: 1,
        },
        ConceptInput {
            title: "Common Failure Modes".into(),
            subtopic: "Mechanics".into(),
            summary: format!("Frequent mistakes, pitfalls, and bottlenecks seen in {t}."),
            depth: 2,
        },
        ConceptInput {
            title: "Trade-offs & Constraints".into(),
            subtopic: "Architecture".into(),
            summary: format!("Balancing speed, simplicity, and correctness in {t}."),
            depth: 2,
        },
        ConceptInput {
            title: "Applied Patterns".into(),
            subtopic: "Practice".into(),
            summary: format!("Proven ways experts structure real-world {t} solutions."),
            depth: 3,
        },
        ConceptInput {
            title: "Diagnostics & Verification".into(),
            subtopic: "Practice".into(),
            summary: format!("How to observe, measure, and debug {t} systematically."),
            depth: 3,
        },
        ConceptInput {
            title: "Subtle Edge Cases".into(),
            subtopic: "Mastery".into(),
            summary: format!("Non-obvious interactions and boundary conditions in {t}."),
            depth: 4,
        },
    ];
    let existing_lower: HashSet<String> = existing.iter().map(|s| s.to_lowercase()).collect();
    templates
        .into_iter()
        .filter(|c| !existing_lower.contains(&c.title.to_lowercase()) && c.depth >= min_depth)
        .collect()
}

fn demo_question(directive: &SchedulerDirective) -> Value {
    let tag = match directive.intent {
        QuestionIntent::Introduce => "Demo · Intro",
        QuestionIntent::ReviewMistake => "Demo · Fresh-Angle Retry",
        QuestionIntent::Reinforce => "Demo · Reinforcement",
        QuestionIntent::DigDeeper => "Demo · Deep Dive",
    };
    json!({
        "prompt": format!("[{tag}] In {}, which statement best captures \"{}\"?", directive.compressed_context.topic_title, directive.concept.title),
        "correctAnswer": directive.concept.summary,
        "distractors": [
            format!("It is a legacy convention that should always be skipped when working with {}.", directive.compressed_context.topic_title),
            format!("It eliminates all trade-offs across every subtopic of {}.", directive.compressed_context.topic_title),
            "It only matters when real-world feedback and constraints are ignored."
        ],
        "explanation": format!("Select Codex CLI, Antigravity CLI, or an API key in Settings for live AI-authored questions. Core idea for \"{}\": {}", directive.concept.title, directive.concept.summary),
        "angleSummary": format!("{} ({:?} #{})", directive.concept.title, directive.intent, directive.concept.attempts + 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CompressedContext, ConceptNode, ConceptStatus};

    struct FakeExec;

    impl CommandExecutor for FakeExec {
        fn run(&self, program: &str, args: &[String], _stdin: Option<&str>) -> Result<CommandOutput> {
            if args.first().map(|s| s.as_str()) == Some("--version") {
                return Ok(CommandOutput {
                    exit_code: if program == "codex" { 0 } else { 1 },
                    stdout: "codex-cli 0.166.0".into(),
                    stderr: String::new(),
                });
            }
            if program == "codex" {
                let out_pos = args.iter().position(|a| a == "-o").unwrap();
                let out_file = &args[out_pos + 1];
                std::fs::write(
                    out_file,
                    r#"{"prompt":"Why does Rust enforce one mutable borrow?","correctAnswer":"To prevent data races and aliasing bugs at compile time.","distractors":["To make binaries smaller.","Because pointers cannot be copied.","To disable heap allocation."],"explanation":"Exclusive mutability prevents aliasing hazards before code runs.","angleSummary":"aliasing xor mutability"}"#,
                )?;
                return Ok(CommandOutput {
                    exit_code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
            Err(anyhow!("unexpected command {program}"))
        }
    }

    #[test]
    fn detects_codex_and_parses_schema_output_safely() {
        let runner = AiRunner::with_executor(Box::new(FakeExec));
        let detected = runner.detect_runners("", "");
        let codex = detected.iter().find(|r| r.kind == RunnerKind::Codex).unwrap();
        assert!(codex.available);
        assert_eq!(codex.version.as_deref(), Some("codex-cli 0.166.0"));

        let directive = SchedulerDirective {
            intent: QuestionIntent::Introduce,
            concept: ConceptNode {
                id: "c1".into(),
                topic_id: "t1".into(),
                slug: "borrowing".into(),
                title: "Borrowing".into(),
                subtopic: "Memory".into(),
                summary: "Shared xor mutable references.".into(),
                depth: 1,
                mastery: 0.0,
                ease_factor: 2.5,
                interval_hours: 0.0,
                attempts: 0,
                correct_count: 0,
                wrong_count: 0,
                consecutive_correct: 0,
                status: ConceptStatus::Unseen,
                misconceptions: vec![],
                recent_angles: vec![],
                last_seen_at: None,
                next_review_at: None,
            },
            target_depth: 1,
            reason: "intro".into(),
            requires_new_concepts: false,
            compressed_context: CompressedContext {
                topic_title: "Rust".into(),
                level: SkillLevel::Beginner,
                mastered_concepts: vec![],
                struggling_concepts: vec![],
                recent_angles: vec![],
            },
        };

        let cfg = RunnerConfig {
            kind: RunnerKind::Codex,
            gemini_api_key: String::new(),
            gemini_model: "gemini-2.5-flash".into(),
            openai_api_key: String::new(),
            openai_model: "gpt-4o-mini".into(),
        };

        let q = runner.generate_question(&directive, &cfg).unwrap();
        assert_eq!(q.choices.len(), 4);
        let right = q.choices.iter().find(|c| c.id == q.correct_choice_id).unwrap();
        assert_eq!(right.text, "To prevent data races and aliasing bugs at compile time.");
    }
}
