use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::models::{
    AudioDemoSpec, ChoiceAudioEffect, ConceptInput, DetectedRunner, DistractorTrapKind,
    ExampleMedia, FlowStep, MediaReveal, QuestionChoice, QuestionIntent, RunnerKind,
    SchedulerDirective, SkillLevel,
};

#[derive(Debug, Clone)]
pub struct GeneratedQuestion {
    pub goal: String,
    pub primer: String,
    pub hint: String,
    pub media: Option<ExampleMedia>,
    pub prompt: String,
    pub choices: Vec<QuestionChoice>,
    pub correct_choice_id: String,
    pub explanation: String,
    pub angle_summary: String,
}

#[derive(Debug, Clone)]
pub struct ClarifyingAnalysis {
    pub answer: String,
    pub missing_concept: Option<ConceptInput>,
    pub gap_note: Option<String>,
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
                "minItems": 6,
                "maxItems": 18,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["title", "subtopic", "summary", "depth", "requires"],
                    "properties": {
                        "title": { "type": "string" },
                        "subtopic": { "type": "string" },
                        "summary": { "type": "string" },
                        "depth": { "type": "integer", "minimum": 1, "maximum": 4 },
                        "requires": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            }
        }
    })
}

pub fn media_object_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "kind", "reveal", "title", "caption",
            "values", "labels", "flowSteps",
            "leftTitle", "leftPoints", "rightTitle", "rightPoints",
            "voice", "bpm", "baseHz", "cutoffStartHz", "cutoffEndHz",
            "afterValues", "accentStep", "nudgeMs", "includeDrums",
            "beforeLabel", "afterLabel", "html"
        ],
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["none", "curve", "flow", "contrast", "audio", "interactive_html"]
            },
            "reveal": {
                "type": "string",
                "enum": ["before_answer", "after_answer"]
            },
            "title": { "type": "string" },
            "caption": { "type": "string" },
            "values": { "type": "array", "items": { "type": "number" } },
            "afterValues": { "type": "array", "items": { "type": "number" } },
            "labels": { "type": "array", "items": { "type": "string" } },
            "flowSteps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["label", "note"],
                    "properties": {
                        "label": { "type": "string" },
                        "note": { "type": "string" }
                    }
                }
            },
            "leftTitle": { "type": "string" },
            "leftPoints": { "type": "array", "items": { "type": "string" } },
            "rightTitle": { "type": "string" },
            "rightPoints": { "type": "array", "items": { "type": "string" } },
            "voice": { "type": "string", "enum": ["synth", "rim_click", "kick_bass", "delay", "reverb_rumble", "saturation"] },
            "bpm": { "type": "number" },
            "baseHz": { "type": "number" },
            "cutoffStartHz": { "type": "number" },
            "cutoffEndHz": { "type": "number" },
            "accentStep": { "type": "integer" },
            "nudgeMs": { "type": "number" },
            "includeDrums": { "type": "boolean" },
            "beforeLabel": { "type": "string" },
            "afterLabel": { "type": "string" },
            "html": { "type": "string" }
        }
    })
}

pub fn choice_audio_effect_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["stepLevels", "cutoffStartHz", "cutoffEndHz", "accentStep", "nudgeMs", "gainScale", "pitchShiftSemis", "wetEffect"],
        "properties": {
            "stepLevels": { "type": "array", "items": { "type": "number" } },
            "cutoffStartHz": { "type": "number" },
            "cutoffEndHz": { "type": "number" },
            "accentStep": { "type": "integer" },
            "nudgeMs": { "type": "number" },
            "gainScale": { "type": "number" },
            "pitchShiftSemis": { "type": "number" },
            "wetEffect": { "type": "string", "enum": ["default", "dry", "wet"] }
        }
    })
}

pub fn question_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["goal", "primer", "hint", "media", "prompt", "correctAnswer", "correctAudioEffect", "distractors", "explanation", "angleSummary"],
        "properties": {
            "goal": { "type": "string" },
            "primer": { "type": "string" },
            "hint": { "type": "string" },
            "media": media_object_schema(),
            "prompt": { "type": "string" },
            "correctAnswer": { "type": "string" },
            "correctAudioEffect": choice_audio_effect_schema(),
            "distractors": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["text", "trapKind", "trapTarget", "whyWrong", "audioEffect"],
                    "properties": {
                        "text": { "type": "string" },
                        "trapKind": {
                            "type": "string",
                            "enum": ["confuses_concept", "misconception", "careless"]
                        },
                        "trapTarget": { "type": "string" },
                        "whyWrong": { "type": "string" },
                        "audioEffect": choice_audio_effect_schema()
                    }
                }
            },
            "explanation": { "type": "string" },
            "angleSummary": { "type": "string" }
        }
    })
}

pub fn clarifying_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["answer", "hasKnowledgeGap", "conceptTitle", "conceptSubtopic", "conceptSummary", "conceptDepth", "gapNote"],
        "properties": {
            "answer": { "type": "string" },
            "hasKnowledgeGap": { "type": "boolean" },
            "conceptTitle": { "type": "string" },
            "conceptSubtopic": { "type": "string" },
            "conceptSummary": { "type": "string" },
            "conceptDepth": { "type": "integer", "minimum": 1, "maximum": 4 },
            "gapNote": { "type": "string" }
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
    #[serde(default)]
    requires: Vec<String>,
}

fn default_depth() -> u8 {
    1
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawDistractor {
    Tagged {
        text: String,
        #[serde(rename = "trapKind", default)]
        trap_kind: Option<String>,
        #[serde(rename = "trapTarget", default)]
        trap_target: Option<String>,
        #[serde(rename = "whyWrong", default)]
        why_wrong: Option<String>,
        #[serde(rename = "audioEffect", default)]
        audio_effect: Option<RawChoiceAudioEffect>,
    },
    Plain(String),
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawChoiceAudioEffect {
    #[serde(default)]
    step_levels: Vec<f32>,
    #[serde(default)]
    cutoff_start_hz: f32,
    #[serde(default)]
    cutoff_end_hz: f32,
    #[serde(default)]
    accent_step: i32,
    #[serde(default)]
    nudge_ms: f32,
    #[serde(default)]
    gain_scale: f32,
    #[serde(default)]
    pitch_shift_semis: f32,
    #[serde(default)]
    wet_effect: String,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawMedia {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    reveal: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    caption: String,
    #[serde(default)]
    values: Vec<f32>,
    #[serde(default)]
    after_values: Vec<f32>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    flow_steps: Vec<FlowStep>,
    #[serde(default)]
    left_title: String,
    #[serde(default)]
    left_points: Vec<String>,
    #[serde(default)]
    right_title: String,
    #[serde(default)]
    right_points: Vec<String>,
    #[serde(default)]
    voice: String,
    #[serde(default)]
    bpm: f32,
    #[serde(default)]
    base_hz: f32,
    #[serde(default)]
    cutoff_start_hz: f32,
    #[serde(default)]
    cutoff_end_hz: f32,
    #[serde(default)]
    accent_step: i32,
    #[serde(default)]
    nudge_ms: f32,
    #[serde(default)]
    include_drums: bool,
    #[serde(default)]
    before_label: String,
    #[serde(default)]
    after_label: String,
    #[serde(default)]
    html: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawQuestion {
    #[serde(default)]
    goal: String,
    #[serde(default)]
    primer: String,
    #[serde(default)]
    hint: String,
    #[serde(default)]
    media: Option<RawMedia>,
    prompt: String,
    correct_answer: String,
    #[serde(default)]
    correct_audio_effect: Option<RawChoiceAudioEffect>,
    distractors: Vec<RawDistractor>,
    explanation: String,
    #[serde(default)]
    angle_summary: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawClarifying {
    answer: String,
    #[serde(default)]
    has_knowledge_gap: bool,
    #[serde(default)]
    concept_title: String,
    #[serde(default)]
    concept_subtopic: String,
    #[serde(default)]
    concept_summary: String,
    #[serde(default = "default_depth")]
    concept_depth: u8,
    #[serde(default)]
    gap_note: String,
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
            requires: item
                .requires
                .into_iter()
                .map(|r| r.trim().to_string())
                .filter(|r| !r.is_empty())
                .take(3)
                .collect(),
        });
    }
    if out.is_empty() {
        return Err(anyhow!("AI returned an empty concept map."));
    }
    Ok(out)
}

fn parse_trap_kind(raw: Option<&str>) -> Option<DistractorTrapKind> {
    match raw?.trim().to_lowercase().as_str() {
        "confuses_concept" => Some(DistractorTrapKind::ConfusesConcept),
        "misconception" => Some(DistractorTrapKind::Misconception),
        "careless" => Some(DistractorTrapKind::Careless),
        _ => None,
    }
}

fn convert_raw_choice_audio(raw: Option<RawChoiceAudioEffect>) -> Option<ChoiceAudioEffect> {
    let r = raw?;
    let wet_effect = match r.wet_effect.trim().to_lowercase().as_str() {
        "wet" => Some(true),
        "dry" => Some(false),
        _ => None,
    };
    let accent_step = if (1..=16).contains(&r.accent_step) {
        Some(r.accent_step as u8)
    } else {
        None
    };
    let has_any = !r.step_levels.is_empty()
        || r.cutoff_start_hz > 60.0
        || r.cutoff_end_hz > 60.0
        || accent_step.is_some()
        || r.nudge_ms.abs() > 0.1
        || (r.gain_scale > 0.05 && (r.gain_scale - 1.0).abs() > 0.05)
        || r.pitch_shift_semis.abs() > 0.05
        || wet_effect.is_some();
    if !has_any {
        return None;
    }
    Some(ChoiceAudioEffect {
        step_levels: r.step_levels.into_iter().take(16).map(|v| v.clamp(0.0, 1.0)).collect(),
        cutoff_start_hz: r.cutoff_start_hz,
        cutoff_end_hz: r.cutoff_end_hz,
        accent_step,
        nudge_ms: r.nudge_ms,
        gain_scale: r.gain_scale,
        pitch_shift_semis: r.pitch_shift_semis,
        wet_effect,
    })
}

pub fn parse_and_shuffle_question(val: Value) -> Result<GeneratedQuestion> {
    let raw: RawQuestion = serde_json::from_value(val).context("invalid question JSON")?;
    let prompt = raw.prompt.trim().to_string();
    let correct = raw.correct_answer.trim().to_string();
    let explanation = raw.explanation.trim().to_string();
    if prompt.is_empty() || correct.is_empty() || explanation.is_empty() {
        return Err(anyhow!("Question JSON missing prompt, correctAnswer, or explanation."));
    }

    let correct_audio = convert_raw_choice_audio(raw.correct_audio_effect);

    let mut seen = HashSet::new();
    seen.insert(correct.to_lowercase());
    let mut distractors: Vec<(
        String,
        Option<DistractorTrapKind>,
        Option<String>,
        Option<String>,
        Option<ChoiceAudioEffect>,
    )> = Vec::new();

    for d in raw.distractors {
        let (text, kind, target, why_wrong, audio_effect) = match d {
            RawDistractor::Plain(s) => (s, None, None, None, None),
            RawDistractor::Tagged {
                text,
                trap_kind,
                trap_target,
                why_wrong,
                audio_effect,
            } => (
                text,
                parse_trap_kind(trap_kind.as_deref()),
                trap_target
                    .map(|t| t.trim().chars().take(120).collect::<String>())
                    .filter(|t| !t.is_empty()),
                why_wrong
                    .map(|w| w.trim().chars().take(260).collect::<String>())
                    .filter(|w| !w.is_empty()),
                convert_raw_choice_audio(audio_effect),
            ),
        };
        let t = text.trim().to_string();
        if !t.is_empty() && seen.insert(t.to_lowercase()) {
            distractors.push((t, kind, target, why_wrong, audio_effect));
        }
    }
    if distractors.len() < 3 {
        return Err(anyhow!("Question JSON must provide 3 distinct distractors."));
    }

    let mut pool: Vec<(
        String,
        bool,
        Option<DistractorTrapKind>,
        Option<String>,
        Option<String>,
        Option<ChoiceAudioEffect>,
    )> = vec![
        (correct, true, None, None, None, correct_audio),
        (
            distractors[0].0.clone(),
            false,
            distractors[0].1,
            distractors[0].2.clone(),
            distractors[0].3.clone(),
            distractors[0].4.clone(),
        ),
        (
            distractors[1].0.clone(),
            false,
            distractors[1].1,
            distractors[1].2.clone(),
            distractors[1].3.clone(),
            distractors[1].4.clone(),
        ),
        (
            distractors[2].0.clone(),
            false,
            distractors[2].1,
            distractors[2].2.clone(),
            distractors[2].3.clone(),
            distractors[2].4.clone(),
        ),
    ];
    let mut rng = rand::thread_rng();
    pool.shuffle(&mut rng);

    let labels = ["A", "B", "C", "D"];
    let mut correct_choice_id = "A".to_string();
    let mut choices = Vec::with_capacity(4);
    for (idx, (text, is_right, trap_kind, trap_target, why_wrong, audio_effect)) in
        pool.into_iter().enumerate()
    {
        let id = labels[idx].to_string();
        if is_right {
            correct_choice_id = id.clone();
        }
        choices.push(QuestionChoice {
            id,
            text,
            trap_kind,
            trap_target,
            why_wrong,
            audio_effect,
        });
    }

    let angle_summary = if raw.angle_summary.trim().is_empty() {
        prompt.chars().take(90).collect()
    } else {
        raw.angle_summary.trim().chars().take(120).collect()
    };

    let media = raw.media.and_then(convert_raw_media);

    Ok(GeneratedQuestion {
        goal: raw.goal.trim().chars().take(280).collect(),
        primer: raw.primer.trim().chars().take(320).collect(),
        hint: raw.hint.trim().chars().take(200).collect(),
        media,
        prompt,
        choices,
        correct_choice_id,
        explanation,
        angle_summary,
    })
}

fn convert_raw_media(m: RawMedia) -> Option<ExampleMedia> {
    let reveal = if m.reveal.trim().eq_ignore_ascii_case("after_answer") {
        MediaReveal::AfterAnswer
    } else {
        MediaReveal::BeforeAnswer
    };
    let title = m.title.trim().chars().take(90).collect::<String>();
    let caption = m.caption.trim().chars().take(240).collect::<String>();

    match m.kind.trim().to_lowercase().as_str() {
        "curve" if m.values.len() >= 3 => Some(ExampleMedia::Curve {
            title: if title.is_empty() { "Visual Curve".into() } else { title },
            caption,
            reveal,
            values: m.values.into_iter().take(24).map(|v| v.clamp(0.0, 1.0)).collect(),
            labels: m.labels.into_iter().take(24).collect(),
        }),
        "flow" if m.flow_steps.len() >= 2 => Some(ExampleMedia::Flow {
            title: if title.is_empty() { "Step Flow".into() } else { title },
            caption,
            reveal,
            steps: m.flow_steps.into_iter().take(6).collect(),
        }),
        "contrast" if !m.left_points.is_empty() && !m.right_points.is_empty() => {
            Some(ExampleMedia::Contrast {
                title: if title.is_empty() { "Side-by-Side Comparison".into() } else { title },
                caption,
                reveal,
                left_title: if m.left_title.trim().is_empty() { "Option A".into() } else { m.left_title.trim().to_string() },
                left_points: m.left_points.into_iter().take(4).collect(),
                right_title: if m.right_title.trim().is_empty() { "Option B".into() } else { m.right_title.trim().to_string() },
                right_points: m.right_points.into_iter().take(4).collect(),
            })
        }
        "audio" => Some(ExampleMedia::Audio {
            title: if title.is_empty() { "Audio Example".into() } else { title },
            caption,
            reveal,
            spec: AudioDemoSpec {
                voice: match m.voice.trim().to_lowercase().as_str() {
                    "rim_click" | "kick_bass" | "delay" | "synth" | "reverb_rumble" | "saturation" => {
                        m.voice.trim().to_lowercase()
                    }
                    _ => "synth".into(),
                },
                bpm: if m.bpm > 40.0 { m.bpm } else { 128.0 },
                base_hz: if m.base_hz > 30.0 { m.base_hz } else { 110.0 },
                cutoff_start_hz: if m.cutoff_start_hz > 60.0 { m.cutoff_start_hz } else { 350.0 },
                cutoff_end_hz: if m.cutoff_end_hz > 60.0 { m.cutoff_end_hz } else { 3200.0 },
                step_levels: m.values.into_iter().take(16).map(|v| v.clamp(0.0, 1.0)).collect(),
                after_step_levels: m.after_values.into_iter().take(16).map(|v| v.clamp(0.0, 1.0)).collect(),
                step_labels: m.labels.into_iter().take(16).collect(),
                accent_step: if (1..=16).contains(&m.accent_step) { Some(m.accent_step as u8) } else { None },
                nudge_ms: m.nudge_ms,
                include_drums: m.include_drums,
                before_label: if m.before_label.trim().is_empty() { None } else { Some(m.before_label.trim().to_string()) },
                after_label: if m.after_label.trim().is_empty() { None } else { Some(m.after_label.trim().to_string()) },
                gain_scale: 1.0,
                pitch_shift_semis: 0.0,
                force_wet: None,
            },
        }),
        "interactive_html" if m.html.trim().len() >= 20 => Some(ExampleMedia::InteractiveHtml {
            title: if title.is_empty() { "Interactive Sandbox".into() } else { title },
            caption,
            reveal,
            html: m.html,
        }),
        _ => None,
    }
}

pub fn parse_clarifying_analysis(val: Value, current_concept_title: &str) -> Result<ClarifyingAnalysis> {
    let raw: RawClarifying = serde_json::from_value(val).context("invalid clarifying JSON")?;
    let answer = raw.answer.trim().to_string();
    if answer.is_empty() {
        return Err(anyhow!("AI returned an empty clarifying answer."));
    }

    let missing_concept = if raw.has_knowledge_gap && !raw.concept_title.trim().is_empty() {
        Some(ConceptInput {
            title: raw.concept_title.trim().chars().take(100).collect(),
            subtopic: if raw.concept_subtopic.trim().is_empty() {
                "Foundations".into()
            } else {
                raw.concept_subtopic.trim().chars().take(60).collect()
            },
            summary: raw.concept_summary.trim().chars().take(220).collect(),
            depth: raw.concept_depth.clamp(1, 4),
            requires: vec![],
        })
    } else {
        None
    };

    let gap_note = {
        let n = raw.gap_note.trim();
        if n.is_empty() {
            missing_concept.as_ref().map(|c| {
                format!(
                    "Learner asked about '{}' while studying '{}'.",
                    c.title, current_concept_title
                )
            })
        } else {
            Some(n.chars().take(140).collect())
        }
    };

    Ok(ClarifyingAnalysis {
        answer,
        missing_concept,
        gap_note,
    })
}

pub fn build_concept_map_prompt(
    topic_title: &str,
    level: SkillLevel,
    existing_titles: &[String],
    existing_subtopics: &[String],
    min_depth: u8,
) -> String {
    let scope_note = if existing_titles.is_empty() {
        format!(
            "Create 12-16 concepts (depths 1 to 3) spread EVENLY across 5-6 DISTINCT major pillars/disciplines of \"{topic_title}\" (2-3 concepts per pillar, with at least one depth 1-2 entry concept in EVERY pillar) for a {} learner.\n\
             CRITICAL BREADTH RULE: Cover the FULL multidisciplinary scope of \"{topic_title}\" — NEVER cluster into just one narrow facet! For example, a music production topic MUST span distinct pillars such as Sound Design & Synthesis, Mixing & EQ/Dynamics, Low-End & Kick/Bass Engineering, Rhythm & Groove, Space/FX & Atmospheres, and Arrangement & Club Energy (adapt the 5-6 pillars naturally to whatever domain \"{topic_title}\" is).",
            level.as_str()
        )
    } else {
        let subtopic_list = if existing_subtopics.is_empty() {
            "none yet".to_string()
        } else {
            existing_subtopics.join(", ")
        };
        format!(
            "Existing subtopics/pillars in the map: {subtopic_list}.\n\
             Already covered concepts (DO NOT repeat these): {}.\n\
             Generate 6-10 NEW concepts.\n\
             CRITICAL BREADTH-FIRST AUDIT: First identify major pillars/disciplines of \"{topic_title}\" that are MISSING or underrepresented above (for example, in music production: Sound Design & Synthesis, Mixing & EQ, Compression & Dynamics, Reverb/Rumble/FX & Spatial Depth, Arrangement & Transitions, Mastering/Club Translation — tailored to \"{topic_title}\"). Create depth 1-2 entry concepts in those missing pillars first! Only add depth {min_depth}-4 deeper concepts when all major pillars of \"{topic_title}\" are already well covered.",
            existing_titles.join(", ")
        )
    };

    format!(
        "You are an expert adaptive tutor building a multidisciplinary Cognitive Mesh Knowledge Map for \"{topic_title}\".\n\
         {scope_note}\n\
         For depth 2-4 concepts, include 1-2 prerequisite concept titles in `requires` (empty array [] for depth 1).\n\
         Return ONLY a valid JSON object matching:\n\
         {{\"concepts\":[{{\"title\":\"...\",\"subtopic\":\"...\",\"summary\":\"1-sentence core mental model\",\"depth\":1,\"requires\":[]}}]}}\n\
         Keep titles concise (2-5 words), use 5-6 clean, distinct pillar names in `subtopic`, and make every summary crisp and accurate."
    )
}

pub fn build_question_prompt(directive: &SchedulerDirective) -> String {
    let intent_guidance = match directive.intent {
        QuestionIntent::Introduce => {
            "Introduce this frontier concept with an intuitive, concrete 4-choice question."
        }
        QuestionIntent::ReviewMistake => {
            "The learner previously missed this concept or triggered a confusion/misconception. Probe the SAME concept from a fresh angle or contrast scenario without reusing old wording."
        }
        QuestionIntent::Reinforce => {
            "Reinforce retention or resolve a high-tension prerequisite seam in a slightly different practical scenario."
        }
        QuestionIntent::DigDeeper => {
            "The learner has strong fundamentals. Ask a deeper trade-off, synthesis, or edge-case question at the target depth."
        }
    };

    let slice_json = serde_json::to_string(&directive.compressed_context.cognitive_slice)
        .unwrap_or_else(|_| "{}".into());

    let mut lines = vec![
        format!(
            "You are a live adaptive tutor for \"{}\" (learner level: {}).",
            directive.compressed_context.topic_title,
            directive.compressed_context.level.as_str()
        ),
        format!(
            "Target concept: \"{}\" (slug: {}, subtopic: {}, μ={:.2}, σ={:.2}, summary: {}).",
            directive.concept.title,
            directive.concept.slug,
            directive.concept.subtopic,
            directive.concept.mastery,
            directive.concept.uncertainty,
            directive.concept.summary
        ),
        format!(
            "Intent: {} (target depth {}/4). {intent_guidance}",
            directive.intent.as_str(),
            directive.target_depth
        ),
        format!("Cognitive Slice (~200-token mesh state): {slice_json}"),
    ];

    if let Some(contrast) = &directive.compressed_context.contrast_concept {
        lines.push(format!(
            "CRITICAL: Craft a contrast question that cleanly separates \"{}\" from \"{}\".",
            directive.concept.title, contrast
        ));
    }
    if let Some(hyp) = &directive.compressed_context.target_hypothesis {
        lines.push(format!(
            "CRITICAL: Design this diagnostic question to directly confirm or refute the misconception hypothesis: \"{}\".",
            hyp
        ));
    }
    if !directive.compressed_context.recent_angles.is_empty() {
        lines.push(format!(
            "Avoid repeating these recent angles: {}",
            directive.compressed_context.recent_angles.join(" | ")
        ));
    }

    let needs_primer = directive.concept.mastery < 0.55
        || matches!(
            directive.concept.status,
            crate::models::ConceptStatus::Unseen | crate::models::ConceptStatus::NeedsReview
        );

    lines.push("Pedagogical & Anti-Confusion Rules (CRITICAL):".into());
    lines.push(format!(
        "- DISCIPLINE FIDELITY (CRITICAL): Stay 100% faithful to the target concept's actual pillar (`{}`) and subject matter. NEVER twist a sound design, synthesis, mixing, EQ, compression, saturation, reverb/space, arrangement, or theory concept into a step-sequencer accent or rim-click timing puzzle!",
        directive.concept.subtopic
    ));
    lines.push("- Write in crystal-clear, concrete, everyday language matched to the learner's level. Briefly define any domain jargon inline.".into());
    lines.push("- `goal`: 1 crisp plain-English sentence explaining the HUMAN, EMOTIONAL, PERCEPTUAL, or PRACTICAL WHY first—why a real practitioner wants this effect in the first place.".into());
    if needs_primer {
        lines.push("- `primer`: REQUIRED here (learner is new to or struggling with this concept). Write 1-2 short, intuitive sentences TEACHING the core mental model BEFORE the question (e.g. 'High-pass filtering clears mud by cutting low frequencies from non-bass elements so the kick and sub have room.'). Teach the principle cleanly WITHOUT spoiling the exact answer letter.".into());
    } else {
        lines.push("- `primer`: Return empty string \"\" since the learner already has solid mastery and should recall unassisted.".into());
    }
    lines.push("- `prompt`: Keep it effortless to grok on the FIRST read! State ONE clean starting setup + ONE clear question. NEVER include decoy backstories (like 'you try a second loop') that confuse what the learner is editing.".into());
    lines.push("- EXPLICIT CONSTRAINT RULE (zero ambiguity): Every single constraint used to reject ANY distractor (e.g., 'without changing the melody', 'using EQ only', 'keeping the kick transient intact') MUST be stated explicitly inside `prompt`. Never penalize an option for breaking an unmentioned constraint!".into());
    lines.push("- CHOICES: Keep each of the 4 choices SHORT (<= 14 words), concrete, and parallel in structure, varying ONE clear dimension at a time. Exactly 1 unambiguously right answer and 3 clearly distinct distractors (never 'All of the above').".into());
    lines.push("- `hint`: 1 crisp sentence nudging the learner toward the mental model WITHOUT revealing the correct choice.".into());
    lines.push("- `media`: ACTIVELY include the BEST supporting example (`reveal: \"before_answer\"` by default!) matched to THIS concept's discipline—without spoiling which option A-D is correct:".into());
    lines.push("  * `audio`: Great for sonic concepts! Match `voice` to the discipline: `synth` (synthesis, timbre, filter cutoff/modulation), `kick_bass` (low-end balance, sidechain/ducking, sub clarity), `reverb_rumble` (dark club space, reverb tails, techno rumble), `saturation` (harmonics, overdrive, analog warmth/grit), `delay` (echo/dub space), and `rim_click` ONLY when the concept is genuinely about percussion/timing/accents. Put the ORIGINAL starting sound in `values` + `labels`, and TARGET in `afterValues`. Populate `correctAudioEffect` AND every distractor's `audioEffect` (`stepLevels`, `cutoffStartHz`/`cutoffEndHz`, `accentStep`, `nudgeMs`, `gainScale`, `pitchShiftSemis`, `wetEffect`: 'default'|'dry'|'wet') so picking ANY option A-D auditions what that edit sounds like! For non-audio questions, leave `audioEffect` fields empty/0/'default'.".into());
    lines.push("  * `curve`: 4-16 normalized values (0.0..1.0 with clear visual contrast) + concise labels — ideal for EQ frequency curves, ADSR envelopes, filter sweeps, compression gain reduction, arrangement energy arcs, or math plots.".into());
    lines.push("  * `flow`: 2-5 connected `flowSteps` (`label`, `note`) — ideal for mixing/synthesis signal chains (e.g. Osc -> Filter -> Saturation -> Send Reverb), routing buses, protocols, or causal pipelines.".into());
    lines.push("  * `contrast`: side-by-side comparison (`leftTitle`/`leftPoints` vs `rightTitle`/`rightPoints`) — ideal for mixing/sound-design trade-offs (e.g. Insert vs Send FX, Mono Sub vs Wide Highs, Subtractive vs FM).".into());
    lines.push("  * `interactive_html`: a compact self-contained offline HTML/SVG/Canvas/WebAudio interactive toy (dark `#040405`, white `#F4F4F6`, red `#E11D2E`, zero external CDNs) when a live interactive sandbox makes the concept click instantly.".into());
    lines.push("  * `none`: Use ONLY when a visual/audio/interactive example truly adds zero clarity.".into());
    lines.push("- For EACH distractor, provide `trapKind` ('confuses_concept' | 'misconception' | 'careless'), `trapTarget`, and `whyWrong` (1 crisp sentence explaining what picking THAT specific option would actually feel like or cause instead of the stated goal).".into());
    lines.push("- `explanation`: 3 clear sentences bridging mechanics to real impact: (1) why you want this move emotionally/practically, (2) how the right answer achieves it, and (3) how the alternatives differ.".into());
    lines.push("- `angleSummary`: 4-8 words describing what angle this question tested.".into());
    lines.push("Return ONLY valid JSON matching the schema.".into());
    lines.join("\n")
}

pub fn build_clarifying_prompt(
    topic_title: &str,
    level: SkillLevel,
    concept_title: &str,
    question_prompt: &str,
    learner_question: &str,
    existing_subtopics: &[String],
    already_answered: bool,
) -> String {
    let spoiler_rule = if already_answered {
        "The learner has already answered the quiz question, so you may reference the solution freely."
    } else {
        "IMPORTANT: The learner has NOT answered the quiz question yet. Explain the concept or term they asked about clearly WITHOUT revealing or hinting which multiple-choice option is correct."
    };
    let pillar_hint = if existing_subtopics.is_empty() {
        "Choose a broad discipline pillar name (not a 1-concept micro-label).".to_string()
    } else {
        format!(
            "Prefer reusing one of the topic's existing broad pillar names ({}) instead of inventing a narrow 1-concept subtopic.",
            existing_subtopics.join(", ")
        )
    };

    format!(
        "You are a helpful adaptive tutor teaching \"{topic_title}\" (learner level: {}).\n\
         Current quiz concept: \"{concept_title}\".\n\
         Current quiz question: \"{question_prompt}\"\n\
         Learner's clarifying question: \"{learner_question}\"\n\
         {spoiler_rule}\n\
         Tasks:\n\
         1. `answer`: Give a super clear, friendly 2-3 sentence plain-English explanation answering the learner's question specifically in the context of \"{topic_title}\".\n\
         2. Identify if this question reveals a missing foundational term or concept (`hasKnowledgeGap`: true/false) that we should add to their Cognitive Mesh so future multiple-choice questions teach and reinforce it (always tied back to \"{topic_title}\").\n\
         3. If `hasKnowledgeGap` is true, provide concise `conceptTitle` (2-4 words), `conceptSubtopic` ({pillar_hint}), `conceptSummary` (1 crisp sentence tied to \"{topic_title}\"), `conceptDepth` (usually 1 for foundational terms), and `gapNote` (1 short sentence summarizing what the learner was unsure about).\n\
         Return ONLY valid JSON matching the schema.",
        level.as_str()
    )
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
        existing_subtopics: &[String],
        min_depth: u8,
    ) -> Result<Vec<ConceptInput>> {
        if cfg.kind == RunnerKind::Demo {
            return Ok(demo_concepts(topic_title, existing_titles, min_depth));
        }
        let prompt = build_concept_map_prompt(
            topic_title,
            level,
            existing_titles,
            existing_subtopics,
            min_depth,
        );
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

    pub fn answer_learner_question(
        &self,
        topic_title: &str,
        level: SkillLevel,
        concept_title: &str,
        question_prompt: &str,
        learner_question: &str,
        existing_subtopics: &[String],
        already_answered: bool,
        cfg: &RunnerConfig,
    ) -> Result<ClarifyingAnalysis> {
        if cfg.kind == RunnerKind::Demo {
            let term = extract_demo_term(learner_question);
            return Ok(ClarifyingAnalysis {
                answer: format!(
                    "[Demo Tutor] In {topic_title}, \"{term}\" is a foundational idea connected to {concept_title}. Switch to Codex CLI, Antigravity CLI, or an API key in Settings for full live AI explanations."
                ),
                missing_concept: Some(ConceptInput {
                    title: term.clone(),
                    subtopic: existing_subtopics
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Foundations".into()),
                    summary: format!("Core meaning and role of {term} inside {topic_title}."),
                    depth: 1,
                    requires: vec![],
                }),
                gap_note: Some(format!("Asked: {}", learner_question.trim())),
            });
        }
        let prompt = build_clarifying_prompt(
            topic_title,
            level,
            concept_title,
            question_prompt,
            learner_question,
            existing_subtopics,
            already_answered,
        );
        let json_val = self.run_json_prompt(&prompt, &clarifying_schema(), cfg)?;
        parse_clarifying_analysis(json_val, concept_title)
    }

    /// Generates an AI-chosen supporting example (`audio`, `curve`, `flow`, `contrast`, or `interactive_html`)
    /// specifically tailored to help the learner understand an existing question's goal and setup.
    pub fn generate_supporting_media(
        &self,
        topic_title: &str,
        level: SkillLevel,
        concept_title: &str,
        goal: &str,
        question_prompt: &str,
        already_answered: bool,
        cfg: &RunnerConfig,
    ) -> Result<ExampleMedia> {
        if cfg.kind == RunnerKind::Demo {
            return Ok(ExampleMedia::Flow {
                title: format!("{concept_title} Mental Model"),
                caption: "Visual breakdown connecting the starting setup to the intended effect.".into(),
                reveal: MediaReveal::BeforeAnswer,
                steps: vec![
                    FlowStep { label: "Setup".into(), note: topic_title.to_string() },
                    FlowStep { label: concept_title.to_string(), note: "Key mechanism".into() },
                    FlowStep { label: "Felt Result".into(), note: "Target outcome".into() },
                ],
            });
        }

        let spoiler_rule = if already_answered {
            "The learner already answered, so you may illustrate the full before/after effect directly."
        } else {
            "The learner has NOT answered yet. Help them understand the goal, feel, or structure of the question WITHOUT spoiling which multiple-choice answer is right."
        };

        let prompt = format!(
            "You are designing ONE concrete supporting example to help a {} learner understand this question in \"{topic_title}\".\n\
             Concept: \"{concept_title}\"\n\
             Goal: \"{goal}\"\n\
             Question: \"{question_prompt}\"\n\
             {spoiler_rule}\n\
             Choose the SINGLE best modality (`audio`, `curve`, `flow`, `contrast`, or `interactive_html` — do NOT return `none`!) and set `reveal: \"before_answer\"`.\n\
             - Pick `audio` for music/sound/rhythm/acoustics so our built-in Rust synth lets them hear it live.\n\
             - Pick `curve` for 4-16 step grids, envelopes, sweeps, or math plots.\n\
             - Pick `flow` for signal chains, protocols, or step-by-step mechanisms.\n\
             - Pick `contrast` for comparing two setups/forces side-by-side.\n\
             - Pick `interactive_html` for a compact self-contained offline dark-mode (`#050506`, `#F4F4F6`, `#E11D2E`) HTML/SVG/Canvas/WebAudio mini-lab.\n\
             Return ONLY valid JSON matching the schema.",
            level.as_str()
        );

        let val = self.run_json_prompt(&prompt, &media_object_schema(), cfg)?;
        let raw: RawMedia = serde_json::from_value(val).context("invalid supporting media JSON")?;
        convert_raw_media(raw).ok_or_else(|| anyhow!("AI did not produce a renderable supporting example."))
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
            requires: vec![],
        },
        ConceptInput {
            title: "Core Primitives & Terms".into(),
            subtopic: "Foundations".into(),
            summary: format!("Key building blocks and terminology required to reason about {t}."),
            depth: 1,
            requires: vec![],
        },
        ConceptInput {
            title: "Step-by-Step Lifecycle".into(),
            subtopic: "Mechanics".into(),
            summary: format!("How inputs transform into outputs across a typical {t} flow."),
            depth: 1,
            requires: vec![],
        },
        ConceptInput {
            title: "Common Failure Modes".into(),
            subtopic: "Mechanics".into(),
            summary: format!("Frequent mistakes, pitfalls, and bottlenecks seen in {t}."),
            depth: 2,
            requires: vec!["Step-by-Step Lifecycle".into()],
        },
        ConceptInput {
            title: "Design & Shaping Craft".into(),
            subtopic: "Design & Synthesis".into(),
            summary: format!("How raw elements are sculpted and modulated into cohesive {t} output."),
            depth: 1,
            requires: vec![],
        },
        ConceptInput {
            title: "Balance & Clarity".into(),
            subtopic: "Tuning & Mixing".into(),
            summary: format!("How competing layers are separated and balanced for clean {t} results."),
            depth: 1,
            requires: vec![],
        },
        ConceptInput {
            title: "Depth & Spatial Context".into(),
            subtopic: "Space & Atmosphere".into(),
            summary: format!("How environment, layering, and subtle movement add depth in {t}."),
            depth: 2,
            requires: vec!["Design & Shaping Craft".into()],
        },
        ConceptInput {
            title: "Trade-offs & Constraints".into(),
            subtopic: "Architecture".into(),
            summary: format!("Balancing speed, simplicity, and correctness in {t}."),
            depth: 2,
            requires: vec![format!("{t} Mental Model")],
        },
        ConceptInput {
            title: "Applied Patterns".into(),
            subtopic: "Practice".into(),
            summary: format!("Proven ways experts structure real-world {t} solutions."),
            depth: 3,
            requires: vec!["Trade-offs & Constraints".into()],
        },
        ConceptInput {
            title: "Diagnostics & Verification".into(),
            subtopic: "Tuning & Mixing".into(),
            summary: format!("How to observe, measure, and debug {t} systematically."),
            depth: 2,
            requires: vec!["Balance & Clarity".into()],
        },
        ConceptInput {
            title: "Subtle Edge Cases".into(),
            subtopic: "Mastery".into(),
            summary: format!("Non-obvious interactions and boundary conditions in {t}."),
            depth: 4,
            requires: vec!["Applied Patterns".into(), "Diagnostics & Verification".into()],
        },
    ];
    let existing_lower: HashSet<String> = existing.iter().map(|s| s.to_lowercase()).collect();
    templates
        .into_iter()
        .filter(|c| !existing_lower.contains(&c.title.to_lowercase()) && (existing.is_empty() || c.depth >= min_depth || c.depth <= 2))
        .collect()
}

fn extract_demo_term(question: &str) -> String {
    let cleaned = question
        .trim()
        .trim_end_matches('?')
        .trim()
        .to_string();
    let lower = cleaned.to_lowercase();
    for prefix in ["what is a ", "what is an ", "what is ", "what does ", "explain ", "define "] {
        if lower.starts_with(prefix) {
            let rest = cleaned[prefix.len()..].trim();
            if !rest.is_empty() {
                return rest.chars().take(48).collect();
            }
        }
    }
    if cleaned.is_empty() {
        "Clarified Concept".into()
    } else {
        cleaned.chars().take(48).collect()
    }
}

fn demo_question(directive: &SchedulerDirective) -> Value {
    let tag = match directive.intent {
        QuestionIntent::Introduce => "Demo · Frontier Probe",
        QuestionIntent::ReviewMistake => "Demo · Contrast / Retry",
        QuestionIntent::Reinforce => "Demo · Seam Check",
        QuestionIntent::DigDeeper => "Demo · Deep Dive",
    };
    json!({
        "goal": format!("Build a clear mental model of how \"{}\" serves real outcomes in {}.", directive.concept.title, directive.compressed_context.topic_title),
        "primer": format!("Core idea: {}", directive.concept.summary),
        "hint": format!("Focus on the core purpose of \"{}\" rather than surface shortcuts.", directive.concept.title),
        "media": {
            "kind": "flow",
            "reveal": "before_answer",
            "title": format!("{} Flow", directive.concept.title),
            "caption": "How the core mental model connects input, mechanism, and outcome.",
            "values": [],
            "afterValues": [],
            "labels": [],
            "flowSteps": [
                { "label": "Context", "note": directive.compressed_context.topic_title },
                { "label": directive.concept.title, "note": "Core mechanism" },
                { "label": "Intended Effect", "note": "Predictable result" }
            ],
            "leftTitle": "",
            "leftPoints": [],
            "rightTitle": "",
            "rightPoints": [],
            "voice": "synth",
            "bpm": 0,
            "baseHz": 0,
            "cutoffStartHz": 0,
            "cutoffEndHz": 0,
            "accentStep": 0,
            "nudgeMs": 0,
            "includeDrums": false,
            "beforeLabel": "",
            "afterLabel": "",
            "html": ""
        },
        "prompt": format!("[{tag}] In {}, which statement best captures \"{}\"?", directive.compressed_context.topic_title, directive.concept.title),
        "correctAnswer": directive.concept.summary,
        "distractors": [
            {
                "text": format!("It eliminates all trade-offs across every subtopic of {}.", directive.compressed_context.topic_title),
                "trapKind": "misconception",
                "trapTarget": format!("Assumes {} has no trade-offs", directive.concept.title),
                "whyWrong": "Real systems always involve trade-offs; treating this concept as a magic cure misses its actual purpose."
            },
            {
                "text": format!("It is interchangeable with Core Primitives & Terms in {}.", directive.compressed_context.topic_title),
                "trapKind": "confuses_concept",
                "trapTarget": "core-primitives-terms",
                "whyWrong": "That confuses this specific mechanism with general vocabulary rather than its distinct role."
            },
            {
                "text": "It only matters when real-world feedback and constraints are ignored.",
                "trapKind": "careless",
                "trapTarget": "surface_distractor",
                "whyWrong": "In practice, real-world constraints and feedback are exactly why this concept exists."
            }
        ],
        "explanation": format!("Select Codex CLI, Antigravity CLI, or an API key in Settings for live AI-authored questions. Core idea for \"{}\": {}", directive.concept.title, directive.concept.summary),
        "angleSummary": format!("{} ({:?} #{})", directive.concept.title, directive.intent, directive.concept.attempts + 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CognitiveSlice, CompressedContext, ConceptNode, ConceptStatus};

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
                    r#"{"prompt":"Why does Rust enforce one mutable borrow?","correctAnswer":"To prevent data races and aliasing bugs at compile time.","distractors":[{"text":"To make binaries smaller.","trapKind":"careless","trapTarget":"size_myth"},{"text":"Because lifetimes free memory early.","trapKind":"misconception","trapTarget":"lifetimes change runtime drop order"},{"text":"Because Copy and Borrow are identical.","trapKind":"confuses_concept","trapTarget":"copy-trait"}],"explanation":"Exclusive mutability prevents aliasing hazards before code runs.","angleSummary":"aliasing xor mutability"}"#,
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
    fn detects_codex_and_parses_tagged_distractors_safely() {
        let runner = AiRunner::with_executor(Box::new(FakeExec));
        let detected = runner.detect_runners("", "");
        let codex = detected.iter().find(|r| r.kind == RunnerKind::Codex).unwrap();
        assert!(codex.available);

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
                mastery: 0.50,
                uncertainty: 0.90,
                stability: 0.0,
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
                cognitive_slice: CognitiveSlice::default(),
                contrast_concept: None,
                target_hypothesis: None,
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
        assert!(q.choices.iter().any(|c| c.trap_kind == Some(DistractorTrapKind::Misconception)));
        assert!(q.choices.iter().any(|c| c.trap_kind == Some(DistractorTrapKind::ConfusesConcept)));
    }
}
