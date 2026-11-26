use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
}

impl SkillLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SkillLevel::Beginner => "beginner",
            SkillLevel::Intermediate => "intermediate",
            SkillLevel::Advanced => "advanced",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            SkillLevel::Beginner => "Beginner",
            SkillLevel::Intermediate => "Intermediate",
            SkillLevel::Advanced => "Advanced",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "intermediate" => SkillLevel::Intermediate,
            "advanced" => SkillLevel::Advanced,
            _ => SkillLevel::Beginner,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunnerKind {
    Codex,
    Agy,
    GeminiCli,
    ClaudeCli,
    GeminiApi,
    OpenaiApi,
    Demo,
}

impl RunnerKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunnerKind::Codex => "codex",
            RunnerKind::Agy => "agy",
            RunnerKind::GeminiCli => "gemini-cli",
            RunnerKind::ClaudeCli => "claude-cli",
            RunnerKind::GeminiApi => "gemini-api",
            RunnerKind::OpenaiApi => "openai-api",
            RunnerKind::Demo => "demo",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            RunnerKind::Codex => "Codex CLI",
            RunnerKind::Agy => "Antigravity CLI",
            RunnerKind::GeminiCli => "Gemini CLI",
            RunnerKind::ClaudeCli => "Claude CLI",
            RunnerKind::GeminiApi => "Gemini API",
            RunnerKind::OpenaiApi => "OpenAI API",
            RunnerKind::Demo => "Offline Demo",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "codex" => RunnerKind::Codex,
            "agy" => RunnerKind::Agy,
            "gemini-cli" => RunnerKind::GeminiCli,
            "claude-cli" => RunnerKind::ClaudeCli,
            "gemini-api" => RunnerKind::GeminiApi,
            "openai-api" => RunnerKind::OpenaiApi,
            _ => RunnerKind::Demo,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectedRunner {
    pub kind: RunnerKind,
    pub label: String,
    pub is_cli: bool,
    pub available: bool,
    pub command: Option<String>,
    pub version: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionIntent {
    ReviewMistake,
    Reinforce,
    Introduce,
    DigDeeper,
}

impl QuestionIntent {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuestionIntent::ReviewMistake => "review_mistake",
            QuestionIntent::Reinforce => "reinforce",
            QuestionIntent::Introduce => "introduce",
            QuestionIntent::DigDeeper => "dig_deeper",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            QuestionIntent::ReviewMistake => "Mistake / Contrast",
            QuestionIntent::Reinforce => "Spaced / Seam Check",
            QuestionIntent::Introduce => "Frontier Probe",
            QuestionIntent::DigDeeper => "Deep Dive",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "review_mistake" => QuestionIntent::ReviewMistake,
            "reinforce" => QuestionIntent::Reinforce,
            "dig_deeper" => QuestionIntent::DigDeeper,
            _ => QuestionIntent::Introduce,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConceptStatus {
    Unseen,
    Learning,
    NeedsReview,
    Mastered,
    Deepened,
}

impl ConceptStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConceptStatus::Unseen => "unseen",
            ConceptStatus::Learning => "learning",
            ConceptStatus::NeedsReview => "needs_review",
            ConceptStatus::Mastered => "mastered",
            ConceptStatus::Deepened => "deepened",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConceptStatus::Unseen => "Unseen",
            ConceptStatus::Learning => "Learning",
            ConceptStatus::NeedsReview => "Needs Review",
            ConceptStatus::Mastered => "Mastered",
            ConceptStatus::Deepened => "Deepened",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "learning" => ConceptStatus::Learning,
            "needs_review" => ConceptStatus::NeedsReview,
            "mastered" => ConceptStatus::Mastered,
            "deepened" => ConceptStatus::Deepened,
            _ => ConceptStatus::Unseen,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Excitatory prerequisite (+w): mastering B propagates credit back to A; failing A raises tension on B.
    Requires,
    /// Excitatory analogy/transfer (+w): bridges a mastered anchor (possibly from another topic) to a target concept.
    Bridges,
    /// Inhibitory confusion synapse (-w): learner mixes up A and B; prompts contrast questions.
    Confuses,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::Requires => "requires",
            EdgeKind::Bridges => "bridges",
            EdgeKind::Confuses => "confuses",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "bridges" => EdgeKind::Bridges,
            "confuses" => EdgeKind::Confuses,
            _ => EdgeKind::Requires,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEdge {
    pub id: String,
    pub topic_id: String,
    /// Concept slug (or external topic::concept label for cross-topic Bridges)
    pub from_slug: String,
    pub from_label: String,
    pub to_slug: String,
    pub to_label: String,
    pub kind: EdgeKind,
    /// Positive magnitude in [0.0, 1.0]; semantics are signed by `kind` (Confuses = inhibitory).
    pub weight: f64,
    /// Predictive-coding tension in [0.0, 1.0] computed on Requires edges.
    pub tension: f64,
    pub note: String,
}

/// Shadow Node representing a latent misconception / mental model hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisconceptionNode {
    pub id: String,
    pub topic_id: String,
    pub slug: String,
    pub label: String,
    /// Posterior probability in [0.0, 1.0]. Pruned when < 0.15 or refuted twice.
    pub probability: f64,
    pub evidence_count: u32,
    pub refute_count: u32,
    /// Concept slugs this shadow node explains errors on.
    pub linked_concepts: Vec<String>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistractorTrapKind {
    ConfusesConcept,
    Misconception,
    Careless,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub runner: RunnerKind,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub openai_api_key: String,
    pub openai_model: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSummary {
    pub id: String,
    pub user_id: String,
    pub title: String,
    pub level: SkillLevel,
    pub concept_count: usize,
    pub mastered_count: usize,
    pub average_mastery: f64,
    pub total_attempts: u32,
    pub correct_attempts: u32,
    pub current_streak: u32,
    pub best_streak: u32,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptInput {
    pub title: String,
    pub subtopic: String,
    pub summary: String,
    pub depth: u8,
    /// Prerequisite concept titles/slugs (`Requires` edges seeded into the mesh).
    #[serde(default)]
    pub requires: Vec<String>,
}

/// Compressed per-concept node carrying the Cognitive Mesh `(μ, σ, S)` state tuple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub topic_id: String,
    pub slug: String,
    pub title: String,
    pub subtopic: String,
    pub summary: String,
    pub depth: u8,
    /// μ — estimated mastery in [0.0, 1.0]. Unseen starts at 0.50 with high σ=0.90.
    pub mastery: f64,
    /// σ — epistemic uncertainty in [0.08, 1.0]. Shrinks as observations accumulate.
    pub uncertainty: f64,
    /// S — memory stability in hours (grows with successful spaced & associative recall).
    pub stability: f64,
    pub ease_factor: f64,
    pub interval_hours: f64,
    pub attempts: u32,
    pub correct_count: u32,
    pub wrong_count: u32,
    pub consecutive_correct: u32,
    pub status: ConceptStatus,
    /// Capped to 3 concise misconception summaries (<= 140 chars), pruned once mastered.
    pub misconceptions: Vec<String>,
    /// Capped to 4 short angles already tested so future questions rephrase cleanly.
    pub recent_angles: Vec<String>,
    pub last_seen_at: Option<i64>,
    pub next_review_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionChoice {
    pub id: String, // "A" | "B" | "C" | "D"
    pub text: String,
    /// Pre-tagged distractor trap metadata so local mesh propagation needs zero extra LLM calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trap_kind: Option<DistractorTrapKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trap_target: Option<String>,
    /// Specific reason why choosing this distractor fails or creates the wrong effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_wrong: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredQuestion {
    pub id: String,
    pub topic_id: String,
    pub concept_id: String,
    pub concept_title: String,
    pub subtopic: String,
    pub depth: u8,
    pub intent: QuestionIntent,
    /// Plain-English framing of the goal/purpose before the question prompt.
    #[serde(default)]
    pub goal: String,
    pub prompt: String,
    pub choices: Vec<QuestionChoice>,
    pub correct_choice_id: String,
    pub explanation: String,
    pub angle_summary: String,
    pub source: RunnerKind,
    pub created_at: i64,
    pub answered_at: Option<i64>,
    pub selected_choice_id: Option<String>,
    pub is_correct: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerResult {
    pub question_id: String,
    pub is_correct: bool,
    pub selected_choice_id: String,
    pub correct_choice_id: String,
    pub correct_choice_text: String,
    /// Tailored reason why the user's specific wrong choice falls short (when wrong).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_why_wrong: Option<String>,
    pub explanation: String,
    pub reward_message: String,
    pub concept: ConceptNode,
    pub topic_streak: u32,
    pub best_streak: u32,
    pub mastery_delta: f64,
    /// Human-readable summary of Cognitive Mesh propagation effects (backprop, confusion, shadow nodes).
    #[serde(default)]
    pub mesh_effects: Vec<String>,
}

/// Cross-topic mastered concept anchor for analogy bridges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossTopicAnchor {
    pub topic_title: String,
    pub concept_slug: String,
    pub concept_title: String,
    pub mastery: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierEntry {
    pub node: String,
    pub mu: f64,
    pub sigma: f64,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionSeam {
    pub edge: String,
    pub tension: f64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisBrief {
    pub misconception: String,
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionPairBrief {
    pub pair: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeBrief {
    pub from: String,
    pub to: String,
    pub from_mu: f64,
}

/// Compact ~150-250 token multi-resolution Cognitive Slice serialized for the AI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveSlice {
    pub macro_map: Vec<String>,
    pub frontier: Vec<FrontierEntry>,
    pub tension_seams: Vec<TensionSeam>,
    pub active_hypotheses: Vec<HypothesisBrief>,
    pub confusion_pairs: Vec<ConfusionPairBrief>,
    pub strong_bridges: Vec<BridgeBrief>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrugglingSummary {
    pub title: String,
    pub misconceptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedContext {
    pub topic_title: String,
    pub level: SkillLevel,
    pub mastered_concepts: Vec<String>,
    pub struggling_concepts: Vec<StrugglingSummary>,
    pub recent_angles: Vec<String>,
    /// Multi-resolution Cognitive Mesh slice (~200 tokens)
    #[serde(default)]
    pub cognitive_slice: CognitiveSlice,
    /// Optional concept slug to contrast against when targeting an inhibitory `Confuses` edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast_concept: Option<String>,
    /// Optional misconception hypothesis to directly confirm/refute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_hypothesis: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerDirective {
    pub intent: QuestionIntent,
    pub concept: ConceptNode,
    pub target_depth: u8,
    pub reason: String,
    pub requires_new_concepts: bool,
    pub compressed_context: CompressedContext,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeStats {
    pub unseen: usize,
    pub learning: usize,
    pub needs_review: usize,
    pub mastered: usize,
    pub deepened: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMap {
    pub topic: TopicSummary,
    pub concepts: Vec<ConceptNode>,
    pub edges: Vec<ConceptEdge>,
    pub misconceptions: Vec<MisconceptionNode>,
    pub cognitive_slice: CognitiveSlice,
    pub stats: KnowledgeStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarifyingReply {
    pub question_asked: String,
    pub answer: String,
    pub added_concept_title: Option<String>,
    pub mesh_note: Option<String>,
}
