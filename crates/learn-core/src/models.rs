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
            QuestionIntent::ReviewMistake => "Mistake Retry",
            QuestionIntent::Reinforce => "Spaced Review",
            QuestionIntent::Introduce => "New Concept",
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
}

/// Compressed per-concept mastery record persisted in SQLite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub topic_id: String,
    pub slug: String,
    pub title: String,
    pub subtopic: String,
    pub summary: String,
    pub depth: u8,
    pub mastery: f64,
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
    pub explanation: String,
    pub reward_message: String,
    pub concept: ConceptNode,
    pub topic_streak: u32,
    pub best_streak: u32,
    pub mastery_delta: f64,
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
    pub stats: KnowledgeStats,
}
