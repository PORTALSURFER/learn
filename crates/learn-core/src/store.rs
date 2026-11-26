use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::mesh::{DEFAULT_UNSEEN_MU, DEFAULT_UNSEEN_SIGMA, slugify_label};
use crate::models::{
    ConceptEdge, ConceptInput, ConceptNode, ConceptStatus, CrossTopicAnchor, EdgeKind,
    MisconceptionNode, QuestionChoice, QuestionIntent, RunnerKind, SkillLevel, StoredQuestion,
    TopicSummary, User,
};

const MAX_MISCONCEPTIONS: usize = 3;
const MAX_MISCONCEPTION_CHARS: usize = 140;
const MAX_RECENT_ANGLES: usize = 4;
const MAX_ANGLE_CHARS: usize = 120;
const MAX_ANSWERED_QUESTIONS_PER_TOPIC: i64 = 10;

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn slugify(input: &str) -> String {
    let clean = slugify_label(input);
    if clean.is_empty() {
        format!("concept-{}", &Uuid::new_v4().to_string()[..8])
    } else {
        clean
    }
}

pub fn clamp_list(items: &[String], max_items: usize, max_chars: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for raw in items {
        let normalized: String = raw
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(max_chars)
            .collect();
        if normalized.is_empty() {
            continue;
        }
        let key = normalized.to_lowercase();
        if seen.insert(key) {
            out.push(normalized);
        }
    }
    if out.len() > max_items {
        out.drain(0..(out.len() - max_items));
    }
    out
}

pub fn hash_password(password: &str, salt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(b":");
    hasher.update(password.as_bytes());
    format!("sha256:{salt}:{:x}", hasher.finalize())
}

pub fn verify_password(password: &str, stored: &str) -> bool {
    let parts: Vec<&str> = stored.split(':').collect();
    if parts.len() != 3 || parts[0] != "sha256" {
        return false;
    }
    let computed = hash_password(password, parts[1]);
    computed == stored
}

pub struct CreateQuestionParams {
    pub topic_id: String,
    pub concept_id: String,
    pub intent: QuestionIntent,
    pub goal: String,
    pub prompt: String,
    pub choices: Vec<QuestionChoice>,
    pub correct_choice_id: String,
    pub explanation: String,
    pub angle_summary: String,
    pub source: RunnerKind,
}

pub struct KnowledgeStore {
    conn: Connection,
}

impl KnowledgeStore {
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("open in-memory sqlite")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    pub fn open_path(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create db directory {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("open sqlite db at {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        self.conn.pragma_update(None, "foreign_keys", "ON")?;
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                runner TEXT NOT NULL DEFAULT 'demo',
                gemini_api_key TEXT NOT NULL DEFAULT '',
                gemini_model TEXT NOT NULL DEFAULT 'gemini-2.5-flash',
                openai_api_key TEXT NOT NULL DEFAULT '',
                openai_model TEXT NOT NULL DEFAULT 'gpt-4o-mini',
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'beginner',
                total_attempts INTEGER NOT NULL DEFAULT 0,
                correct_attempts INTEGER NOT NULL DEFAULT 0,
                current_streak INTEGER NOT NULL DEFAULT 0,
                best_streak INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                slug TEXT NOT NULL,
                title TEXT NOT NULL,
                subtopic TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                depth INTEGER NOT NULL DEFAULT 1,
                mastery REAL NOT NULL DEFAULT 0.50,
                uncertainty REAL NOT NULL DEFAULT 0.90,
                stability REAL NOT NULL DEFAULT 0.0,
                ease_factor REAL NOT NULL DEFAULT 2.5,
                interval_hours REAL NOT NULL DEFAULT 0.0,
                attempts INTEGER NOT NULL DEFAULT 0,
                correct_count INTEGER NOT NULL DEFAULT 0,
                wrong_count INTEGER NOT NULL DEFAULT 0,
                consecutive_correct INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'unseen',
                misconceptions_json TEXT NOT NULL DEFAULT '[]',
                recent_angles_json TEXT NOT NULL DEFAULT '[]',
                last_seen_at INTEGER,
                next_review_at INTEGER,
                UNIQUE(topic_id, slug)
            );

            CREATE TABLE IF NOT EXISTS concept_edges (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                from_slug TEXT NOT NULL,
                from_label TEXT NOT NULL,
                to_slug TEXT NOT NULL,
                to_label TEXT NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.75,
                tension REAL NOT NULL DEFAULT 0.0,
                note TEXT NOT NULL DEFAULT '',
                UNIQUE(topic_id, from_slug, to_slug, kind)
            );

            CREATE TABLE IF NOT EXISTS misconception_nodes (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                slug TEXT NOT NULL,
                label TEXT NOT NULL,
                probability REAL NOT NULL DEFAULT 0.55,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                refute_count INTEGER NOT NULL DEFAULT 0,
                linked_concepts_json TEXT NOT NULL DEFAULT '[]',
                updated_at INTEGER NOT NULL,
                UNIQUE(topic_id, slug)
            );

            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                intent TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '',
                prompt TEXT NOT NULL,
                choices_json TEXT NOT NULL,
                correct_choice_id TEXT NOT NULL,
                explanation TEXT NOT NULL,
                angle_summary TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                answered_at INTEGER,
                selected_choice_id TEXT,
                is_correct INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_concepts_topic ON concepts(topic_id, depth, mastery);
            CREATE INDEX IF NOT EXISTS idx_edges_topic ON concept_edges(topic_id, kind, tension);
            CREATE INDEX IF NOT EXISTS idx_misconceptions_topic ON misconception_nodes(topic_id, probability);
            CREATE INDEX IF NOT EXISTS idx_questions_open ON questions(topic_id, answered_at, created_at);
            "#,
        )?;

        // Additive migration for databases created before the Cognitive Mesh upgrade:
        let _ = self
            .conn
            .execute("ALTER TABLE concepts ADD COLUMN uncertainty REAL NOT NULL DEFAULT 0.90", []);
        let _ = self
            .conn
            .execute("ALTER TABLE concepts ADD COLUMN stability REAL NOT NULL DEFAULT 0.0", []);
        let _ = self
            .conn
            .execute("ALTER TABLE questions ADD COLUMN goal TEXT NOT NULL DEFAULT ''", []);
        // Clear legacy unanswered buffered questions created before `goal` / `whyWrong` prompts existed
        // so users immediately receive the upgraded question format after updating.
        let _ = self
            .conn
            .execute("DELETE FROM questions WHERE answered_at IS NULL AND goal = ''", []);
        // Normalize legacy unseen rows (mastery=0.0, attempts=0) to the neutral Bayesian prior (0.50, σ=0.90)
        let _ = self.conn.execute(
            "UPDATE concepts SET mastery = ?1, uncertainty = ?2 WHERE attempts = 0 AND mastery = 0.0",
            params![DEFAULT_UNSEEN_MU, DEFAULT_UNSEEN_SIGMA],
        );
        Ok(())
    }

    // ---------- Users ----------

    pub fn create_user(&self, username: &str, password: &str, default_runner: RunnerKind) -> Result<User> {
        let clean = username.trim();
        if clean.len() < 2 || clean.len() > 40 {
            return Err(anyhow!("Username must be between 2 and 40 characters."));
        }
        if password.len() < 4 {
            return Err(anyhow!("Password must be at least 4 characters."));
        }
        if self.get_user_by_username(clean)?.is_some() {
            return Err(anyhow!("Username '{clean}' already exists."));
        }

        let id = Uuid::new_v4().to_string();
        let salt = Uuid::new_v4().to_string();
        let password_hash = hash_password(password, &salt);
        let created_at = now_ms();

        self.conn.execute(
            r#"
            INSERT INTO users (id, username, password_hash, runner, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![id, clean, password_hash, default_runner.as_str(), created_at],
        )?;

        self.get_user_by_id(&id)?
            .ok_or_else(|| anyhow!("failed to load created user"))
    }

    pub fn authenticate_user(&self, username: &str, password: &str) -> Result<User> {
        let row: Option<(String, String)> = self
            .conn
            .query_row(
                "SELECT id, password_hash FROM users WHERE username = ?1 COLLATE NOCASE",
                params![username.trim()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;

        let Some((id, hash)) = row else {
            return Err(anyhow!("Invalid username or password."));
        };

        if !verify_password(password, &hash) {
            return Err(anyhow!("Invalid username or password."));
        }

        self.get_user_by_id(&id)?
            .ok_or_else(|| anyhow!("user not found"))
    }

    pub fn list_usernames(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT username FROM users ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn get_user_by_id(&self, id: &str) -> Result<Option<User>> {
        self.conn
            .query_row(
                r#"
                SELECT id, username, runner, gemini_api_key, gemini_model, openai_api_key, openai_model, created_at
                FROM users WHERE id = ?1
                "#,
                params![id],
                Self::map_user,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_user_by_username(&self, username: &str) -> Result<Option<User>> {
        self.conn
            .query_row(
                r#"
                SELECT id, username, runner, gemini_api_key, gemini_model, openai_api_key, openai_model, created_at
                FROM users WHERE username = ?1 COLLATE NOCASE
                "#,
                params![username.trim()],
                Self::map_user,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn update_user_runner(
        &self,
        user_id: &str,
        runner: RunnerKind,
        gemini_api_key: Option<&str>,
        openai_api_key: Option<&str>,
    ) -> Result<User> {
        let existing = self
            .get_user_by_id(user_id)?
            .ok_or_else(|| anyhow!("User not found"))?;
        let gem = gemini_api_key
            .map(|s| s.trim().to_string())
            .unwrap_or(existing.gemini_api_key);
        let oai = openai_api_key
            .map(|s| s.trim().to_string())
            .unwrap_or(existing.openai_api_key);

        self.conn.execute(
            r#"
            UPDATE users
            SET runner = ?1, gemini_api_key = ?2, openai_api_key = ?3
            WHERE id = ?4
            "#,
            params![runner.as_str(), gem, oai, user_id],
        )?;

        self.get_user_by_id(user_id)?
            .ok_or_else(|| anyhow!("User not found"))
    }

    fn map_user(row: &rusqlite::Row<'_>) -> rusqlite::Result<User> {
        let runner_str: String = row.get(2)?;
        Ok(User {
            id: row.get(0)?,
            username: row.get(1)?,
            runner: RunnerKind::from_str(&runner_str),
            gemini_api_key: row.get(3)?,
            gemini_model: row.get(4)?,
            openai_api_key: row.get(5)?,
            openai_model: row.get(6)?,
            created_at: row.get(7)?,
        })
    }

    // ---------- Topics ----------

    pub fn create_topic(&self, user_id: &str, title: &str, level: SkillLevel) -> Result<TopicSummary> {
        let clean: String = title.trim().chars().take(120).collect();
        if clean.is_empty() {
            return Err(anyhow!("Topic title cannot be empty."));
        }
        let id = Uuid::new_v4().to_string();
        let now = now_ms();
        self.conn.execute(
            r#"
            INSERT INTO topics (id, user_id, title, level, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![id, user_id, clean, level.as_str(), now, now],
        )?;

        self.get_topic(&id, user_id)?
            .ok_or_else(|| anyhow!("failed to load created topic"))
    }

    pub fn list_topics(&self, user_id: &str) -> Result<Vec<TopicSummary>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT t.id, t.user_id, t.title, t.level,
                   COUNT(c.id) AS concept_count,
                   COALESCE(SUM(CASE WHEN c.status IN ('mastered', 'deepened') THEN 1 ELSE 0 END), 0) AS mastered_count,
                   COALESCE(AVG(CASE WHEN c.attempts > 0 THEN c.mastery ELSE 0.0 END), 0.0) AS average_mastery,
                   t.total_attempts, t.correct_attempts, t.current_streak, t.best_streak,
                   t.created_at, t.updated_at
            FROM topics t
            LEFT JOIN concepts c ON c.topic_id = t.id
            WHERE t.user_id = ?1
            GROUP BY t.id
            ORDER BY t.updated_at DESC
            "#,
        )?;
        let rows = stmt.query_map(params![user_id], Self::map_topic)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_topic(&self, topic_id: &str, user_id: &str) -> Result<Option<TopicSummary>> {
        self.conn
            .query_row(
                r#"
                SELECT t.id, t.user_id, t.title, t.level,
                       COUNT(c.id) AS concept_count,
                       COALESCE(SUM(CASE WHEN c.status IN ('mastered', 'deepened') THEN 1 ELSE 0 END), 0) AS mastered_count,
                       COALESCE(AVG(CASE WHEN c.attempts > 0 THEN c.mastery ELSE 0.0 END), 0.0) AS average_mastery,
                       t.total_attempts, t.correct_attempts, t.current_streak, t.best_streak,
                       t.created_at, t.updated_at
                FROM topics t
                LEFT JOIN concepts c ON c.topic_id = t.id
                WHERE t.id = ?1 AND t.user_id = ?2
                GROUP BY t.id
                "#,
                params![topic_id, user_id],
                Self::map_topic,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn delete_topic(&self, topic_id: &str, user_id: &str) -> Result<bool> {
        let changed = self.conn.execute(
            "DELETE FROM topics WHERE id = ?1 AND user_id = ?2",
            params![topic_id, user_id],
        )?;
        Ok(changed > 0)
    }

    pub fn record_topic_attempt(&self, topic_id: &str, is_correct: bool, now: i64) -> Result<(u32, u32)> {
        let (cur, best): (u32, u32) = self.conn.query_row(
            "SELECT current_streak, best_streak FROM topics WHERE id = ?1",
            params![topic_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let next_streak = if is_correct { cur + 1 } else { 0 };
        let next_best = best.max(next_streak);
        self.conn.execute(
            r#"
            UPDATE topics
            SET total_attempts = total_attempts + 1,
                correct_attempts = correct_attempts + ?1,
                current_streak = ?2,
                best_streak = ?3,
                updated_at = ?4
            WHERE id = ?5
            "#,
            params![if is_correct { 1 } else { 0 }, next_streak, next_best, now, topic_id],
        )?;
        Ok((next_streak, next_best))
    }

    /// Queries high-mastery concepts from the user's *other* topics for cross-domain analogy bridges.
    pub fn list_cross_topic_anchors(
        &self,
        user_id: &str,
        exclude_topic_id: &str,
        limit: usize,
    ) -> Result<Vec<CrossTopicAnchor>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT t.title, c.slug, c.title, c.mastery
            FROM concepts c
            JOIN topics t ON t.id = c.topic_id
            WHERE t.user_id = ?1
              AND t.id != ?2
              AND c.attempts > 0
              AND c.mastery >= 0.72
            ORDER BY c.mastery DESC, c.attempts DESC
            LIMIT ?3
            "#,
        )?;
        let rows = stmt.query_map(params![user_id, exclude_topic_id, limit as i64], |r| {
            Ok(CrossTopicAnchor {
                topic_title: r.get(0)?,
                concept_slug: r.get(1)?,
                concept_title: r.get(2)?,
                mastery: r.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    fn map_topic(row: &rusqlite::Row<'_>) -> rusqlite::Result<TopicSummary> {
        let level_str: String = row.get(3)?;
        let avg: f64 = row.get(6)?;
        Ok(TopicSummary {
            id: row.get(0)?,
            user_id: row.get(1)?,
            title: row.get(2)?,
            level: SkillLevel::from_str(&level_str),
            concept_count: row.get::<_, i64>(4)? as usize,
            mastered_count: row.get::<_, i64>(5)? as usize,
            average_mastery: (avg * 1000.0).round() / 1000.0,
            total_attempts: row.get(7)?,
            correct_attempts: row.get(8)?,
            current_streak: row.get(9)?,
            best_streak: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    }

    // ---------- Compressed Concepts + Seeded Prerequisite Edges ----------

    pub fn upsert_concepts(&self, topic_id: &str, inputs: &[ConceptInput]) -> Result<Vec<ConceptNode>> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO concepts (id, topic_id, slug, title, subtopic, summary, depth, mastery, uncertainty, stability)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0.0)
                ON CONFLICT(topic_id, slug) DO UPDATE SET
                    title = excluded.title,
                    subtopic = excluded.subtopic,
                    summary = CASE WHEN length(excluded.summary) > 0 THEN excluded.summary ELSE concepts.summary END,
                    depth = excluded.depth
                "#,
            )?;
            for item in inputs {
                let title: String = item.title.trim().chars().take(100).collect();
                if title.is_empty() {
                    continue;
                }
                let slug = slugify(&title);
                let subtopic: String = item.subtopic.trim().chars().take(60).collect();
                let subtopic = if subtopic.is_empty() { "Core".to_string() } else { subtopic };
                let summary: String = item.summary.trim().chars().take(220).collect();
                let depth = item.depth.clamp(1, 4);
                stmt.execute(params![
                    Uuid::new_v4().to_string(),
                    topic_id,
                    slug,
                    title,
                    subtopic,
                    summary,
                    depth,
                    DEFAULT_UNSEEN_MU,
                    DEFAULT_UNSEEN_SIGMA,
                ])?;
            }
            tx.execute(
                "UPDATE topics SET updated_at = ?1 WHERE id = ?2",
                params![now_ms(), topic_id],
            )?;
        }
        tx.commit()?;

        let all_concepts = self.list_concepts(topic_id)?;
        let by_slug: HashMap<String, String> = all_concepts
            .iter()
            .map(|c| (c.slug.clone(), c.title.clone()))
            .collect();

        // Seed `Requires` edges from explicit `item.requires` OR infer light depth-chain edges if none provided
        let mut edges_to_upsert = Vec::new();
        for item in inputs {
            let to_slug = slugify_label(item.title.trim());
            let Some(to_label) = by_slug.get(&to_slug) else { continue };
            for req in &item.requires {
                let from_slug = slugify_label(req.trim());
                if from_slug != to_slug {
                    if let Some(from_label) = by_slug.get(&from_slug) {
                        edges_to_upsert.push(ConceptEdge {
                            id: format!("req:{topic_id}:{from_slug}:{to_slug}"),
                            topic_id: topic_id.to_string(),
                            from_slug,
                            from_label: from_label.clone(),
                            to_slug: to_slug.clone(),
                            to_label: to_label.clone(),
                            kind: EdgeKind::Requires,
                            weight: 0.80,
                            tension: 0.0,
                            note: String::new(),
                        });
                    }
                }
            }
        }

        // If the AI didn't emit explicit `requires` links, synthesize sensible depth (d-1 -> d) edges
        if edges_to_upsert.is_empty() && self.list_edges(topic_id)?.is_empty() {
            for target in &all_concepts {
                if target.depth > 1 {
                    if let Some(prereq) = all_concepts
                        .iter()
                        .find(|c| c.depth == target.depth - 1 && c.subtopic == target.subtopic)
                        .or_else(|| all_concepts.iter().find(|c| c.depth == target.depth - 1))
                    {
                        edges_to_upsert.push(ConceptEdge {
                            id: format!("req:{topic_id}:{}:{}", prereq.slug, target.slug),
                            topic_id: topic_id.to_string(),
                            from_slug: prereq.slug.clone(),
                            from_label: prereq.title.clone(),
                            to_slug: target.slug.clone(),
                            to_label: target.title.clone(),
                            kind: EdgeKind::Requires,
                            weight: 0.70,
                            tension: 0.0,
                            note: String::new(),
                        });
                    }
                }
            }
        }

        if !edges_to_upsert.is_empty() {
            self.upsert_edges(&edges_to_upsert)?;
        }

        Ok(all_concepts)
    }

    pub fn list_concepts(&self, topic_id: &str) -> Result<Vec<ConceptNode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, topic_id, slug, title, subtopic, summary, depth,
                   mastery, uncertainty, stability, ease_factor, interval_hours,
                   attempts, correct_count, wrong_count, consecutive_correct,
                   status, misconceptions_json, recent_angles_json,
                   last_seen_at, next_review_at
            FROM concepts
            WHERE topic_id = ?1
            ORDER BY depth ASC, subtopic ASC, title ASC
            "#,
        )?;
        let rows = stmt.query_map(params![topic_id], Self::map_concept)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_concept(&self, concept_id: &str) -> Result<Option<ConceptNode>> {
        self.conn
            .query_row(
                r#"
                SELECT id, topic_id, slug, title, subtopic, summary, depth,
                       mastery, uncertainty, stability, ease_factor, interval_hours,
                       attempts, correct_count, wrong_count, consecutive_correct,
                       status, misconceptions_json, recent_angles_json,
                       last_seen_at, next_review_at
                FROM concepts
                WHERE id = ?1
                "#,
                params![concept_id],
                Self::map_concept,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_concept_progress(&self, node: &ConceptNode) -> Result<ConceptNode> {
        let misconceptions = clamp_list(&node.misconceptions, MAX_MISCONCEPTIONS, MAX_MISCONCEPTION_CHARS);
        let recent_angles = clamp_list(&node.recent_angles, MAX_RECENT_ANGLES, MAX_ANGLE_CHARS);

        self.conn.execute(
            r#"
            UPDATE concepts
            SET mastery = ?1, uncertainty = ?2, stability = ?3,
                ease_factor = ?4, interval_hours = ?5,
                attempts = ?6, correct_count = ?7, wrong_count = ?8,
                consecutive_correct = ?9, status = ?10,
                misconceptions_json = ?11, recent_angles_json = ?12,
                last_seen_at = ?13, next_review_at = ?14
            WHERE id = ?15
            "#,
            params![
                node.mastery.clamp(0.0, 1.0),
                node.uncertainty.clamp(0.05, 1.0),
                node.stability.max(0.0),
                node.ease_factor,
                node.interval_hours,
                node.attempts,
                node.correct_count,
                node.wrong_count,
                node.consecutive_correct,
                node.status.as_str(),
                serde_json::to_string(&misconceptions)?,
                serde_json::to_string(&recent_angles)?,
                node.last_seen_at,
                node.next_review_at,
                node.id
            ],
        )?;

        self.get_concept(&node.id)?
            .ok_or_else(|| anyhow!("concept not found after update"))
    }

    fn map_concept(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConceptNode> {
        let status_str: String = row.get(16)?;
        let mis_json: String = row.get(17)?;
        let ang_json: String = row.get(18)?;
        Ok(ConceptNode {
            id: row.get(0)?,
            topic_id: row.get(1)?,
            slug: row.get(2)?,
            title: row.get(3)?,
            subtopic: row.get(4)?,
            summary: row.get(5)?,
            depth: row.get(6)?,
            mastery: row.get(7)?,
            uncertainty: row.get(8)?,
            stability: row.get(9)?,
            ease_factor: row.get(10)?,
            interval_hours: row.get(11)?,
            attempts: row.get(12)?,
            correct_count: row.get(13)?,
            wrong_count: row.get(14)?,
            consecutive_correct: row.get(15)?,
            status: ConceptStatus::from_str(&status_str),
            misconceptions: serde_json::from_str(&mis_json).unwrap_or_default(),
            recent_angles: serde_json::from_str(&ang_json).unwrap_or_default(),
            last_seen_at: row.get(19)?,
            next_review_at: row.get(20)?,
        })
    }

    // ---------- Cognitive Mesh Edges & Shadow Misconception Nodes ----------

    pub fn upsert_edges(&self, edges: &[ConceptEdge]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO concept_edges (id, topic_id, from_slug, from_label, to_slug, to_label, kind, weight, tension, note)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(topic_id, from_slug, to_slug, kind) DO UPDATE SET
                    from_label = excluded.from_label,
                    to_label = excluded.to_label,
                    weight = excluded.weight,
                    tension = excluded.tension,
                    note = excluded.note
                "#,
            )?;
            for e in edges {
                stmt.execute(params![
                    e.id,
                    e.topic_id,
                    e.from_slug,
                    e.from_label,
                    e.to_slug,
                    e.to_label,
                    e.kind.as_str(),
                    e.weight.clamp(0.0, 1.0),
                    e.tension.clamp(0.0, 1.0),
                    e.note,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_edges_by_ids(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.conn
                .execute("DELETE FROM concept_edges WHERE id = ?1", params![id])?;
        }
        Ok(())
    }

    pub fn list_edges(&self, topic_id: &str) -> Result<Vec<ConceptEdge>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, topic_id, from_slug, from_label, to_slug, to_label, kind, weight, tension, note
            FROM concept_edges
            WHERE topic_id = ?1
            ORDER BY tension DESC, weight DESC
            "#,
        )?;
        let rows = stmt.query_map(params![topic_id], |r| {
            let kind_str: String = r.get(6)?;
            Ok(ConceptEdge {
                id: r.get(0)?,
                topic_id: r.get(1)?,
                from_slug: r.get(2)?,
                from_label: r.get(3)?,
                to_slug: r.get(4)?,
                to_label: r.get(5)?,
                kind: EdgeKind::from_str(&kind_str),
                weight: r.get(7)?,
                tension: r.get(8)?,
                note: r.get(9)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn upsert_misconceptions(&self, items: &[MisconceptionNode]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO misconception_nodes (id, topic_id, slug, label, probability, evidence_count, refute_count, linked_concepts_json, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ON CONFLICT(topic_id, slug) DO UPDATE SET
                    label = excluded.label,
                    probability = excluded.probability,
                    evidence_count = excluded.evidence_count,
                    refute_count = excluded.refute_count,
                    linked_concepts_json = excluded.linked_concepts_json,
                    updated_at = excluded.updated_at
                "#,
            )?;
            for m in items {
                stmt.execute(params![
                    m.id,
                    m.topic_id,
                    m.slug,
                    m.label,
                    m.probability.clamp(0.0, 1.0),
                    m.evidence_count,
                    m.refute_count,
                    serde_json::to_string(&m.linked_concepts)?,
                    m.updated_at,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_misconceptions_by_ids(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.conn
                .execute("DELETE FROM misconception_nodes WHERE id = ?1", params![id])?;
        }
        Ok(())
    }

    pub fn list_misconceptions(&self, topic_id: &str) -> Result<Vec<MisconceptionNode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, topic_id, slug, label, probability, evidence_count, refute_count, linked_concepts_json, updated_at
            FROM misconception_nodes
            WHERE topic_id = ?1
            ORDER BY probability DESC, updated_at DESC
            "#,
        )?;
        let rows = stmt.query_map(params![topic_id], |r| {
            let linked_json: String = r.get(7)?;
            Ok(MisconceptionNode {
                id: r.get(0)?,
                topic_id: r.get(1)?,
                slug: r.get(2)?,
                label: r.get(3)?,
                probability: r.get(4)?,
                evidence_count: r.get(5)?,
                refute_count: r.get(6)?,
                linked_concepts: serde_json::from_str(&linked_json).unwrap_or_default(),
                updated_at: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------- Compact Question Queue ----------

    pub fn create_question(&self, input: CreateQuestionParams) -> Result<StoredQuestion> {
        let id = Uuid::new_v4().to_string();
        let now = now_ms();
        let angle: String = input.angle_summary.trim().chars().take(MAX_ANGLE_CHARS).collect();
        self.conn.execute(
            r#"
            INSERT INTO questions (
                id, topic_id, concept_id, intent, goal, prompt, choices_json,
                correct_choice_id, explanation, angle_summary, source, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                id,
                input.topic_id,
                input.concept_id,
                input.intent.as_str(),
                input.goal.trim(),
                input.prompt.trim(),
                serde_json::to_string(&input.choices)?,
                input.correct_choice_id,
                input.explanation.trim(),
                angle,
                input.source.as_str(),
                now
            ],
        )?;
        self.get_question(&id)?
            .ok_or_else(|| anyhow!("failed to load created question"))
    }

    pub fn get_question(&self, question_id: &str) -> Result<Option<StoredQuestion>> {
        self.conn
            .query_row(
                r#"
                SELECT q.id, q.topic_id, q.concept_id, c.title, c.subtopic, c.depth,
                       q.intent, q.goal, q.prompt, q.choices_json, q.correct_choice_id,
                       q.explanation, q.angle_summary, q.source, q.created_at,
                       q.answered_at, q.selected_choice_id, q.is_correct
                FROM questions q
                JOIN concepts c ON c.id = q.concept_id
                WHERE q.id = ?1
                "#,
                params![question_id],
                Self::map_question,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_next_open_question(&self, topic_id: &str) -> Result<Option<StoredQuestion>> {
        self.conn
            .query_row(
                r#"
                SELECT q.id, q.topic_id, q.concept_id, c.title, c.subtopic, c.depth,
                       q.intent, q.goal, q.prompt, q.choices_json, q.correct_choice_id,
                       q.explanation, q.angle_summary, q.source, q.created_at,
                       q.answered_at, q.selected_choice_id, q.is_correct
                FROM questions q
                JOIN concepts c ON c.id = q.concept_id
                WHERE q.topic_id = ?1 AND q.answered_at IS NULL
                ORDER BY q.created_at ASC
                LIMIT 1
                "#,
                params![topic_id],
                Self::map_question,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn list_open_question_concept_ids(&self, topic_id: &str) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT concept_id FROM questions WHERE topic_id = ?1 AND answered_at IS NULL",
        )?;
        let rows = stmt.query_map(params![topic_id], |r| r.get::<_, String>(0))?;
        let mut set = HashSet::new();
        for r in rows {
            set.insert(r?);
        }
        Ok(set)
    }

    pub fn count_open_questions(&self, topic_id: &str) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM questions WHERE topic_id = ?1 AND answered_at IS NULL",
            params![topic_id],
            |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    pub fn count_answered_questions(&self, topic_id: &str) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM questions WHERE topic_id = ?1 AND answered_at IS NOT NULL",
            params![topic_id],
            |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    pub fn delete_open_questions_for_concept(
        &self,
        topic_id: &str,
        concept_id: &str,
    ) -> Result<usize> {
        let changed = self.conn.execute(
            "DELETE FROM questions WHERE topic_id = ?1 AND concept_id = ?2 AND answered_at IS NULL",
            params![topic_id, concept_id],
        )?;
        Ok(changed)
    }

    pub fn delete_open_questions_by_source(
        &self,
        user_id: &str,
        source: RunnerKind,
    ) -> Result<usize> {
        let changed = self.conn.execute(
            r#"
            DELETE FROM questions
            WHERE answered_at IS NULL
              AND source = ?1
              AND topic_id IN (SELECT id FROM topics WHERE user_id = ?2)
            "#,
            params![source.as_str(), user_id],
        )?;
        Ok(changed)
    }

    pub fn mark_question_answered(
        &self,
        question_id: &str,
        selected_choice_id: &str,
        is_correct: bool,
        now: i64,
    ) -> Result<Option<StoredQuestion>> {
        let Some(existing) = self.get_question(question_id)? else {
            return Ok(None);
        };
        if existing.answered_at.is_some() {
            return Ok(None);
        }

        self.conn.execute(
            r#"
            UPDATE questions
            SET answered_at = ?1, selected_choice_id = ?2, is_correct = ?3
            WHERE id = ?4
            "#,
            params![now, selected_choice_id, if is_correct { 1 } else { 0 }, question_id],
        )?;

        self.conn.execute(
            r#"
            DELETE FROM questions
            WHERE topic_id = ?1 AND answered_at IS NOT NULL
              AND id NOT IN (
                SELECT id FROM questions
                WHERE topic_id = ?1 AND answered_at IS NOT NULL
                ORDER BY answered_at DESC
                LIMIT ?2
              )
            "#,
            params![existing.topic_id, MAX_ANSWERED_QUESTIONS_PER_TOPIC],
        )?;

        self.get_question(question_id)
    }

    fn map_question(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredQuestion> {
        let intent_str: String = row.get(6)?;
        let choices_json: String = row.get(9)?;
        let source_str: String = row.get(13)?;
        let is_correct_int: Option<i64> = row.get(17)?;
        Ok(StoredQuestion {
            id: row.get(0)?,
            topic_id: row.get(1)?,
            concept_id: row.get(2)?,
            concept_title: row.get(3)?,
            subtopic: row.get(4)?,
            depth: row.get(5)?,
            intent: QuestionIntent::from_str(&intent_str),
            goal: row.get(7)?,
            prompt: row.get(8)?,
            choices: serde_json::from_str(&choices_json).unwrap_or_default(),
            correct_choice_id: row.get(10)?,
            explanation: row.get(11)?,
            angle_summary: row.get(12)?,
            source: RunnerKind::from_str(&source_str),
            created_at: row.get(14)?,
            answered_at: row.get(15)?,
            selected_choice_id: row.get(16)?,
            is_correct: is_correct_int.map(|v| v == 1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_misconceptions_angles_edges_and_prunes_old_questions() {
        let store = KnowledgeStore::open_in_memory().unwrap();
        let user = store.create_user("alice", "secret123", RunnerKind::Codex).unwrap();
        assert_eq!(store.authenticate_user("alice", "secret123").unwrap().id, user.id);

        let topic = store.create_topic(&user.id, "Rust Ownership", SkillLevel::Beginner).unwrap();
        let concepts = store
            .upsert_concepts(
                &topic.id,
                &[
                    ConceptInput {
                        title: "Borrow Checker".into(),
                        subtopic: "Memory".into(),
                        summary: "Tracks mutable and shared borrows.".into(),
                        depth: 1,
                        requires: vec![],
                    },
                    ConceptInput {
                        title: "Lifetimes".into(),
                        subtopic: "Memory".into(),
                        summary: "Ensure references never outlive data.".into(),
                        depth: 2,
                        requires: vec!["Borrow Checker".into()],
                    },
                ],
            )
            .unwrap();
        assert_eq!(concepts.len(), 2);

        let edges = store.list_edges(&topic.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].kind, EdgeKind::Requires);
        assert_eq!(edges[0].from_slug, "borrow-checker");
        assert_eq!(edges[0].to_slug, "lifetimes");

        let mut bc = concepts.iter().find(|c| c.slug == "borrow-checker").unwrap().clone();
        assert_eq!(bc.mastery, DEFAULT_UNSEEN_MU);
        assert_eq!(bc.uncertainty, DEFAULT_UNSEEN_SIGMA);

        bc.misconceptions = vec!["m1".into(), "m2".into(), "m3".into(), "m4".into(), "m4".into()];
        bc.recent_angles = vec!["a1".into(), "a2".into(), "a3".into(), "a4".into(), "a5".into()];
        let saved = store.save_concept_progress(&bc).unwrap();
        assert_eq!(saved.misconceptions, vec!["m2", "m3", "m4"]);
        assert_eq!(saved.recent_angles, vec!["a2", "a3", "a4", "a5"]);

        for i in 0..14 {
            let q = store
                .create_question(CreateQuestionParams {
                    topic_id: topic.id.clone(),
                    concept_id: bc.id.clone(),
                    intent: QuestionIntent::Introduce,
                    goal: "Understand why Rust tracks borrows.".into(),
                    prompt: format!("Q {i}?"),
                    choices: vec![
                        QuestionChoice { id: "A".into(), text: "1".into(), trap_kind: None, trap_target: None, why_wrong: None },
                        QuestionChoice { id: "B".into(), text: "2".into(), trap_kind: None, trap_target: None, why_wrong: None },
                        QuestionChoice { id: "C".into(), text: "3".into(), trap_kind: None, trap_target: None, why_wrong: None },
                        QuestionChoice { id: "D".into(), text: "4".into(), trap_kind: None, trap_target: None, why_wrong: None },
                    ],
                    correct_choice_id: "A".into(),
                    explanation: "A is right.".into(),
                    angle_summary: format!("angle {i}"),
                    source: RunnerKind::Demo,
                })
                .unwrap();
            store.mark_question_answered(&q.id, "A", true, 1000 + i).unwrap();
        }
        assert_eq!(store.count_open_questions(&topic.id).unwrap(), 0);
        assert_eq!(store.count_answered_questions(&topic.id).unwrap(), 10);
    }
}
