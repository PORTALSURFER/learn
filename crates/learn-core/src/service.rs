use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::models::{
    AnswerResult, DetectedRunner, KnowledgeMap, RunnerKind, SkillLevel, StoredQuestion,
    TopicSummary, User,
};
use crate::runner::{AiRunner, RunnerConfig};
use crate::scheduler::{apply_answer_outcome, select_next_directive, summarize_stats};
use crate::store::{CreateQuestionParams, KnowledgeStore, now_ms};

/// Number of ready-to-go questions kept buffered ahead of the active question.
pub const DEFAULT_BUFFER_TARGET: usize = 3;

/// Total open questions maintained while one question is currently on screen (1 active + 3 buffered).
pub const DEFAULT_OPEN_QUEUE_TARGET: usize = DEFAULT_BUFFER_TARGET + 1;

/// Thread-safe application service coordinating storage, scheduling, and AI generation.
#[derive(Clone)]
pub struct LearnService {
    store: Arc<Mutex<KnowledgeStore>>,
    runner: Arc<AiRunner>,
    in_flight_prefetch: Arc<Mutex<HashSet<String>>>,
}

impl LearnService {
    pub fn open_default() -> Result<Self> {
        let path = default_db_path();
        Self::open_path(&path)
    }

    pub fn open_path(path: &Path) -> Result<Self> {
        let store = KnowledgeStore::open_path(path)?;
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            runner: Arc::new(AiRunner::default()),
            in_flight_prefetch: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    pub fn open_in_memory(runner: AiRunner) -> Result<Self> {
        let store = KnowledgeStore::open_in_memory()?;
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            runner: Arc::new(runner),
            in_flight_prefetch: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    pub fn detect_runners(&self, user: Option<&User>) -> Vec<DetectedRunner> {
        let (gem, oai) = user
            .map(|u| (u.gemini_api_key.as_str(), u.openai_api_key.as_str()))
            .unwrap_or(("", ""));
        self.runner.detect_runners(gem, oai)
    }

    pub fn list_usernames(&self) -> Result<Vec<String>> {
        self.store.lock().unwrap().list_usernames()
    }

    pub fn register(&self, username: &str, password: &str) -> Result<User> {
        let default_runner = self.runner.pick_best_default_runner();
        self.store
            .lock()
            .unwrap()
            .create_user(username, password, default_runner)
    }

    pub fn login(&self, username: &str, password: &str) -> Result<User> {
        self.store.lock().unwrap().authenticate_user(username, password)
    }

    pub fn update_runner(
        &self,
        user_id: &str,
        runner: RunnerKind,
        gemini_api_key: Option<&str>,
        openai_api_key: Option<&str>,
    ) -> Result<User> {
        let store = self.store.lock().unwrap();
        let prev = store.get_user_by_id(user_id)?;
        let updated = store.update_user_runner(user_id, runner, gemini_api_key, openai_api_key)?;
        // If switching away from Demo to a live AI runner, clear queued demo questions so live ones take over.
        if let Some(old) = prev {
            if old.runner == RunnerKind::Demo && runner != RunnerKind::Demo {
                let _ = store.delete_open_questions_by_source(user_id, RunnerKind::Demo);
            }
        }
        Ok(updated)
    }

    pub fn list_topics(&self, user_id: &str) -> Result<Vec<TopicSummary>> {
        self.store.lock().unwrap().list_topics(user_id)
    }

    pub fn delete_topic(&self, user_id: &str, topic_id: &str) -> Result<bool> {
        self.store.lock().unwrap().delete_topic(topic_id, user_id)
    }

    pub fn get_knowledge_map(&self, user_id: &str, topic_id: &str) -> Result<KnowledgeMap> {
        let store = self.store.lock().unwrap();
        let topic = store
            .get_topic(topic_id, user_id)?
            .ok_or_else(|| anyhow!("Topic not found"))?;
        let concepts = store.list_concepts(topic_id)?;
        let stats = summarize_stats(&concepts);
        Ok(KnowledgeMap {
            topic,
            concepts,
            stats,
        })
    }

    /// Creates a new topic and immediately asks the active AI runner to build its compact Knowledge Map.
    pub fn create_topic_with_map(
        &self,
        user: &User,
        title: &str,
        level: SkillLevel,
    ) -> Result<TopicSummary> {
        let topic = self
            .store
            .lock()
            .unwrap()
            .create_topic(&user.id, title, level)?;

        let cfg = runner_cfg_for(user);
        let concepts = match self.runner.generate_concept_map(&topic.title, level, &cfg, &[], 1) {
            Ok(c) => c,
            Err(err) => {
                // Clean up empty topic on hard error so the UI never holds a broken topic.
                let _ = self.store.lock().unwrap().delete_topic(&topic.id, &user.id);
                return Err(err);
            }
        };

        let store = self.store.lock().unwrap();
        store.upsert_concepts(&topic.id, &concepts)?;
        let updated = store
            .get_topic(&topic.id, &user.id)?
            .ok_or_else(|| anyhow!("Topic disappeared"))?;
        Ok(updated)
    }

    /// Returns the next buffered open question immediately from SQLite without invoking any AI runner.
    pub fn peek_buffered_question(&self, topic_id: &str) -> Result<Option<StoredQuestion>> {
        self.store.lock().unwrap().get_next_open_question(topic_id)
    }

    /// Counts how many open questions are currently ready in the queue for a topic.
    pub fn count_open_questions(&self, topic_id: &str) -> Result<usize> {
        self.store.lock().unwrap().count_open_questions(topic_id)
    }

    /// Fetches the next open question (returning instantaneously if buffered),
    /// or generates one live from the deterministic scheduler directive.
    pub fn get_or_generate_next_question(
        &self,
        user: &User,
        topic_id: &str,
    ) -> Result<StoredQuestion> {
        if let Some(existing) = self.store.lock().unwrap().get_next_open_question(topic_id)? {
            return Ok(existing);
        }
        self.generate_and_store_one_question(user, topic_id, &HashSet::new())
    }

    /// Generates a single additional buffered question if the open queue is below `target_open`.
    /// Returns `Ok(true)` if a new question was generated, or `Ok(false)` if already full or busy.
    pub fn prefetch_one_into_buffer(
        &self,
        user: &User,
        topic_id: &str,
        target_open: usize,
    ) -> Result<bool> {
        {
            let mut active = self.in_flight_prefetch.lock().unwrap();
            if active.contains(topic_id) {
                return Ok(false);
            }
            active.insert(topic_id.to_string());
        }

        let res = (|| -> Result<bool> {
            let (open_count, exclude_ids) = {
                let store = self.store.lock().unwrap();
                (
                    store.count_open_questions(topic_id)?,
                    store.list_open_question_concept_ids(topic_id)?,
                )
            };
            if open_count >= target_open {
                return Ok(false);
            }
            let _ = self.generate_and_store_one_question(user, topic_id, &exclude_ids)?;
            Ok(true)
        })();

        self.in_flight_prefetch.lock().unwrap().remove(topic_id);
        res
    }

    /// Tops up the question buffer until `target_open` open questions are ready.
    /// Returns the number of questions newly generated.
    pub fn fill_question_buffer(
        &self,
        user: &User,
        topic_id: &str,
        target_open: usize,
    ) -> Result<usize> {
        let mut generated = 0;
        while self.prefetch_one_into_buffer(user, topic_id, target_open)? {
            generated += 1;
        }
        Ok(generated)
    }

    /// Backwards-compatible helper: ensures at least 1 buffered question is queued.
    pub fn prefetch_next_question(&self, user: &User, topic_id: &str) -> Result<bool> {
        self.prefetch_one_into_buffer(user, topic_id, DEFAULT_OPEN_QUEUE_TARGET)
    }

    /// Grades an answer, updates the compressed Knowledge Map & SM-2 schedule, and returns feedback.
    pub fn submit_answer(
        &self,
        user_id: &str,
        topic_id: &str,
        question_id: &str,
        selected_choice_id: &str,
    ) -> Result<AnswerResult> {
        let now = now_ms();
        let store = self.store.lock().unwrap();

        let _topic = store
            .get_topic(topic_id, user_id)?
            .ok_or_else(|| anyhow!("Topic not found"))?;
        let question = store
            .get_question(question_id)?
            .ok_or_else(|| anyhow!("Question not found"))?;
        if question.topic_id != topic_id {
            return Err(anyhow!("Question does not belong to topic"));
        }
        if question.answered_at.is_some() {
            return Err(anyhow!("Question has already been answered"));
        }

        let is_correct = question.correct_choice_id == selected_choice_id;
        store.mark_question_answered(question_id, selected_choice_id, is_correct, now)?;

        let concept = store
            .get_concept(&question.concept_id)?
            .ok_or_else(|| anyhow!("Concept not found"))?;

        let selected_text = question
            .choices
            .iter()
            .find(|c| c.id == selected_choice_id)
            .map(|c| c.text.as_str())
            .unwrap_or("");
        let correct_text = question
            .choices
            .iter()
            .find(|c| c.id == question.correct_choice_id)
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let misconception_note = if !is_correct {
            Some(format!(
                "Picked '{}' on {} instead of '{}'",
                selected_text.chars().take(45).collect::<String>(),
                question.angle_summary.chars().take(35).collect::<String>(),
                correct_text.chars().take(45).collect::<String>(),
            ))
        } else {
            None
        };

        let outcome = apply_answer_outcome(
            &concept,
            is_correct,
            &question.angle_summary,
            misconception_note.as_deref(),
            now,
        );
        let saved_concept = store.save_concept_progress(&outcome.updated_concept)?;
        // Invalidate any stale buffered questions for this same concept so future questions reflect updated mastery/misconceptions.
        let _ = store.delete_open_questions_for_concept(topic_id, &question.concept_id);
        let (topic_streak, best_streak) = store.record_topic_attempt(topic_id, is_correct, now)?;

        Ok(AnswerResult {
            question_id: question.id,
            is_correct,
            selected_choice_id: selected_choice_id.to_string(),
            correct_choice_id: question.correct_choice_id,
            correct_choice_text: correct_text,
            explanation: question.explanation,
            reward_message: outcome.reward_message,
            concept: saved_concept,
            topic_streak,
            best_streak,
            mastery_delta: outcome.mastery_delta,
        })
    }

    /// Explicitly expands a topic's knowledge map with deeper concepts.
    pub fn expand_knowledge_map(&self, user: &User, topic_id: &str) -> Result<KnowledgeMap> {
        let (topic, existing_titles, next_depth) = {
            let store = self.store.lock().unwrap();
            let topic = store
                .get_topic(topic_id, &user.id)?
                .ok_or_else(|| anyhow!("Topic not found"))?;
            let concepts = store.list_concepts(topic_id)?;
            let titles: Vec<String> = concepts.iter().map(|c| c.title.clone()).collect();
            let max_depth = concepts.iter().map(|c| c.depth).max().unwrap_or(1);
            (topic, titles, (max_depth + 1).min(4))
        };

        let cfg = runner_cfg_for(user);
        let new_concepts = self.runner.generate_concept_map(
            &topic.title,
            topic.level,
            &cfg,
            &existing_titles,
            next_depth,
        )?;
        self.store
            .lock()
            .unwrap()
            .upsert_concepts(topic_id, &new_concepts)?;
        self.get_knowledge_map(&user.id, topic_id)
    }

    fn generate_and_store_one_question(
        &self,
        user: &User,
        topic_id: &str,
        exclude_ids: &HashSet<String>,
    ) -> Result<StoredQuestion> {
        let (topic, mut concepts) = {
            let store = self.store.lock().unwrap();
            let topic = store
                .get_topic(topic_id, &user.id)?
                .ok_or_else(|| anyhow!("Topic not found"))?;
            let concepts = store.list_concepts(topic_id)?;
            (topic, concepts)
        };

        let cfg = runner_cfg_for(user);

        if concepts.is_empty() {
            let initial = self
                .runner
                .generate_concept_map(&topic.title, topic.level, &cfg, &[], 1)?;
            concepts = self.store.lock().unwrap().upsert_concepts(topic_id, &initial)?;
        }

        let mut directive = select_next_directive(&topic, &concepts, now_ms(), exclude_ids)?;

        // Automatically deepen the knowledge map when the learner masters all current concepts.
        if directive.requires_new_concepts {
            let existing_titles: Vec<String> = concepts.iter().map(|c| c.title.clone()).collect();
            if let Ok(deeper) = self.runner.generate_concept_map(
                &topic.title,
                topic.level,
                &cfg,
                &existing_titles,
                directive.target_depth,
            ) {
                if !deeper.is_empty() {
                    if let Ok(updated_concepts) =
                        self.store.lock().unwrap().upsert_concepts(topic_id, &deeper)
                    {
                        if let Ok(next_dir) =
                            select_next_directive(&topic, &updated_concepts, now_ms(), exclude_ids)
                        {
                            directive = next_dir;
                        }
                    }
                }
            }
        }

        let generated = self.runner.generate_question(&directive, &cfg)?;
        let stored = self.store.lock().unwrap().create_question(CreateQuestionParams {
            topic_id: topic_id.to_string(),
            concept_id: directive.concept.id,
            intent: directive.intent,
            prompt: generated.prompt,
            choices: generated.choices,
            correct_choice_id: generated.correct_choice_id,
            explanation: generated.explanation,
            angle_summary: generated.angle_summary,
            source: cfg.kind,
        })?;

        Ok(stored)
    }
}

fn runner_cfg_for(user: &User) -> RunnerConfig {
    RunnerConfig {
        kind: user.runner,
        gemini_api_key: user.gemini_api_key.clone(),
        gemini_model: user.gemini_model.clone(),
        openai_api_key: user.openai_api_key.clone(),
        openai_model: user.openai_model.clone(),
    }
}

pub fn default_db_path() -> PathBuf {
    if let Ok(custom) = std::env::var("LEARN_DB_PATH") {
        if !custom.trim().is_empty() {
            return PathBuf::from(custom);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".learn-iocore").join("learn.db");
    }
    PathBuf::from("data/learn.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ConceptStatus;

    #[test]
    fn full_learning_loop_runs_fast_with_demo_runner() {
        let svc = LearnService::open_in_memory(AiRunner::default()).unwrap();
        let mut user = svc.register("tester", "pass1234").unwrap();
        user = svc.update_runner(&user.id, RunnerKind::Demo, None, None).unwrap();

        let topic = svc
            .create_topic_with_map(&user, "Quantum Computing", SkillLevel::Beginner)
            .unwrap();
        assert!(topic.concept_count >= 6);

        let q1 = svc.get_or_generate_next_question(&user, &topic.id).unwrap();
        assert_eq!(q1.choices.len(), 4);

        // Wrong answer -> records misconception & needs_review
        let wrong_id = q1
            .choices
            .iter()
            .find(|c| c.id != q1.correct_choice_id)
            .unwrap()
            .id
            .clone();
        let res1 = svc.submit_answer(&user.id, &topic.id, &q1.id, &wrong_id).unwrap();
        assert!(!res1.is_correct);
        assert_eq!(res1.concept.status, ConceptStatus::NeedsReview);
        assert_eq!(res1.concept.misconceptions.len(), 1);

        // Prefetch fills buffer up to target and avoids duplicating concepts
        let generated = svc
            .fill_question_buffer(&user, &topic.id, DEFAULT_BUFFER_TARGET)
            .unwrap();
        assert_eq!(generated, DEFAULT_BUFFER_TARGET);
        assert_eq!(svc.count_open_questions(&topic.id).unwrap(), DEFAULT_BUFFER_TARGET);
        // Calling fill again when full generates 0 extra questions (bounded buffer)
        assert_eq!(
            svc.fill_question_buffer(&user, &topic.id, DEFAULT_BUFFER_TARGET)
                .unwrap(),
            0
        );

        // Next question is popped instantaneously from buffer
        let q2 = svc.peek_buffered_question(&topic.id).unwrap().unwrap();
        let res2 = svc
            .submit_answer(&user.id, &topic.id, &q2.id, &q2.correct_choice_id)
            .unwrap();
        assert!(res2.is_correct);
        assert_eq!(res2.topic_streak, 1);
        assert_eq!(
            svc.count_open_questions(&topic.id).unwrap(),
            DEFAULT_BUFFER_TARGET - 1
        );

        // Buffer tops back up by exactly 1
        assert_eq!(
            svc.fill_question_buffer(&user, &topic.id, DEFAULT_BUFFER_TARGET)
                .unwrap(),
            1
        );
    }
}
