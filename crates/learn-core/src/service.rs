use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::mesh::{
    MeshObservation, compile_cognitive_slice, propagate_answer_observation, slugify_label,
};
use crate::models::{
    AnswerResult, ClarifyingReply, ConceptEdge, ConceptStatus, DetectedRunner, EdgeKind,
    KnowledgeMap, RunnerKind, SkillLevel, StoredQuestion, TopicSummary, User,
};
use crate::runner::{AiRunner, RunnerConfig};
use crate::scheduler::{select_next_directive, summarize_stats};
use crate::store::{CreateQuestionParams, KnowledgeStore, now_ms};

/// Number of ready-to-go questions kept buffered ahead of the active question.
pub const DEFAULT_BUFFER_TARGET: usize = 3;

/// Total open questions maintained while one question is currently on screen (1 active + 3 buffered).
pub const DEFAULT_OPEN_QUEUE_TARGET: usize = DEFAULT_BUFFER_TARGET + 1;

/// Thread-safe application service coordinating storage, Cognitive Mesh propagation, scheduling, and AI generation.
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
        let edges = store.list_edges(topic_id)?;
        let misconceptions = store.list_misconceptions(topic_id)?;
        let anchors = store.list_cross_topic_anchors(user_id, topic_id, 4)?;
        let cognitive_slice = compile_cognitive_slice(&concepts, &edges, &misconceptions, &anchors);
        let stats = summarize_stats(&concepts);
        Ok(KnowledgeMap {
            topic,
            concepts,
            edges,
            misconceptions,
            cognitive_slice,
            stats,
        })
    }

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

    pub fn peek_buffered_question(&self, topic_id: &str) -> Result<Option<StoredQuestion>> {
        self.store.lock().unwrap().get_next_open_question(topic_id)
    }

    pub fn count_open_questions(&self, topic_id: &str) -> Result<usize> {
        self.store.lock().unwrap().count_open_questions(topic_id)
    }

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

    pub fn prefetch_next_question(&self, user: &User, topic_id: &str) -> Result<bool> {
        self.prefetch_one_into_buffer(user, topic_id, DEFAULT_OPEN_QUEUE_TARGET)
    }

    /// Grades an answer and runs 1-hop Cognitive Mesh belief propagation locally in < 1ms.
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

        let concepts = store.list_concepts(topic_id)?;
        let edges = store.list_edges(topic_id)?;
        let misconceptions = store.list_misconceptions(topic_id)?;

        let selected_choice = question.choices.iter().find(|c| c.id == selected_choice_id);
        let correct_text = question
            .choices
            .iter()
            .find(|c| c.id == question.correct_choice_id)
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let fallback_note = if !is_correct {
            Some(format!(
                "Picked '{}' on {} instead of '{}'",
                selected_choice
                    .map(|c| c.text.as_str())
                    .unwrap_or("")
                    .chars()
                    .take(45)
                    .collect::<String>(),
                question.angle_summary.chars().take(35).collect::<String>(),
                correct_text.chars().take(45).collect::<String>(),
            ))
        } else {
            None
        };

        let prop = propagate_answer_observation(
            &concepts,
            &edges,
            &misconceptions,
            MeshObservation {
                target_concept_id: &question.concept_id,
                intent: question.intent,
                is_correct,
                angle_summary: &question.angle_summary,
                selected_choice,
                all_choices: &question.choices,
                fallback_misconception: fallback_note.as_deref(),
                now_ms: now,
            },
        );

        for c in &prop.updated_concepts {
            store.save_concept_progress(c)?;
            // Invalidate stale buffered questions for touched concepts so upcoming questions adapt.
            let _ = store.delete_open_questions_for_concept(topic_id, &c.id);
        }
        store.upsert_edges(&prop.updated_edges)?;
        store.delete_edges_by_ids(&prop.pruned_edge_ids)?;
        store.upsert_misconceptions(&prop.updated_misconceptions)?;
        store.delete_misconceptions_by_ids(&prop.pruned_misconception_ids)?;

        let (topic_streak, best_streak) = store.record_topic_attempt(topic_id, is_correct, now)?;

        let selected_why_wrong = if !is_correct {
            selected_choice.and_then(|c| c.why_wrong.clone())
        } else {
            None
        };

        Ok(AnswerResult {
            question_id: question.id,
            is_correct,
            selected_choice_id: selected_choice_id.to_string(),
            correct_choice_id: question.correct_choice_id,
            correct_choice_text: correct_text,
            selected_why_wrong,
            explanation: question.explanation,
            reward_message: prop.reward_message,
            concept: prop.target_concept,
            topic_streak,
            best_streak,
            mastery_delta: prop.mastery_delta,
            mesh_effects: prop.effects,
        })
    }

    /// Answers a learner's mid-quiz clarifying question (e.g. "What is a motif?"),
    /// captures any missing foundational concept into the Cognitive Mesh as a `NeedsReview`
    /// prerequisite linked to the current concept, and returns a spoiler-free reply.
    pub fn ask_clarifying_question(
        &self,
        user: &User,
        topic_id: &str,
        question_id: &str,
        learner_question: &str,
        already_answered: bool,
    ) -> Result<ClarifyingReply> {
        let clean_q = learner_question.trim();
        if clean_q.is_empty() {
            return Err(anyhow!("Type a question first."));
        }

        let (topic, question, current_concept) = {
            let store = self.store.lock().unwrap();
            let topic = store
                .get_topic(topic_id, &user.id)?
                .ok_or_else(|| anyhow!("Topic not found"))?;
            let question = store
                .get_question(question_id)?
                .ok_or_else(|| anyhow!("Question not found"))?;
            let concept = store
                .get_concept(&question.concept_id)?
                .ok_or_else(|| anyhow!("Concept not found"))?;
            (topic, question, concept)
        };

        let cfg = runner_cfg_for(user);
        let analysis = self.runner.answer_learner_question(
            &topic.title,
            topic.level,
            &current_concept.title,
            &question.prompt,
            clean_q,
            already_answered,
            &cfg,
        )?;

        let mut added_concept_title = None;
        let mut mesh_note = None;

        if let Some(missing) = analysis.missing_concept {
            let now = now_ms();
            let store = self.store.lock().unwrap();
            let all_concepts = store.upsert_concepts(topic_id, &[missing.clone()])?;
            let target_slug = slugify_label(&missing.title);
            if let Some(mut gap_node) = all_concepts.into_iter().find(|c| c.slug == target_slug) {
                // Mark as high-priority knowledge gap so upcoming questions reinforce it!
                gap_node.mastery = gap_node.mastery.min(0.28);
                gap_node.uncertainty = gap_node.uncertainty.max(0.82);
                gap_node.status = ConceptStatus::NeedsReview;
                gap_node.wrong_count = gap_node.wrong_count.max(1);
                gap_node.consecutive_correct = 0;
                gap_node.next_review_at = Some(now);
                if let Some(note) = &analysis.gap_note {
                    gap_node.misconceptions.push(note.clone());
                }
                let saved = store.save_concept_progress(&gap_node)?;

                // Wire `gap_node --Requires--> current_concept` if they are distinct
                if saved.slug != current_concept.slug {
                    let edge = ConceptEdge {
                        id: format!("req:{topic_id}:{}:{}", saved.slug, current_concept.slug),
                        topic_id: topic_id.to_string(),
                        from_slug: saved.slug.clone(),
                        from_label: saved.title.clone(),
                        to_slug: current_concept.slug.clone(),
                        to_label: current_concept.title.clone(),
                        kind: EdgeKind::Requires,
                        weight: 0.85,
                        tension: 0.35,
                        note: "learner_asked_clarifying_question".into(),
                    };
                    let _ = store.upsert_edges(&[edge]);
                }

                added_concept_title = Some(saved.title.clone());
                mesh_note = Some(format!(
                    "Added '{}' to your Cognitive Mesh as a prerequisite for '{}' — upcoming questions will reinforce it.",
                    saved.title, current_concept.title
                ));
            }
        }

        Ok(ClarifyingReply {
            question_asked: clean_q.to_string(),
            answer: analysis.answer,
            added_concept_title,
            mesh_note,
        })
    }

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
        let (topic, mut concepts, mut edges, misconceptions, anchors) = {
            let store = self.store.lock().unwrap();
            let topic = store
                .get_topic(topic_id, &user.id)?
                .ok_or_else(|| anyhow!("Topic not found"))?;
            let concepts = store.list_concepts(topic_id)?;
            let edges = store.list_edges(topic_id)?;
            let misconceptions = store.list_misconceptions(topic_id)?;
            let anchors = store.list_cross_topic_anchors(&user.id, topic_id, 4)?;
            (topic, concepts, edges, misconceptions, anchors)
        };

        let cfg = runner_cfg_for(user);

        if concepts.is_empty() {
            let initial = self
                .runner
                .generate_concept_map(&topic.title, topic.level, &cfg, &[], 1)?;
            let store = self.store.lock().unwrap();
            concepts = store.upsert_concepts(topic_id, &initial)?;
            edges = store.list_edges(topic_id)?;
        }

        let mut directive = select_next_directive(
            &topic,
            &concepts,
            &edges,
            &misconceptions,
            &anchors,
            now_ms(),
            exclude_ids,
        )?;

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
                    let store = self.store.lock().unwrap();
                    if let Ok(updated_concepts) = store.upsert_concepts(topic_id, &deeper) {
                        let updated_edges = store.list_edges(topic_id).unwrap_or(edges);
                        if let Ok(next_dir) = select_next_directive(
                            &topic,
                            &updated_concepts,
                            &updated_edges,
                            &misconceptions,
                            &anchors,
                            now_ms(),
                            exclude_ids,
                        ) {
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
            goal: generated.goal,
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
    use crate::models::{ConceptStatus, DistractorTrapKind};

    #[test]
    fn full_cognitive_mesh_loop_runs_fast_with_buffered_prefetch() {
        let svc = LearnService::open_in_memory(AiRunner::default()).unwrap();
        let mut user = svc.register("tester", "pass1234").unwrap();
        user = svc.update_runner(&user.id, RunnerKind::Demo, None, None).unwrap();

        let topic = svc
            .create_topic_with_map(&user, "Quantum Computing", SkillLevel::Beginner)
            .unwrap();
        assert!(topic.concept_count >= 6);

        let q1 = svc.get_or_generate_next_question(&user, &topic.id).unwrap();
        assert_eq!(q1.choices.len(), 4);

        // Pick the misconception distractor to verify Shadow Node activation
        let mis_choice = q1
            .choices
            .iter()
            .find(|c| c.trap_kind == Some(DistractorTrapKind::Misconception))
            .unwrap();
        let res1 = svc.submit_answer(&user.id, &topic.id, &q1.id, &mis_choice.id).unwrap();
        assert!(!res1.is_correct);
        assert_eq!(res1.concept.status, ConceptStatus::NeedsReview);
        assert!(!res1.mesh_effects.is_empty());

        let map = svc.get_knowledge_map(&user.id, &topic.id).unwrap();
        assert_eq!(map.misconceptions.len(), 1);
        assert!(!map.edges.is_empty());

        // Prefetch fills buffer up to target and caps there
        let generated = svc
            .fill_question_buffer(&user, &topic.id, DEFAULT_BUFFER_TARGET)
            .unwrap();
        assert_eq!(generated, DEFAULT_BUFFER_TARGET);
        assert_eq!(svc.count_open_questions(&topic.id).unwrap(), DEFAULT_BUFFER_TARGET);
        assert_eq!(
            svc.fill_question_buffer(&user, &topic.id, DEFAULT_BUFFER_TARGET).unwrap(),
            0
        );

        let q2 = svc.peek_buffered_question(&topic.id).unwrap().unwrap();
        let res2 = svc
            .submit_answer(&user.id, &topic.id, &q2.id, &q2.correct_choice_id)
            .unwrap();
        assert!(res2.is_correct);
        assert_eq!(res2.topic_streak, 1);

        // Clarifying question adds missing prerequisite concept to the Cognitive Mesh
        let reply = svc
            .ask_clarifying_question(&user, &topic.id, &q2.id, "What is a qubit?", true)
            .unwrap();
        assert_eq!(reply.added_concept_title.as_deref(), Some("qubit"));
        let map_after = svc.get_knowledge_map(&user.id, &topic.id).unwrap();
        assert!(map_after.concepts.iter().any(|c| c.slug == "qubit" && c.status == ConceptStatus::NeedsReview));
    }
}
