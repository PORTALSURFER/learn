use std::collections::HashSet;

use anyhow::{Result, anyhow};

use crate::models::{
    CompressedContext, ConceptNode, ConceptStatus, KnowledgeStats, QuestionIntent,
    SchedulerDirective, StrugglingSummary, TopicSummary,
};

const HOUR_MS: f64 = 3_600_000.0;

const REWARD_MESSAGES: &[&str] = &[
    "You're right! Solid recall.",
    "You're right! That mental model is locking in.",
    "You're right! Clean reasoning.",
    "You're right! Real mastery building.",
    "You're right! Sharp pick.",
];

pub struct AnswerOutcome {
    pub updated_concept: ConceptNode,
    pub mastery_delta: f64,
    pub reward_message: String,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub fn compute_concept_status(
    attempts: u32,
    mastery: f64,
    consecutive_correct: u32,
    wrong_count: u32,
    depth: u8,
) -> ConceptStatus {
    if attempts == 0 {
        return ConceptStatus::Unseen;
    }
    if mastery >= 0.88 && consecutive_correct >= 3 && depth >= 2 {
        return ConceptStatus::Deepened;
    }
    if mastery >= 0.78 && consecutive_correct >= 2 {
        return ConceptStatus::Mastered;
    }
    if consecutive_correct == 0 && wrong_count > 0 {
        return ConceptStatus::NeedsReview;
    }
    ConceptStatus::Learning
}

/// Pure SM-2 inspired mastery update for a single concept.
pub fn apply_answer_outcome(
    concept: &ConceptNode,
    is_correct: bool,
    angle_summary: &str,
    misconception_note: Option<&str>,
    now_ms: i64,
) -> AnswerOutcome {
    let prev_mastery = concept.mastery;
    let attempts = concept.attempts + 1;
    let correct_count = concept.correct_count + if is_correct { 1 } else { 0 };
    let wrong_count = concept.wrong_count + if is_correct { 0 } else { 1 };
    let consecutive_correct = if is_correct {
        concept.consecutive_correct + 1
    } else {
        0
    };

    let (mastery, ease_factor, interval_hours, mut misconceptions) = if is_correct {
        let streak_boost = (consecutive_correct.min(4) as f64) * 0.06;
        let gain = ((1.0 - prev_mastery) * (0.34 + streak_boost)).max(0.08);
        let next_mastery = round3((prev_mastery + gain).min(1.0));
        let next_ease = round3((concept.ease_factor + 0.1).min(3.0));
        let next_interval = match consecutive_correct {
            1 => 2.0,
            2 => 12.0,
            _ => (concept.interval_hours.max(12.0) * next_ease).round().min(24.0 * 30.0),
        };
        let mut mis = concept.misconceptions.clone();
        if consecutive_correct >= 2 {
            mis.clear();
        } else if !mis.is_empty() {
            mis.remove(0);
        }
        (next_mastery, next_ease, next_interval, mis)
    } else {
        let next_mastery = round3((prev_mastery * 0.55).max(0.05));
        let next_ease = round3((concept.ease_factor - 0.25).max(1.3));
        // Re-test quickly (~2 minutes) within the active session
        let next_interval = 0.03;
        let mut mis = concept.misconceptions.clone();
        if let Some(note) = misconception_note {
            let trimmed: String = note.trim().chars().take(140).collect();
            if !trimmed.is_empty() {
                mis.push(trimmed);
                if mis.len() > 3 {
                    mis.drain(0..(mis.len() - 3));
                }
            }
        }
        (next_mastery, next_ease, next_interval, mis)
    };

    let next_review_at = now_ms + (interval_hours * HOUR_MS).round() as i64;
    let mut recent_angles = concept.recent_angles.clone();
    let trimmed_angle: String = angle_summary.trim().chars().take(120).collect();
    if !trimmed_angle.is_empty() {
        recent_angles.push(trimmed_angle);
        if recent_angles.len() > 4 {
            recent_angles.drain(0..(recent_angles.len() - 4));
        }
    }

    let status = compute_concept_status(attempts, mastery, consecutive_correct, wrong_count, concept.depth);
    let updated_concept = ConceptNode {
        mastery,
        ease_factor,
        interval_hours,
        attempts,
        correct_count,
        wrong_count,
        consecutive_correct,
        status,
        misconceptions: {
            if misconceptions.len() > 3 {
                misconceptions.drain(0..(misconceptions.len() - 3));
            }
            misconceptions
        },
        recent_angles,
        last_seen_at: Some(now_ms),
        next_review_at: Some(next_review_at),
        ..concept.clone()
    };

    let mastery_delta = round3(mastery - prev_mastery);
    let reward_message = if is_correct {
        REWARD_MESSAGES[(attempts as usize) % REWARD_MESSAGES.len()].to_string()
    } else {
        "Not quite — study the explanation below and we'll re-test this from another angle soon."
            .to_string()
    };

    AnswerOutcome {
        updated_concept,
        mastery_delta,
        reward_message,
    }
}

/// Deterministic Learning Scheduler:
/// 1. Due mistake reviews (`review_mistake`)
/// 2. Due spaced-repetition reinforcements (`reinforce`)
/// 3. Interleaved introduction of unseen concepts (`introduce`) & active learning (`reinforce`)
/// 4. Deepening (`dig_deeper`) & triggering Knowledge Map expansion once fundamentals are mastered
pub fn select_next_directive(
    topic: &TopicSummary,
    concepts: &[ConceptNode],
    now_ms: i64,
    exclude_ids: &HashSet<String>,
) -> Result<SchedulerDirective> {
    if concepts.is_empty() {
        return Err(anyhow!("Cannot schedule a question without concepts in the knowledge map."));
    }

    let filtered: Vec<&ConceptNode> = concepts
        .iter()
        .filter(|c| !exclude_ids.contains(&c.id))
        .collect();
    let candidates: Vec<&ConceptNode> = if filtered.is_empty() {
        concepts.iter().collect()
    } else {
        filtered
    };

    let mastered_concepts: Vec<String> = concepts
        .iter()
        .filter(|c| matches!(c.status, ConceptStatus::Mastered | ConceptStatus::Deepened))
        .take(8)
        .map(|c| c.title.clone())
        .collect();

    let struggling_concepts: Vec<StrugglingSummary> = concepts
        .iter()
        .filter(|c| c.status == ConceptStatus::NeedsReview || !c.misconceptions.is_empty())
        .take(4)
        .map(|c| StrugglingSummary {
            title: c.title.clone(),
            misconceptions: c.misconceptions.clone(),
        })
        .collect();

    let make_directive = |pick: &ConceptNode,
                          intent: QuestionIntent,
                          target_depth: u8,
                          reason: String,
                          requires_new_concepts: bool| SchedulerDirective {
        intent,
        concept: pick.clone(),
        target_depth,
        reason,
        requires_new_concepts,
        compressed_context: CompressedContext {
            topic_title: topic.title.clone(),
            level: topic.level,
            mastered_concepts: mastered_concepts.clone(),
            struggling_concepts: struggling_concepts.clone(),
            recent_angles: pick.recent_angles.clone(),
        },
    };

    // 1. Due mistake reviews
    let mut due_mistakes: Vec<&ConceptNode> = candidates
        .iter()
        .copied()
        .filter(|c| {
            c.status == ConceptStatus::NeedsReview
                && c.next_review_at.map_or(true, |ts| ts <= now_ms)
        })
        .collect();
    due_mistakes.sort_by(|a, b| {
        a.mastery
            .total_cmp(&b.mastery)
            .then_with(|| a.last_seen_at.unwrap_or(0).cmp(&b.last_seen_at.unwrap_or(0)))
    });
    if let Some(pick) = due_mistakes.first() {
        return Ok(make_directive(
            pick,
            QuestionIntent::ReviewMistake,
            pick.depth,
            format!("Re-testing '{}' from a fresh angle to clear earlier mistakes.", pick.title),
            false,
        ));
    }

    // 2. Due spaced-repetition checks for mastered concepts
    let mut due_reinforcements: Vec<&ConceptNode> = candidates
        .iter()
        .copied()
        .filter(|c| {
            matches!(c.status, ConceptStatus::Mastered | ConceptStatus::Deepened)
                && c.next_review_at.map_or(false, |ts| ts <= now_ms)
        })
        .collect();
    due_reinforcements.sort_by_key(|c| c.next_review_at.unwrap_or(0));
    if let Some(pick) = due_reinforcements.first() {
        let depth = (pick.depth + if pick.mastery >= 0.9 { 1 } else { 0 }).min(4);
        return Ok(make_directive(
            pick,
            QuestionIntent::Reinforce,
            depth,
            format!("Spaced-repetition check on '{}' to verify retention.", pick.title),
            false,
        ));
    }

    // 3. Interleave unseen introductions (keep <= 3 active learning concepts at a time)
    let mut in_progress: Vec<&ConceptNode> = candidates
        .iter()
        .copied()
        .filter(|c| c.status == ConceptStatus::Learning)
        .collect();
    in_progress.sort_by(|a, b| {
        a.last_seen_at
            .unwrap_or(0)
            .cmp(&b.last_seen_at.unwrap_or(0))
            .then_with(|| a.mastery.total_cmp(&b.mastery))
    });

    let mut unseen: Vec<&ConceptNode> = candidates
        .iter()
        .copied()
        .filter(|c| c.status == ConceptStatus::Unseen)
        .collect();
    unseen.sort_by(|a, b| a.depth.cmp(&b.depth).then_with(|| a.title.cmp(&b.title)));

    if !unseen.is_empty() && in_progress.len() < 3 {
        let pick = unseen[0];
        return Ok(make_directive(
            pick,
            QuestionIntent::Introduce,
            pick.depth,
            format!("Introducing '{}' (depth {}) to expand your knowledge map.", pick.title, pick.depth),
            false,
        ));
    }

    if let Some(pick) = in_progress.first() {
        return Ok(make_directive(
            pick,
            QuestionIntent::Reinforce,
            pick.depth,
            format!("Strengthening '{}' ({}% mastery).", pick.title, (pick.mastery * 100.0).round() as u32),
            false,
        ));
    }

    // Fallback to mistakes still in brief cooldown if nothing else is active
    let mut pending_mistakes: Vec<&ConceptNode> = candidates
        .iter()
        .copied()
        .filter(|c| c.status == ConceptStatus::NeedsReview)
        .collect();
    pending_mistakes.sort_by_key(|c| c.last_seen_at.unwrap_or(0));
    if let Some(pick) = pending_mistakes.first() {
        return Ok(make_directive(
            pick,
            QuestionIntent::ReviewMistake,
            pick.depth,
            format!("Revisiting '{}' from a new angle to lock it in.", pick.title),
            false,
        ));
    }

    // 4. Everything mastered -> dig deeper and expand map if room remains
    let mut sorted = candidates;
    sorted.sort_by(|a, b| {
        a.mastery
            .total_cmp(&b.mastery)
            .then_with(|| a.last_seen_at.unwrap_or(0).cmp(&b.last_seen_at.unwrap_or(0)))
    });
    let pick = sorted[0];
    let all_mastered = concepts
        .iter()
        .all(|c| matches!(c.status, ConceptStatus::Mastered | ConceptStatus::Deepened));
    let max_depth = concepts.iter().map(|c| c.depth).max().unwrap_or(1);
    let target_depth = (pick.depth + 1).min(4);

    Ok(make_directive(
        pick,
        QuestionIntent::DigDeeper,
        target_depth,
        format!("High confidence across concepts — digging deeper on '{}' (depth {}).", pick.title, target_depth),
        all_mastered && max_depth < 4,
    ))
}

pub fn summarize_stats(concepts: &[ConceptNode]) -> KnowledgeStats {
    let mut stats = KnowledgeStats::default();
    for c in concepts {
        match c.status {
            ConceptStatus::Unseen => stats.unseen += 1,
            ConceptStatus::Learning => stats.learning += 1,
            ConceptStatus::NeedsReview => stats.needs_review += 1,
            ConceptStatus::Mastered => stats.mastered += 1,
            ConceptStatus::Deepened => stats.deepened += 1,
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SkillLevel;

    fn sample_topic() -> TopicSummary {
        TopicSummary {
            id: "t1".into(),
            user_id: "u1".into(),
            title: "Distributed Systems".into(),
            level: SkillLevel::Intermediate,
            concept_count: 2,
            mastered_count: 0,
            average_mastery: 0.0,
            total_attempts: 0,
            correct_attempts: 0,
            current_streak: 0,
            best_streak: 0,
            created_at: 1000,
            updated_at: 1000,
        }
    }

    fn sample_concept(id: &str, title: &str, status: ConceptStatus) -> ConceptNode {
        ConceptNode {
            id: id.into(),
            topic_id: "t1".into(),
            slug: id.into(),
            title: title.into(),
            subtopic: "Core".into(),
            summary: "Summary".into(),
            depth: 1,
            mastery: 0.0,
            ease_factor: 2.5,
            interval_hours: 0.0,
            attempts: 0,
            correct_count: 0,
            wrong_count: 0,
            consecutive_correct: 0,
            status,
            misconceptions: vec![],
            recent_angles: vec![],
            last_seen_at: None,
            next_review_at: None,
        }
    }

    #[test]
    fn updates_mastery_and_prunes_misconceptions() {
        let mut c = sample_concept("c1", "Raft", ConceptStatus::NeedsReview);
        c.attempts = 1;
        c.wrong_count = 1;
        c.mastery = 0.25;
        c.misconceptions = vec!["Confused timeout with heartbeat".into()];

        let r1 = apply_answer_outcome(&c, true, "election timeout", None, 10_000);
        assert!(r1.mastery_delta > 0.0);
        assert_eq!(r1.updated_concept.consecutive_correct, 1);
        assert!(r1.updated_concept.misconceptions.is_empty());

        let r2 = apply_answer_outcome(
            &r1.updated_concept,
            false,
            "split brain",
            Some("Thought minority quorum can commit"),
            20_000,
        );
        assert!(r2.mastery_delta < 0.0);
        assert_eq!(r2.updated_concept.status, ConceptStatus::NeedsReview);
        assert_eq!(r2.updated_concept.misconceptions.len(), 1);
    }

    #[test]
    fn prioritizes_mistakes_then_reinforcements_then_unseen_then_deep_dive() {
        let topic = sample_topic();
        let now = 1_000_000;
        let empty = HashSet::new();

        let mut mistake = sample_concept("m1", "Vector Clocks", ConceptStatus::NeedsReview);
        mistake.next_review_at = Some(now - 500);
        let unseen = sample_concept("u1", "CAP Theorem", ConceptStatus::Unseen);

        let d1 = select_next_directive(&topic, &[unseen.clone(), mistake], now, &empty).unwrap();
        assert_eq!(d1.intent, QuestionIntent::ReviewMistake);
        assert_eq!(d1.concept.id, "m1");

        let mut mastered = sample_concept("r1", "Idempotency", ConceptStatus::Mastered);
        mastered.mastery = 0.85;
        mastered.next_review_at = Some(now - 100);
        let d2 = select_next_directive(&topic, &[unseen, mastered.clone()], now, &empty).unwrap();
        assert_eq!(d2.intent, QuestionIntent::Reinforce);
        assert_eq!(d2.concept.id, "r1");

        mastered.next_review_at = Some(now + 999_999);
        let d3 = select_next_directive(&topic, &[mastered], now, &empty).unwrap();
        assert_eq!(d3.intent, QuestionIntent::DigDeeper);
        assert_eq!(d3.target_depth, 2);
        assert!(d3.requires_new_concepts);
    }
}
