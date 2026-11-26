use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};

use crate::mesh::compile_cognitive_slice;
use crate::models::{
    CompressedContext, ConceptEdge, ConceptNode, ConceptStatus, CrossTopicAnchor, EdgeKind,
    KnowledgeStats, MisconceptionNode, QuestionIntent, SchedulerDirective, StrugglingSummary,
    TopicSummary,
};

/// Cognitive-Mesh-aware Learning Scheduler:
/// 1. High-probability active shadow misconception / inhibitory `Confuses` pair / due `NeedsReview`
/// 2. High-tension predictive-coding seam (`tension >= 0.45`)
/// 3. Due spaced-repetition reinforcement
/// 4. Highest information-gain frontier concept (prerequisites met + highest epistemic uncertainty `σ`)
/// 5. Deepening (`dig_deeper`) & Knowledge Map expansion once current nodes are mastered
pub fn select_next_directive(
    topic: &TopicSummary,
    concepts: &[ConceptNode],
    edges: &[ConceptEdge],
    misconceptions: &[MisconceptionNode],
    cross_topic_anchors: &[CrossTopicAnchor],
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

    let by_slug: HashMap<&str, &ConceptNode> =
        concepts.iter().map(|c| (c.slug.as_str(), c)).collect();
    let candidate_ids: HashSet<&str> = candidates.iter().map(|c| c.id.as_str()).collect();

    let cognitive_slice =
        compile_cognitive_slice(concepts, edges, misconceptions, cross_topic_anchors);

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
                          requires_new_concepts: bool,
                          contrast_concept: Option<String>,
                          target_hypothesis: Option<String>| SchedulerDirective {
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
            cognitive_slice: cognitive_slice.clone(),
            contrast_concept,
            target_hypothesis,
        },
    };

    // 1A. Diagnostic probe for a high-probability active Misconception Shadow Node
    if let Some(top_hyp) = misconceptions
        .iter()
        .filter(|m| m.probability >= 0.55)
        .max_by(|a, b| a.probability.total_cmp(&b.probability))
    {
        if let Some(pick) = top_hyp
            .linked_concepts
            .iter()
            .filter_map(|slug| by_slug.get(slug.as_str()).copied())
            .find(|c| {
                candidate_ids.contains(c.id.as_str())
                    && c.next_review_at.map_or(true, |ts| ts <= now_ms)
            })
        {
            return Ok(make_directive(
                pick,
                QuestionIntent::ReviewMistake,
                pick.depth,
                format!(
                    "Diagnostic probe to confirm or refute shadow hypothesis '{}' ({}%).",
                    top_hyp.label,
                    (top_hyp.probability * 100.0).round() as u32
                ),
                false,
                None,
                Some(top_hyp.label.clone()),
            ));
        }
    }

    // 1B. Contrast question for a strong inhibitory `Confuses` synapse
    if let Some(conf) = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Confuses && e.weight >= 0.40)
        .max_by(|a, b| a.weight.total_cmp(&b.weight))
    {
        let a = by_slug.get(conf.from_slug.as_str()).copied();
        let b = by_slug.get(conf.to_slug.as_str()).copied();
        let target_and_other = match (a, b) {
            (Some(na), Some(nb)) if candidate_ids.contains(na.id.as_str()) => Some((na, nb.title.clone())),
            (Some(na), Some(nb)) if candidate_ids.contains(nb.id.as_str()) => Some((nb, na.title.clone())),
            _ => None,
        };
        if let Some((pick, other_title)) = target_and_other {
            if pick.next_review_at.map_or(true, |ts| ts <= now_ms) {
                return Ok(make_directive(
                    pick,
                    QuestionIntent::ReviewMistake,
                    pick.depth,
                    format!(
                        "Contrast question separating '{}' from '{}' (confusion weight {:.2}).",
                        pick.title, other_title, conf.weight
                    ),
                    false,
                    Some(other_title),
                    None,
                ));
            }
        }
    }

    // 1C. Due mistake reviews
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
            None,
            None,
        ));
    }

    // 2. High-tension predictive-coding seam (`Requires` edge where upstream & downstream disagree)
    if let Some(seam) = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Requires && e.tension >= 0.45)
        .max_by(|a, b| a.tension.total_cmp(&b.tension))
    {
        let target_slug = if seam.note == "passed_downstream_weak_foundation" {
            &seam.from_slug
        } else {
            &seam.to_slug
        };
        if let Some(pick) = by_slug
            .get(target_slug.as_str())
            .copied()
            .filter(|c| candidate_ids.contains(c.id.as_str()))
        {
            return Ok(make_directive(
                pick,
                QuestionIntent::Reinforce,
                pick.depth,
                format!(
                    "Resolving high-tension seam '{} -> {}' (tension {:.2}).",
                    seam.from_label, seam.to_label, seam.tension
                ),
                false,
                None,
                None,
            ));
        }
    }

    // 3. Due spaced-repetition checks for mastered concepts
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
        let depth = (pick.depth + if pick.mastery >= 0.90 { 1 } else { 0 }).min(4);
        return Ok(make_directive(
            pick,
            QuestionIntent::Reinforce,
            depth,
            format!("Spaced-repetition check on '{}' to verify retention.", pick.title),
            false,
            None,
            None,
        ));
    }

    // 4. Information-gain frontier: prefer concepts whose prerequisites are satisfied & σ is highest!
    let in_progress_count = candidates
        .iter()
        .filter(|c| c.status == ConceptStatus::Learning)
        .count();

    let mut frontier: Vec<(f64, &ConceptNode)> = candidates
        .iter()
        .copied()
        .filter(|c| matches!(c.status, ConceptStatus::Unseen | ConceptStatus::Learning))
        .map(|c| {
            let prereqs_met = edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Requires && e.to_slug == c.slug)
                .filter_map(|e| by_slug.get(e.from_slug.as_str()).copied())
                .all(|p| p.mastery >= 0.62 || p.attempts == 0);
            let mut score = c.uncertainty * 1.5 - (c.depth as f64) * 0.15;
            if prereqs_met {
                score += 0.60;
            }
            if c.status == ConceptStatus::Unseen && in_progress_count >= 3 {
                score -= 0.45;
            }
            (score, c)
        })
        .collect();
    frontier.sort_by(|a, b| b.0.total_cmp(&a.0));

    if let Some((_, pick)) = frontier.first() {
        let intent = if pick.status == ConceptStatus::Unseen {
            QuestionIntent::Introduce
        } else {
            QuestionIntent::Reinforce
        };
        return Ok(make_directive(
            pick,
            intent,
            pick.depth,
            format!(
                "Frontier probe on '{}' (μ={:.2}, σ={:.2}) for maximum information gain.",
                pick.title, pick.mastery, pick.uncertainty
            ),
            false,
            None,
            None,
        ));
    }

    // Fallback to mistakes still in brief cooldown
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
            None,
            None,
        ));
    }

    // 5. Everything mastered -> dig deeper and expand map if room remains
    let mut sorted = candidates;
    sorted.sort_by(|a, b| {
        b.uncertainty
            .total_cmp(&a.uncertainty)
            .then_with(|| a.mastery.total_cmp(&b.mastery))
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
        format!(
            "High confidence across concepts — digging deeper on '{}' (depth {}).",
            pick.title, target_depth
        ),
        all_mastered && max_depth < 4,
        None,
        None,
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

    fn sample_concept(id: &str, slug: &str, title: &str, status: ConceptStatus) -> ConceptNode {
        ConceptNode {
            id: id.into(),
            topic_id: "t1".into(),
            slug: slug.into(),
            title: title.into(),
            subtopic: "Core".into(),
            summary: "Summary".into(),
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
            status,
            misconceptions: vec![],
            recent_angles: vec![],
            last_seen_at: None,
            next_review_at: None,
        }
    }

    #[test]
    fn prioritizes_shadow_hypothesis_and_confusion_contrast_over_unseen() {
        let topic = sample_topic();
        let now = 1_000_000;
        let empty = HashSet::new();

        let c1 = sample_concept("c1", "fn-once", "FnOnce", ConceptStatus::Learning);
        let c2 = sample_concept("c2", "fn-mut", "FnMut", ConceptStatus::Unseen);

        let hyp = MisconceptionNode {
            id: "h1".into(),
            topic_id: "t1".into(),
            slug: "closures-spawn-threads".into(),
            label: "Closures spawn threads".into(),
            probability: 0.72,
            evidence_count: 2,
            refute_count: 0,
            linked_concepts: vec!["fn-once".into()],
            updated_at: now - 1000,
        };

        let d1 = select_next_directive(&topic, &[c1.clone(), c2.clone()], &[], &[hyp], &[], now, &empty).unwrap();
        assert_eq!(d1.intent, QuestionIntent::ReviewMistake);
        assert_eq!(d1.compressed_context.target_hypothesis.as_deref(), Some("Closures spawn threads"));

        let conf = ConceptEdge {
            id: "e1".into(),
            topic_id: "t1".into(),
            from_slug: "fn-mut".into(),
            from_label: "FnMut".into(),
            to_slug: "fn-once".into(),
            to_label: "FnOnce".into(),
            kind: EdgeKind::Confuses,
            weight: 0.65,
            tension: 0.0,
            note: "Mixed up".into(),
        };
        let d2 = select_next_directive(&topic, &[c1, c2], &[conf], &[], &[], now, &empty).unwrap();
        assert_eq!(d2.intent, QuestionIntent::ReviewMistake);
        assert!(d2.compressed_context.contrast_concept.is_some());
    }
}
