use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};

use crate::mesh::compile_cognitive_slice;
use crate::models::{
    CompressedContext, ConceptEdge, ConceptNode, ConceptStatus, CrossTopicAnchor, EdgeKind,
    KnowledgeStats, MisconceptionNode, QuestionIntent, SchedulerDirective, StrugglingSummary,
    TopicSummary,
};

fn subtopic_diversity_adjustment(
    subtopic: &str,
    subtopic_attempts: &HashMap<String, u32>,
    total_attempts: u32,
    recent_trail: &[(String, String, QuestionIntent)],
) -> f64 {
    let target = subtopic.trim().to_lowercase();

    // 1. Recency penalty across the last 6 turns so we continually jump across pillars.
    let recency = recent_trail
        .iter()
        .take(6)
        .enumerate()
        .find(|(_, (_, recent_sub, _))| recent_sub.trim().to_lowercase() == target)
        .map(|(idx, _)| match idx {
            0 => -1.25, // Immediately previous question's pillar
            1 => -0.70, // 2 turns ago
            2 => -0.40, // 3 turns ago
            3 => -0.22, // 4 turns ago
            4 => -0.12, // 5 turns ago
            _ => -0.06,
        })
        .unwrap_or(0.75);

    // 2. Exploration vs saturation balance across pillars:
    // Untouched pillars (0 attempts) get a strong boost; heavily drilled pillars cool down.
    let attempts_here = subtopic_attempts.get(&target).copied().unwrap_or(0);
    let balance = if attempts_here == 0 {
        0.85
    } else if total_attempts >= 4 {
        let share = (attempts_here as f64) / (total_attempts as f64);
        // Penalize pillars that already hold >30% of all attempts
        -((share - 0.25).max(0.0) * 1.10).min(0.55)
    } else {
        0.0
    };

    recency + balance
}

fn is_recently_asked_concept(
    concept_id: &str,
    recent_trail: &[(String, String, QuestionIntent)],
    window: usize,
) -> bool {
    recent_trail
        .iter()
        .take(window)
        .any(|(cid, _, _)| cid == concept_id)
}

/// Detects when a topic's knowledge map is too narrow (few concepts / missing major pillars /
/// running low on unseen material) so `LearnService` can automatically broaden it without
/// forcing the learner to master 100% of one narrow subtopic first.
pub fn should_broaden_map(concepts: &[ConceptNode], total_attempts: u32) -> bool {
    if concepts.len() >= 14 {
        return false;
    }
    let distinct_subtopics: HashSet<String> = concepts
        .iter()
        .map(|c| c.subtopic.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let unseen_count = concepts
        .iter()
        .filter(|c| c.status == ConceptStatus::Unseen)
        .count();

    total_attempts >= 3 && (distinct_subtopics.len() < 5 || unseen_count <= 2)
}

/// Cognitive-Mesh-aware Interleaved Learning Scheduler:
/// Actively rotates across all major pillars/subtopics of a subject so the learner never gets
/// tunnel vision, throttling mistake reviews when unexplored pillars exist while balancing:
/// 1. High-probability active shadow misconception / inhibitory `Confuses` pair / due `NeedsReview` (throttled when fresh unseen pillars exist)
/// 2. High-tension predictive-coding seam (`tension >= 0.45`)
/// 3. Due spaced-repetition reinforcement
/// 4. Highest information-gain + freshest/unexplored-pillar frontier concept
/// 5. Automatic breadth broadening (narrow maps) & deepening (`dig_deeper`) once mastered
pub fn select_next_directive(
    topic: &TopicSummary,
    concepts: &[ConceptNode],
    edges: &[ConceptEdge],
    misconceptions: &[MisconceptionNode],
    cross_topic_anchors: &[CrossTopicAnchor],
    now_ms: i64,
    exclude_ids: &HashSet<String>,
    recent_trail: &[(String, String, QuestionIntent)],
) -> Result<SchedulerDirective> {
    if concepts.is_empty() {
        return Err(anyhow!("Cannot schedule a question without concepts in the knowledge map."));
    }

    let mut subtopic_attempts: HashMap<String, u32> = HashMap::new();
    let mut total_concept_attempts: u32 = 0;
    for c in concepts {
        *subtopic_attempts
            .entry(c.subtopic.trim().to_lowercase())
            .or_insert(0) += c.attempts;
        total_concept_attempts += c.attempts;
    }
    let effective_attempts = topic.total_attempts.max(total_concept_attempts);
    let broaden_map = should_broaden_map(concepts, effective_attempts);

    // Prefer concepts not already sitting in the open queue AND not asked in the last 2 turns.
    let mut non_excluded: Vec<&ConceptNode> = concepts
        .iter()
        .filter(|c| !exclude_ids.contains(&c.id))
        .collect();
    if non_excluded.is_empty() {
        non_excluded = concepts.iter().collect();
    }

    let spaced: Vec<&ConceptNode> = non_excluded
        .iter()
        .copied()
        .filter(|c| !is_recently_asked_concept(&c.id, recent_trail, 2))
        .collect();
    let candidates: Vec<&ConceptNode> = if !spaced.is_empty() {
        spaced
    } else {
        // If only 1-2 concepts exist total, avoid immediate back-to-back repeat if possible.
        let not_last: Vec<&ConceptNode> = non_excluded
            .iter()
            .copied()
            .filter(|c| !is_recently_asked_concept(&c.id, recent_trail, 1))
            .collect();
        if !not_last.is_empty() {
            not_last
        } else {
            non_excluded
        }
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
        requires_new_concepts: requires_new_concepts || broaden_map,
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

    let subtopic_is_just_asked = |sub: &str| {
        recent_trail
            .first()
            .map(|(_, s, _)| s.eq_ignore_ascii_case(sub))
            .unwrap_or(false)
    };

    // Throttle mistake-review spam when unexplored concepts exist in other subtopics:
    // never run two ReviewMistake turns back-to-back or >1 in the last 2 turns when fresh pillars await!
    let has_fresh_unseen_elsewhere = candidates
        .iter()
        .any(|c| c.status == ConceptStatus::Unseen && !subtopic_is_just_asked(&c.subtopic));
    let last_was_review = recent_trail
        .first()
        .map(|(_, _, intent)| *intent == QuestionIntent::ReviewMistake)
        .unwrap_or(false);
    let reviews_in_last_3 = recent_trail
        .iter()
        .take(3)
        .filter(|(_, _, intent)| *intent == QuestionIntent::ReviewMistake)
        .count();
    let allow_review_priority =
        !has_fresh_unseen_elsewhere || (!last_was_review && reviews_in_last_3 < 2);

    if allow_review_priority {
        // 1A. Diagnostic probe for a high-probability active Misconception Shadow Node (prefer fresh subtopic)
        let mut eligible_hyps: Vec<(&MisconceptionNode, &ConceptNode, f64)> = Vec::new();
        for hyp in misconceptions.iter().filter(|m| m.probability >= 0.55) {
            for slug in &hyp.linked_concepts {
                if let Some(pick) = by_slug.get(slug.as_str()).copied() {
                    if candidate_ids.contains(pick.id.as_str())
                        && (!has_fresh_unseen_elsewhere || !is_recently_asked_concept(&pick.id, recent_trail, 3))
                        && pick.next_review_at.map_or(true, |ts| ts <= now_ms)
                    {
                        let score = hyp.probability
                            + subtopic_diversity_adjustment(
                                &pick.subtopic,
                                &subtopic_attempts,
                                effective_attempts,
                                recent_trail,
                            ) * 0.5;
                        eligible_hyps.push((hyp, pick, score));
                    }
                }
            }
        }
        eligible_hyps.sort_by(|a, b| b.2.total_cmp(&a.2));
        if let Some((top_hyp, pick, _)) = eligible_hyps.first() {
            if !subtopic_is_just_asked(&pick.subtopic)
                || candidates.iter().all(|c| subtopic_is_just_asked(&c.subtopic))
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

        // 1B. Contrast question for a strong inhibitory `Confuses` synapse (prefer fresh subtopic)
        let mut eligible_confs: Vec<(&ConceptEdge, &ConceptNode, String, f64)> = Vec::new();
        for conf in edges.iter().filter(|e| e.kind == EdgeKind::Confuses && e.weight >= 0.40) {
            let a = by_slug.get(conf.from_slug.as_str()).copied();
            let b = by_slug.get(conf.to_slug.as_str()).copied();
            let target_and_other = match (a, b) {
                (Some(na), Some(nb)) if candidate_ids.contains(na.id.as_str()) => {
                    Some((na, nb.title.clone()))
                }
                (Some(na), Some(nb)) if candidate_ids.contains(nb.id.as_str()) => {
                    Some((nb, na.title.clone()))
                }
                _ => None,
            };
            if let Some((pick, other_title)) = target_and_other {
                if (!has_fresh_unseen_elsewhere || !is_recently_asked_concept(&pick.id, recent_trail, 3))
                    && pick.next_review_at.map_or(true, |ts| ts <= now_ms)
                {
                    let score = conf.weight
                        + subtopic_diversity_adjustment(
                            &pick.subtopic,
                            &subtopic_attempts,
                            effective_attempts,
                            recent_trail,
                        ) * 0.5;
                    eligible_confs.push((conf, pick, other_title, score));
                }
            }
        }
        eligible_confs.sort_by(|a, b| b.3.total_cmp(&a.3));
        if let Some((conf, pick, other_title, _)) = eligible_confs.first() {
            if !subtopic_is_just_asked(&pick.subtopic)
                || candidates.iter().all(|c| subtopic_is_just_asked(&c.subtopic))
            {
                return Ok(make_directive(
                    pick,
                    QuestionIntent::ReviewMistake,
                    pick.depth,
                    format!(
                        "Contrast question separating '{}' from '{}' (confusion weight {:.2}).",
                        pick.title, other_title, conf.weight
                    ),
                    false,
                    Some(other_title.clone()),
                    None,
                ));
            }
        }

        // 1C. Due mistake reviews — strongly prefer mistakes from a DIFFERENT subtopic than the last question
        let mut due_mistakes: Vec<(&ConceptNode, f64)> = candidates
            .iter()
            .copied()
            .filter(|c| {
                c.status == ConceptStatus::NeedsReview
                    && (!has_fresh_unseen_elsewhere || !is_recently_asked_concept(&c.id, recent_trail, 3))
                    && c.next_review_at.map_or(true, |ts| ts <= now_ms)
            })
            .map(|c| {
                let score = (1.0 - c.mastery)
                    + subtopic_diversity_adjustment(
                        &c.subtopic,
                        &subtopic_attempts,
                        effective_attempts,
                        recent_trail,
                    );
                (c, score)
            })
            .collect();
        due_mistakes.sort_by(|a, b| b.1.total_cmp(&a.1));
        if let Some((pick, _)) = due_mistakes.first() {
            if !subtopic_is_just_asked(&pick.subtopic)
                || candidates.iter().all(|c| subtopic_is_just_asked(&c.subtopic))
            {
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
        }
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
            .filter(|c| candidate_ids.contains(c.id.as_str()) && !subtopic_is_just_asked(&c.subtopic))
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
    let mut due_reinforcements: Vec<(&ConceptNode, f64)> = candidates
        .iter()
        .copied()
        .filter(|c| {
            matches!(c.status, ConceptStatus::Mastered | ConceptStatus::Deepened)
                && c.next_review_at.map_or(false, |ts| ts <= now_ms)
        })
        .map(|c| {
            (
                c,
                subtopic_diversity_adjustment(
                    &c.subtopic,
                    &subtopic_attempts,
                    effective_attempts,
                    recent_trail,
                ),
            )
        })
        .collect();
    due_reinforcements.sort_by(|a, b| b.1.total_cmp(&a.1));
    if let Some((pick, _)) = due_reinforcements.first() {
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

    // 4. Interleaved Information-Gain Frontier:
    // Strongly favors UNSEEN & active concepts in FRESH / unexplored pillars!
    let mut frontier: Vec<(f64, &ConceptNode)> = candidates
        .iter()
        .copied()
        .filter(|c| {
            matches!(
                c.status,
                ConceptStatus::Unseen | ConceptStatus::Learning | ConceptStatus::NeedsReview
            )
        })
        .map(|c| {
            let prereqs_met = edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Requires && e.to_slug == c.slug)
                .filter_map(|e| by_slug.get(e.from_slug.as_str()).copied())
                .all(|p| p.mastery >= 0.55 || p.attempts == 0);

            let diversity = subtopic_diversity_adjustment(
                &c.subtopic,
                &subtopic_attempts,
                effective_attempts,
                recent_trail,
            );
            let unseen_bonus = if c.status == ConceptStatus::Unseen { 0.45 } else { 0.0 };
            let prereq_bonus = if prereqs_met { 0.30 } else { -0.15 };
            let depth_nudge = -(c.depth as f64) * 0.05;
            let score = c.uncertainty * 1.35 + diversity + unseen_bonus + prereq_bonus + depth_nudge;
            (score, c)
        })
        .collect();
    frontier.sort_by(|a, b| b.0.total_cmp(&a.0));

    if let Some((_, pick)) = frontier.first() {
        let intent = match pick.status {
            ConceptStatus::Unseen => QuestionIntent::Introduce,
            ConceptStatus::NeedsReview => QuestionIntent::ReviewMistake,
            _ => QuestionIntent::Reinforce,
        };
        return Ok(make_directive(
            pick,
            intent,
            pick.depth,
            format!(
                "Interleaved probe on '{}' in '{}' (μ={:.2}, σ={:.2}).",
                pick.title, pick.subtopic, pick.mastery, pick.uncertainty
            ),
            false,
            None,
            None,
        ));
    }

    // 5. Everything mastered -> dig deeper and expand map if room remains
    let mut sorted: Vec<(&ConceptNode, f64)> = candidates
        .into_iter()
        .map(|c| {
            let score = c.uncertainty
                + subtopic_diversity_adjustment(
                    &c.subtopic,
                    &subtopic_attempts,
                    effective_attempts,
                    recent_trail,
                )
                - c.mastery * 0.3;
            (c, score)
        })
        .collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    let pick = sorted[0].0;
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
            title: "Hypnotic Techno Production".into(),
            level: SkillLevel::Intermediate,
            concept_count: 3,
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

    fn sample_concept(id: &str, slug: &str, title: &str, subtopic: &str, depth: u8, status: ConceptStatus) -> ConceptNode {
        ConceptNode {
            id: id.into(),
            topic_id: "t1".into(),
            slug: slug.into(),
            title: title.into(),
            subtopic: subtopic.into(),
            summary: "Summary".into(),
            depth,
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

        let c1 = sample_concept("c1", "fn-once", "FnOnce", "Closures", 1, ConceptStatus::Learning);
        let c2 = sample_concept("c2", "fn-mut", "FnMut", "Closures", 1, ConceptStatus::Unseen);

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

        let d1 = select_next_directive(&topic, &[c1.clone(), c2.clone()], &[], &[hyp], &[], now, &empty, &[]).unwrap();
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
        let d2 = select_next_directive(&topic, &[c1, c2], &[conf], &[], &[], now, &empty, &[]).unwrap();
        assert_eq!(d2.intent, QuestionIntent::ReviewMistake);
        assert!(d2.compressed_context.contrast_concept.is_some());
    }

    #[test]
    fn rotates_across_subtopics_to_prevent_tunnel_vision() {
        let topic = sample_topic();
        let now = 1_000_000;
        let empty = HashSet::new();

        let r1 = sample_concept("r1", "pulse", "Pulse", "Rhythm and groove", 1, ConceptStatus::Learning);
        let r2 = sample_concept("r2", "microtiming", "Microtiming", "Rhythm and groove", 1, ConceptStatus::Unseen);
        let s1 = sample_concept("s1", "kick-bass", "Kick & Bass Envelopes", "Sound and space", 2, ConceptStatus::Unseen);
        let p1 = sample_concept("p1", "tension", "Long-Form Tension", "Pattern and form", 3, ConceptStatus::Unseen);

        // Last question was in "Rhythm and groove" -> must jump to a different subtopic!
        let trail = vec![(
            "r1".into(),
            "Rhythm and groove".into(),
            QuestionIntent::Introduce,
        )];
        let d1 = select_next_directive(&topic, &[r1.clone(), r2.clone(), s1.clone(), p1.clone()], &[], &[], &[], now, &empty, &trail).unwrap();
        assert_ne!(d1.concept.subtopic, "Rhythm and groove");
        assert_eq!(d1.concept.id, "s1");

        // Now both Rhythm and Sound were seen -> jumps to "Pattern and form"!
        let trail2 = vec![
            ("s1".into(), "Sound and space".into(), QuestionIntent::Introduce),
            ("r1".into(), "Rhythm and groove".into(), QuestionIntent::Introduce),
        ];
        let d2 = select_next_directive(&topic, &[r1, r2, s1, p1], &[], &[], &[], now, &empty, &trail2).unwrap();
        assert_eq!(d2.concept.subtopic, "Pattern and form");
        assert_eq!(d2.concept.id, "p1");
    }

    #[test]
    fn throttles_reviews_to_explore_unseen_pillars_and_flags_narrow_map() {
        let mut topic = sample_topic();
        topic.total_attempts = 8;
        let now = 1_000_000;
        let empty = HashSet::new();

        let mut r1 = sample_concept("r1", "cycle", "Interlocking Cycle Lengths", "Rhythm & Groove", 2, ConceptStatus::NeedsReview);
        r1.attempts = 4;
        r1.mastery = 0.15;
        let mut r2 = sample_concept("r2", "accent", "Accent & Microtiming", "Rhythm & Groove", 1, ConceptStatus::Learning);
        r2.attempts = 4;
        let m1 = sample_concept("m1", "eq-carving", "Low-End EQ Carving", "Mixing & Low-End", 1, ConceptStatus::Unseen);
        let sd1 = sample_concept("sd1", "fm-timbre", "FM & Filter Modulation", "Sound Design & Synthesis", 1, ConceptStatus::Unseen);

        let hyp = MisconceptionNode {
            id: "h1".into(),
            topic_id: "t1".into(),
            slug: "accent-restarts".into(),
            label: "accent_count_restarts_each_bar".into(),
            probability: 0.77,
            evidence_count: 2,
            refute_count: 0,
            linked_concepts: vec!["cycle".into()],
            updated_at: now - 1000,
        };

        // Immediately after a ReviewMistake, scheduler must pivot to an unexplored pillar (Mixing or Sound Design)
        let trail = vec![(
            "r2".into(),
            "Rhythm & Groove".into(),
            QuestionIntent::ReviewMistake,
        )];
        let d = select_next_directive(
            &topic,
            &[r1, r2, m1, sd1],
            &[],
            &[hyp],
            &[],
            now,
            &empty,
            &trail,
        )
        .unwrap();
        assert_eq!(d.intent, QuestionIntent::Introduce);
        assert!(
            d.concept.subtopic == "Mixing & Low-End"
                || d.concept.subtopic == "Sound Design & Synthesis",
            "expected unexplored non-rhythm pillar, got {}",
            d.concept.subtopic
        );
        // Narrow 4-concept / 3-pillar map after 8 attempts must request automatic broadening
        assert!(d.requires_new_concepts);
    }
}
