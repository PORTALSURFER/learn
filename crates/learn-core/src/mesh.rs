use std::collections::{HashMap, HashSet};

use crate::models::{
    BridgeBrief, CognitiveSlice, ConceptEdge, ConceptNode, ConceptStatus, ConfusionPairBrief,
    CrossTopicAnchor, DistractorTrapKind, EdgeKind, FrontierEntry, HypothesisBrief,
    MisconceptionNode, QuestionChoice, QuestionIntent, TensionSeam,
};

const HOUR_MS: f64 = 3_600_000.0;
const MIN_SIGMA: f64 = 0.08;
const MAX_SIGMA: f64 = 0.95;
const PRUNE_HYPOTHESIS_PROB: f64 = 0.15;
const PRUNE_CONFUSION_WEIGHT: f64 = 0.10;

pub const DEFAULT_UNSEEN_MU: f64 = 0.50;
pub const DEFAULT_UNSEEN_SIGMA: f64 = 0.90;

const REWARD_MESSAGES: &[&str] = &[
    "You're right! Solid recall.",
    "You're right! That mental model is locking in.",
    "You're right! Clean reasoning.",
    "You're right! Real mastery building.",
    "You're right! Sharp pick.",
];

pub struct MeshObservation<'a> {
    pub target_concept_id: &'a str,
    pub intent: QuestionIntent,
    pub is_correct: bool,
    pub angle_summary: &'a str,
    pub selected_choice: Option<&'a QuestionChoice>,
    pub all_choices: &'a [QuestionChoice],
    pub fallback_misconception: Option<&'a str>,
    pub now_ms: i64,
}

pub struct MeshPropagationResult {
    pub updated_concepts: Vec<ConceptNode>,
    pub updated_edges: Vec<ConceptEdge>,
    pub updated_misconceptions: Vec<MisconceptionNode>,
    pub pruned_misconception_ids: Vec<String>,
    pub pruned_edge_ids: Vec<String>,
    pub target_concept: ConceptNode,
    pub mastery_delta: f64,
    pub reward_message: String,
    pub effects: Vec<String>,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub fn slugify_label(input: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
        if slug.len() >= 64 {
            break;
        }
    }
    slug.trim_matches('-').to_string()
}

pub fn compute_mesh_status(node: &ConceptNode) -> ConceptStatus {
    if node.attempts == 0 && node.uncertainty >= 0.80 {
        return ConceptStatus::Unseen;
    }
    if node.mastery >= 0.86 && node.uncertainty <= 0.30 && node.consecutive_correct >= 3 && node.depth >= 2 {
        return ConceptStatus::Deepened;
    }
    if node.mastery >= 0.76 && node.uncertainty <= 0.42 && node.consecutive_correct >= 2 {
        return ConceptStatus::Mastered;
    }
    if node.consecutive_correct == 0 && node.wrong_count > 0 {
        return ConceptStatus::NeedsReview;
    }
    ConceptStatus::Learning
}

/// Pure, deterministic 1-hop Cognitive Mesh belief propagation.
/// Updates `(μ, σ, S)` on the target concept, backpropagates damped credit/doubt + associative
/// stability along `Requires` edges, updates inhibitory `Confuses` edges and `MisconceptionNode`
/// posteriors from pre-tagged distractor traps, and recomputes predictive-coding edge tension.
pub fn propagate_answer_observation(
    concepts: &[ConceptNode],
    edges: &[ConceptEdge],
    misconceptions: &[MisconceptionNode],
    obs: MeshObservation<'_>,
) -> MeshPropagationResult {
    let mut nodes: Vec<ConceptNode> = concepts.to_vec();
    let mut edge_list: Vec<ConceptEdge> = edges.to_vec();
    let mut shadow_list: Vec<MisconceptionNode> = misconceptions.to_vec();
    let mut effects: Vec<String> = Vec::new();
    let mut touched_concept_ids: HashSet<String> = HashSet::new();

    let target_idx = nodes
        .iter()
        .position(|c| c.id == obs.target_concept_id)
        .expect("target concept must exist in slice");

    let evidence_weight = match obs.intent {
        QuestionIntent::DigDeeper => 1.20,
        QuestionIntent::Reinforce => 1.05,
        QuestionIntent::ReviewMistake => 1.00,
        QuestionIntent::Introduce => 0.90,
    };

    // 1. Update target node (μ, σ, S)
    let prev_mu = nodes[target_idx].mastery;
    let prev_sigma = nodes[target_idx].uncertainty;
    let target_slug = nodes[target_idx].slug.clone();
    let target_title = nodes[target_idx].title.clone();
    let target_topic_id = nodes[target_idx].topic_id.clone();

    let prediction_error = (if obs.is_correct { 1.0 } else { 0.0 }) - prev_mu;
    // Kalman-like gain scaled by epistemic uncertainty σ
    let gain = (0.28 + 0.42 * prev_sigma) * evidence_weight;

    {
        let node = &mut nodes[target_idx];
        node.attempts += 1;
        if obs.is_correct {
            node.correct_count += 1;
            node.consecutive_correct += 1;
            let raw_mu = prev_mu + gain * (1.0 - prev_mu).max(0.12);
            node.mastery = round3(raw_mu.clamp(0.05, 0.99));
            // Surprising success on a previously weak node shrinks uncertainty slightly slower
            let sigma_decay = if prediction_error > 0.55 { 0.78 } else { 0.68 };
            node.uncertainty = round3((prev_sigma * sigma_decay).clamp(MIN_SIGMA, MAX_SIGMA));
            node.ease_factor = round3((node.ease_factor + 0.1).min(3.0));
            let next_stability = match node.consecutive_correct {
                1 => 2.5,
                2 => 14.0,
                _ => (node.stability.max(14.0) * node.ease_factor).round().min(24.0 * 30.0),
            };
            node.stability = next_stability;
            node.interval_hours = next_stability;
            if node.consecutive_correct >= 2 {
                node.misconceptions.clear();
            } else if !node.misconceptions.is_empty() {
                node.misconceptions.remove(0);
            }
        } else {
            node.wrong_count += 1;
            node.consecutive_correct = 0;
            let raw_mu = prev_mu - gain * prev_mu.max(0.25);
            node.mastery = round3(raw_mu.clamp(0.05, 0.95));
            // Surprising failure on a previously confident node slightly widens σ (signals fragile mental model!)
            let next_sigma = if prev_mu >= 0.70 && prev_sigma <= 0.40 {
                (prev_sigma + 0.18).min(0.65)
            } else {
                (prev_sigma * 0.82).clamp(MIN_SIGMA, MAX_SIGMA)
            };
            node.uncertainty = round3(next_sigma);
            node.ease_factor = round3((node.ease_factor - 0.25).max(1.3));
            node.stability = 0.05;
            node.interval_hours = 0.03; // Re-test within ~2 minutes

            if let Some(note) = obs.fallback_misconception {
                let clean: String = note.trim().chars().take(140).collect();
                if !clean.is_empty() {
                    node.misconceptions.push(clean);
                    if node.misconceptions.len() > 3 {
                        node.misconceptions.drain(0..(node.misconceptions.len() - 3));
                    }
                }
            }
        }

        let trimmed_angle: String = obs.angle_summary.trim().chars().take(120).collect();
        if !trimmed_angle.is_empty() {
            node.recent_angles.push(trimmed_angle);
            if node.recent_angles.len() > 4 {
                node.recent_angles.drain(0..(node.recent_angles.len() - 4));
            }
        }
        node.last_seen_at = Some(obs.now_ms);
        node.next_review_at = Some(obs.now_ms + (node.interval_hours * HOUR_MS).round() as i64);
        node.status = compute_mesh_status(node);
        touched_concept_ids.insert(node.id.clone());
    }

    let mastery_delta = round3(nodes[target_idx].mastery - prev_mu);

    // 2. 1-Hop Damped Backpropagation + Associative Stability Re-activation along `Requires` edges
    let slug_to_idx: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.slug.clone(), i))
        .collect();

    if obs.is_correct {
        // Credit propagates backward to prerequisites: `prereq --Requires--> target`
        let prereqs: Vec<(usize, f64)> = edge_list
            .iter()
            .filter(|e| e.kind == EdgeKind::Requires && e.to_slug == target_slug)
            .filter_map(|e| slug_to_idx.get(&e.from_slug).map(|&idx| (idx, e.weight)))
            .collect();

        for (p_idx, weight) in prereqs {
            if p_idx == target_idx {
                continue;
            }
            let p = &mut nodes[p_idx];
            let boost = round3(0.20 * weight * (1.0 - p.mastery).max(0.05));
            let old_p_mu = p.mastery;
            p.mastery = round3((p.mastery + boost).clamp(0.05, 0.99));
            p.uncertainty = round3((p.uncertainty * (1.0 - 0.18 * weight)).clamp(MIN_SIGMA, MAX_SIGMA));
            // Associative memory re-activation: rehearsing downstream slows upstream decay
            if p.stability > 0.0 {
                p.stability = round3((p.stability * (1.0 + 0.15 * weight)).min(24.0 * 30.0));
                p.interval_hours = p.stability;
                if let Some(last) = p.last_seen_at {
                    p.next_review_at = Some(last + (p.interval_hours * HOUR_MS).round() as i64);
                }
            }
            p.status = compute_mesh_status(p);
            touched_concept_ids.insert(p.id.clone());
            if p.mastery - old_p_mu >= 0.03 {
                effects.push(format!(
                    "Backpropagated +{}% credit & associative refresh to prerequisite '{}'",
                    ((p.mastery - old_p_mu) * 100.0).round() as i32,
                    p.title
                ));
            }
        }
    } else {
        // Doubt propagates forward to dependents: `target --Requires--> dependent`
        let dependents: Vec<(usize, f64)> = edge_list
            .iter()
            .filter(|e| e.kind == EdgeKind::Requires && e.from_slug == target_slug)
            .filter_map(|e| slug_to_idx.get(&e.to_slug).map(|&idx| (idx, e.weight)))
            .collect();

        for (d_idx, weight) in dependents {
            if d_idx == target_idx {
                continue;
            }
            let d = &mut nodes[d_idx];
            // Raise epistemic uncertainty on downstream nodes that rely on a shaky prerequisite
            d.uncertainty = round3((d.uncertainty + 0.14 * weight).clamp(MIN_SIGMA, MAX_SIGMA));
            if d.mastery > 0.60 {
                d.mastery = round3((d.mastery - 0.08 * weight).max(0.20));
            }
            d.status = compute_mesh_status(d);
            touched_concept_ids.insert(d.id.clone());
            effects.push(format!(
                "Flagged downstream tension on '{}' (relies on '{}')",
                d.title, target_title
            ));
        }
    }

    // 3. Inhibitory `Confuses` edges & Shadow Misconception Nodes from pre-tagged distractors
    let mut pruned_edge_ids = Vec::new();
    let mut pruned_misconception_ids = Vec::new();

    if !obs.is_correct {
        if let Some(choice) = obs.selected_choice {
            match (choice.trap_kind, choice.trap_target.as_deref()) {
                (Some(DistractorTrapKind::ConfusesConcept), Some(raw_other)) if !raw_other.trim().is_empty() => {
                    let other_slug = resolve_concept_slug(&nodes, raw_other);
                    if other_slug != target_slug && !other_slug.is_empty() {
                        let other_label = nodes
                            .iter()
                            .find(|c| c.slug == other_slug)
                            .map(|c| c.title.clone())
                            .unwrap_or_else(|| raw_other.trim().to_string());

                        let (a_slug, a_label, b_slug, b_label) = if target_slug < other_slug {
                            (target_slug.clone(), target_title.clone(), other_slug.clone(), other_label.clone())
                        } else {
                            (other_slug.clone(), other_label.clone(), target_slug.clone(), target_title.clone())
                        };

                        if let Some(edge) = edge_list.iter_mut().find(|e| {
                            e.kind == EdgeKind::Confuses
                                && ((e.from_slug == a_slug && e.to_slug == b_slug)
                                    || (e.from_slug == b_slug && e.to_slug == a_slug))
                        }) {
                            edge.weight = round3((edge.weight + 0.30).min(1.0));
                            edge.note = format!("Mixed up '{}' with '{}'", target_title, other_label);
                        } else {
                            edge_list.push(ConceptEdge {
                                id: format!("confuses:{a_slug}:{b_slug}"),
                                topic_id: target_topic_id.clone(),
                                from_slug: a_slug,
                                from_label: a_label,
                                to_slug: b_slug,
                                to_label: b_label,
                                kind: EdgeKind::Confuses,
                                weight: 0.45,
                                tension: 0.0,
                                note: format!("Mixed up '{}' with '{}'", target_title, other_label),
                            });
                        }
                        effects.push(format!(
                            "Strengthened inhibitory confusion link: '{}' <-> '{}'",
                            target_title, other_label
                        ));
                    }
                }
                (Some(DistractorTrapKind::Misconception), Some(raw_hyp)) if !raw_hyp.trim().is_empty() => {
                    let label = raw_hyp.trim().chars().take(120).collect::<String>();
                    let h_slug = slugify_label(&label);
                    if !h_slug.is_empty() {
                        if let Some(sh) = shadow_list.iter_mut().find(|m| m.slug == h_slug) {
                            sh.probability = round3((sh.probability + (1.0 - sh.probability) * 0.45).min(0.98));
                            sh.evidence_count += 1;
                            sh.refute_count = 0;
                            if !sh.linked_concepts.contains(&target_slug) {
                                sh.linked_concepts.push(target_slug.clone());
                            }
                            sh.updated_at = obs.now_ms;
                            effects.push(format!(
                                "Shadow hypothesis '{}' rose to {}%",
                                sh.label,
                                (sh.probability * 100.0).round() as u32
                            ));
                        } else {
                            let prob = 0.58;
                            shadow_list.push(MisconceptionNode {
                                id: format!("hyp:{target_topic_id}:{h_slug}"),
                                topic_id: target_topic_id.clone(),
                                slug: h_slug,
                                label: label.clone(),
                                probability: prob,
                                evidence_count: 1,
                                refute_count: 0,
                                linked_concepts: vec![target_slug.clone()],
                                updated_at: obs.now_ms,
                            });
                            effects.push(format!(
                                "Activated shadow hypothesis '{}' ({}%)",
                                label,
                                (prob * 100.0).round() as u32
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    } else {
        // Correct answer: weaken inhibitory `Confuses` edges touching this concept & refute linked hypotheses!
        for edge in &mut edge_list {
            if edge.kind == EdgeKind::Confuses
                && (edge.from_slug == target_slug || edge.to_slug == target_slug)
            {
                edge.weight = round3(edge.weight * 0.65);
                if edge.weight < PRUNE_CONFUSION_WEIGHT {
                    pruned_edge_ids.push(edge.id.clone());
                    effects.push(format!(
                        "Cleared confusion link between '{}' and '{}'",
                        edge.from_label, edge.to_label
                    ));
                }
            }
        }
        edge_list.retain(|e| !pruned_edge_ids.contains(&e.id));

        // Any misconception distractor the learner successfully avoided (or linked to this concept) loses posterior mass
        let avoided_hyp_slugs: HashSet<String> = obs
            .all_choices
            .iter()
            .filter(|c| c.trap_kind == Some(DistractorTrapKind::Misconception))
            .filter_map(|c| c.trap_target.as_deref())
            .map(slugify_label)
            .collect();

        for sh in &mut shadow_list {
            let linked = sh.linked_concepts.contains(&target_slug);
            let avoided = avoided_hyp_slugs.contains(&sh.slug);
            if linked || avoided {
                sh.probability = round3(sh.probability * 0.52);
                sh.refute_count += 1;
                sh.updated_at = obs.now_ms;
                if sh.probability < PRUNE_HYPOTHESIS_PROB || sh.refute_count >= 2 {
                    pruned_misconception_ids.push(sh.id.clone());
                    effects.push(format!("Refuted shadow hypothesis '{}'", sh.label));
                }
            }
        }
        shadow_list.retain(|m| !pruned_misconception_ids.contains(&m.id));
    }

    // 4. Recompute predictive-coding Tension on every `Requires` edge
    recompute_edge_tensions(&nodes, &mut edge_list);

    let updated_concepts = nodes
        .iter()
        .filter(|c| touched_concept_ids.contains(&c.id))
        .cloned()
        .collect();
    let target_concept = nodes[target_idx].clone();

    let reward_message = if obs.is_correct {
        REWARD_MESSAGES[(target_concept.attempts as usize) % REWARD_MESSAGES.len()].to_string()
    } else {
        "Not quite — study the explanation below and we'll probe this seam from a fresh angle."
            .to_string()
    };

    MeshPropagationResult {
        updated_concepts,
        updated_edges: edge_list,
        updated_misconceptions: shadow_list,
        pruned_misconception_ids,
        pruned_edge_ids,
        target_concept,
        mastery_delta,
        reward_message,
        effects,
    }
}

/// Computes predictive-coding tension on `Requires` edges (`from -> to`).
/// High tension emerges when observed performance across connected nodes violates expected consistency:
/// - `fragile_mental_model`: downstream mastered/high while prerequisite is struggling/low
/// - `transfer_gap`: prerequisite solidly mastered while downstream is actively struggling
pub fn recompute_edge_tensions(concepts: &[ConceptNode], edges: &mut [ConceptEdge]) {
    let by_slug: HashMap<&str, &ConceptNode> =
        concepts.iter().map(|c| (c.slug.as_str(), c)).collect();

    for edge in edges.iter_mut() {
        if edge.kind != EdgeKind::Requires {
            edge.tension = 0.0;
            continue;
        }
        let (Some(from), Some(to)) = (
            by_slug.get(edge.from_slug.as_str()),
            by_slug.get(edge.to_slug.as_str()),
        ) else {
            edge.tension = 0.0;
            continue;
        };

        // Only meaningful tension once at least one side has real observations
        if from.attempts == 0 && to.attempts == 0 {
            edge.tension = 0.0;
            edge.note.clear();
            continue;
        }

        let confidence = ((1.0 - from.uncertainty) * 0.5 + (1.0 - to.uncertainty) * 0.5).clamp(0.2, 1.0);
        let diff = to.mastery - from.mastery;

        if diff > 0.22 && from.attempts > 0 && from.mastery < 0.55 {
            // Downstream high, upstream weak -> fragile/rote understanding!
            edge.tension = round3((diff * confidence * 1.25).clamp(0.0, 1.0));
            edge.note = "passed_downstream_weak_foundation".into();
        } else if diff > 0.28 && from.attempts == 0 {
            // Positive surprise: nailed advanced topic despite untested basics
            edge.tension = round3((diff * 0.7).clamp(0.0, 1.0));
            edge.note = "hidden_prior_knowledge".into();
        } else if (-diff) > 0.28 && to.attempts > 0 && to.mastery < 0.55 && from.mastery >= 0.70 {
            // Solid foundation, stuck applying it downstream
            edge.tension = round3(((-diff) * confidence * 1.1).clamp(0.0, 1.0));
            edge.note = "transfer_gap_from_foundation".into();
        } else {
            edge.tension = round3((diff.abs() * confidence * 0.35).clamp(0.0, 0.35));
            edge.note = if edge.tension < 0.20 {
                String::new()
            } else {
                "mild_mismatch".into()
            };
        }
    }
}

/// Compiles the ~150-250 token multi-resolution Cognitive Slice from the graph.
pub fn compile_cognitive_slice(
    concepts: &[ConceptNode],
    edges: &[ConceptEdge],
    misconceptions: &[MisconceptionNode],
    cross_topic_anchors: &[CrossTopicAnchor],
) -> CognitiveSlice {
    // Layer 2: Macro map pooled by subtopic
    let mut subtopic_groups: HashMap<&str, Vec<&ConceptNode>> = HashMap::new();
    for c in concepts {
        subtopic_groups.entry(c.subtopic.as_str()).or_default().push(c);
    }
    let mut subtopic_keys: Vec<&str> = subtopic_groups.keys().copied().collect();
    subtopic_keys.sort();

    let macro_map: Vec<String> = subtopic_keys
        .into_iter()
        .take(6)
        .map(|sub| {
            let group = &subtopic_groups[sub];
            let avg_mu = group.iter().map(|c| c.mastery).sum::<f64>() / group.len() as f64;
            let avg_sig = group.iter().map(|c| c.uncertainty).sum::<f64>() / group.len() as f64;
            format!("{sub}: {:.2}±{:.2}", avg_mu, avg_sig)
        })
        .collect();

    let by_slug: HashMap<&str, &ConceptNode> =
        concepts.iter().map(|c| (c.slug.as_str(), c)).collect();

    // Frontier: concepts whose prerequisites are satisfied & uncertainty is high (max info gain)
    let mut frontier_candidates: Vec<(f64, FrontierEntry)> = Vec::new();
    for c in concepts {
        if matches!(c.status, ConceptStatus::Mastered | ConceptStatus::Deepened) {
            continue;
        }
        let prereqs: Vec<&ConceptNode> = edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Requires && e.to_slug == c.slug)
            .filter_map(|e| by_slug.get(e.from_slug.as_str()).copied())
            .collect();
        let prereqs_met = prereqs.iter().all(|p| p.mastery >= 0.65 || p.attempts == 0);
        let score = c.uncertainty * 1.4
            + if c.status == ConceptStatus::NeedsReview { 0.8 } else { 0.0 }
            + if prereqs_met { 0.5 } else { -0.4 }
            - (c.depth as f64) * 0.12;
        let why = if c.status == ConceptStatus::NeedsReview {
            "needs_review_mistake"
        } else if prereqs_met && c.uncertainty >= 0.70 {
            "prereqs_met_high_uncertainty"
        } else if prereqs_met {
            "active_frontier"
        } else {
            "building_prerequisites"
        };
        frontier_candidates.push((
            score,
            FrontierEntry {
                node: c.slug.clone(),
                mu: round3(c.mastery),
                sigma: round3(c.uncertainty),
                why: why.into(),
            },
        ));
    }
    frontier_candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let frontier: Vec<FrontierEntry> = frontier_candidates
        .into_iter()
        .take(2)
        .map(|(_, e)| e)
        .collect();

    // High-tension seams
    let mut seams: Vec<&ConceptEdge> = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Requires && e.tension >= 0.25)
        .collect();
    seams.sort_by(|a, b| b.tension.total_cmp(&a.tension));
    let tension_seams: Vec<TensionSeam> = seams
        .into_iter()
        .take(2)
        .map(|e| TensionSeam {
            edge: format!("{} -> {}", e.from_slug, e.to_slug),
            tension: round3(e.tension),
            note: if e.note.is_empty() {
                "prediction_mismatch".into()
            } else {
                e.note.clone()
            },
        })
        .collect();

    // Active shadow hypotheses
    let mut hyps: Vec<&MisconceptionNode> = misconceptions
        .iter()
        .filter(|m| m.probability >= 0.25)
        .collect();
    hyps.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    let active_hypotheses: Vec<HypothesisBrief> = hyps
        .into_iter()
        .take(2)
        .map(|m| HypothesisBrief {
            misconception: m.label.clone(),
            prob: round3(m.probability),
        })
        .collect();

    // Active inhibitory confusion pairs
    let mut confusions: Vec<&ConceptEdge> = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Confuses && e.weight >= 0.20)
        .collect();
    confusions.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let confusion_pairs: Vec<ConfusionPairBrief> = confusions
        .into_iter()
        .take(2)
        .map(|e| ConfusionPairBrief {
            pair: format!("{} <-> {}", e.from_slug, e.to_slug),
            weight: round3(e.weight),
        })
        .collect();

    // Strong bridges: explicit Bridges edges + top mastered cross-topic anchors!
    let mut strong_bridges: Vec<BridgeBrief> = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Bridges)
        .take(2)
        .map(|e| BridgeBrief {
            from: e.from_label.clone(),
            to: e.to_slug.clone(),
            from_mu: round3(e.weight),
        })
        .collect();

    if strong_bridges.len() < 2 {
        let target_frontier = frontier.first().map(|f| f.node.clone()).unwrap_or_else(|| "current_topic".into());
        for anchor in cross_topic_anchors.iter().take(2 - strong_bridges.len()) {
            strong_bridges.push(BridgeBrief {
                from: format!("{}: {}", anchor.topic_title, anchor.concept_title),
                to: target_frontier.clone(),
                from_mu: round3(anchor.mastery),
            });
        }
    }

    CognitiveSlice {
        macro_map,
        frontier,
        tension_seams,
        active_hypotheses,
        confusion_pairs,
        strong_bridges,
    }
}

fn resolve_concept_slug(concepts: &[ConceptNode], raw: &str) -> String {
    let candidate = slugify_label(raw);
    if let Some(exact) = concepts.iter().find(|c| c.slug == candidate) {
        return exact.slug.clone();
    }
    let lower = raw.trim().to_lowercase();
    if let Some(by_title) = concepts.iter().find(|c| c.title.to_lowercase() == lower) {
        return by_title.slug.clone();
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, slug: &str, title: &str, depth: u8, mu: f64, sigma: f64) -> ConceptNode {
        ConceptNode {
            id: id.into(),
            topic_id: "t1".into(),
            slug: slug.into(),
            title: title.into(),
            subtopic: "Memory".into(),
            summary: "Summary".into(),
            depth,
            mastery: mu,
            uncertainty: sigma,
            stability: if mu > 0.6 { 12.0 } else { 0.0 },
            ease_factor: 2.5,
            interval_hours: 0.0,
            attempts: if sigma < 0.8 { 2 } else { 0 },
            correct_count: 0,
            wrong_count: 0,
            consecutive_correct: 0,
            status: ConceptStatus::Unseen,
            misconceptions: vec![],
            recent_angles: vec![],
            last_seen_at: Some(1_000),
            next_review_at: None,
        }
    }

    #[test]
    fn propagates_credit_back_to_prerequisite_and_refreshes_stability() {
        let prereq = node("c1", "borrowing", "Borrowing", 1, 0.50, 0.90);
        let downstream = node("c2", "struct-lifetimes", "Struct Lifetimes", 2, 0.50, 0.90);
        let edges = vec![ConceptEdge {
            id: "e1".into(),
            topic_id: "t1".into(),
            from_slug: "borrowing".into(),
            from_label: "Borrowing".into(),
            to_slug: "struct-lifetimes".into(),
            to_label: "Struct Lifetimes".into(),
            kind: EdgeKind::Requires,
            weight: 0.85,
            tension: 0.0,
            note: String::new(),
        }];

        let res = propagate_answer_observation(
            &[prereq, downstream],
            &edges,
            &[],
            MeshObservation {
                target_concept_id: "c2",
                intent: QuestionIntent::Introduce,
                is_correct: true,
                angle_summary: "lifetime annotations on structs",
                selected_choice: None,
                all_choices: &[],
                fallback_misconception: None,
                now_ms: 50_000,
            },
        );

        assert!(res.target_concept.mastery > 0.70);
        assert!(res.target_concept.uncertainty < 0.70);
        let updated_prereq = res.updated_concepts.iter().find(|c| c.id == "c1").unwrap();
        assert!(updated_prereq.mastery > 0.55);
        assert!(updated_prereq.uncertainty < 0.85);
    }

    #[test]
    fn tracks_inhibitory_confuses_edge_and_refutes_shadow_misconception() {
        let c1 = node("c1", "fn-once", "FnOnce", 2, 0.50, 0.80);
        let c2 = node("c2", "fn-mut", "FnMut", 2, 0.50, 0.80);
        let trap_confuse = QuestionChoice {
            id: "B".into(),
            text: "Can be called repeatedly while mutating state".into(),
            trap_kind: Some(DistractorTrapKind::ConfusesConcept),
            trap_target: Some("fn-mut".into()),
            why_wrong: Some("That describes FnMut, not FnOnce.".into()),
        };
        let trap_hyp = QuestionChoice {
            id: "C".into(),
            text: "Runs on a background thread automatically".into(),
            trap_kind: Some(DistractorTrapKind::Misconception),
            trap_target: Some("closures_spawn_threads".into()),
            why_wrong: Some("Closures run synchronously unless explicitly spawned.".into()),
        };

        // 1. Wrong pick triggers inhibitory Confuses edge
        let r1 = propagate_answer_observation(
            &[c1.clone(), c2.clone()],
            &[],
            &[],
            MeshObservation {
                target_concept_id: "c1",
                intent: QuestionIntent::Introduce,
                is_correct: false,
                angle_summary: "closure trait bounds",
                selected_choice: Some(&trap_confuse),
                all_choices: &[trap_confuse.clone(), trap_hyp.clone()],
                fallback_misconception: None,
                now_ms: 10_000,
            },
        );
        assert!(r1.updated_edges.iter().any(|e| e.kind == EdgeKind::Confuses && e.weight >= 0.4));

        // 2. Wrong pick on misconception activates shadow node
        let r2 = propagate_answer_observation(
            &[r1.target_concept.clone(), c2.clone()],
            &r1.updated_edges,
            &r1.updated_misconceptions,
            MeshObservation {
                target_concept_id: "c1",
                intent: QuestionIntent::ReviewMistake,
                is_correct: false,
                angle_summary: "closure execution model",
                selected_choice: Some(&trap_hyp),
                all_choices: &[trap_confuse.clone(), trap_hyp.clone()],
                fallback_misconception: None,
                now_ms: 20_000,
            },
        );
        assert_eq!(r2.updated_misconceptions.len(), 1);
        assert!(r2.updated_misconceptions[0].probability >= 0.5);

        // 3. Two correct answers refute and prune the shadow node
        let r3 = propagate_answer_observation(
            &[r2.target_concept.clone(), c2.clone()],
            &r2.updated_edges,
            &r2.updated_misconceptions,
            MeshObservation {
                target_concept_id: "c1",
                intent: QuestionIntent::ReviewMistake,
                is_correct: true,
                angle_summary: "fresh contrast check",
                selected_choice: None,
                all_choices: &[trap_confuse.clone(), trap_hyp.clone()],
                fallback_misconception: None,
                now_ms: 30_000,
            },
        );
        let r4 = propagate_answer_observation(
            &[r3.target_concept.clone(), c2.clone()],
            &r3.updated_edges,
            &r3.updated_misconceptions,
            MeshObservation {
                target_concept_id: "c1",
                intent: QuestionIntent::Reinforce,
                is_correct: true,
                angle_summary: "second verification",
                selected_choice: None,
                all_choices: &[trap_confuse, trap_hyp],
                fallback_misconception: None,
                now_ms: 40_000,
            },
        );
        assert!(r4.updated_misconceptions.is_empty());
        assert_eq!(r4.pruned_misconception_ids.len(), 1);
    }

    #[test]
    fn compiles_compact_cognitive_slice_with_tension_and_cross_topic_bridges() {
        let mut prereq = node("c1", "borrowing", "Borrowing", 1, 0.25, 0.25);
        prereq.attempts = 3;
        prereq.status = ConceptStatus::NeedsReview;
        let mut downstream = node("c2", "struct-lifetimes", "Struct Lifetimes", 2, 0.85, 0.20);
        downstream.attempts = 3;
        downstream.status = ConceptStatus::Mastered;

        let mut edges = vec![ConceptEdge {
            id: "e1".into(),
            topic_id: "t1".into(),
            from_slug: "borrowing".into(),
            from_label: "Borrowing".into(),
            to_slug: "struct-lifetimes".into(),
            to_label: "Struct Lifetimes".into(),
            kind: EdgeKind::Requires,
            weight: 0.85,
            tension: 0.0,
            note: String::new(),
        }];
        recompute_edge_tensions(&[prereq.clone(), downstream.clone()], &mut edges);
        assert!(edges[0].tension >= 0.45);
        assert_eq!(edges[0].note, "passed_downstream_weak_foundation");

        let slice = compile_cognitive_slice(
            &[prereq, downstream],
            &edges,
            &[],
            &[CrossTopicAnchor {
                topic_title: "SQL".into(),
                concept_slug: "foreign-keys".into(),
                concept_title: "Foreign Keys".into(),
                mastery: 0.92,
            }],
        );
        assert_eq!(slice.tension_seams.len(), 1);
        assert_eq!(slice.strong_bridges.len(), 1);
        assert!(slice.macro_map[0].contains("Memory:"));
    }
}
