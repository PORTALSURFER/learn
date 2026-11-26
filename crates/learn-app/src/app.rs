use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Default)]
struct GlobalAudioState {
    child: Option<Child>,
    shutting_down: bool,
}

static GLOBAL_AUDIO: OnceLock<Mutex<GlobalAudioState>> = OnceLock::new();

fn global_audio_slot() -> &'static Mutex<GlobalAudioState> {
    GLOBAL_AUDIO.get_or_init(|| Mutex::new(GlobalAudioState::default()))
}

fn kill_child(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

/// Immediately kills and reaps any active `afplay` audio subprocess.
pub fn stop_global_audio() {
    if let Ok(mut guard) = global_audio_slot().lock() {
        kill_child(&mut guard.child);
    }
}

/// Permanently halts audio playback for app shutdown so no background loop can respawn `afplay`.
pub fn shutdown_global_audio() {
    if let Ok(mut guard) = global_audio_slot().lock() {
        guard.shutting_down = true;
        kill_child(&mut guard.child);
    }
}

fn is_audio_shutting_down() -> bool {
    global_audio_slot()
        .lock()
        .map(|g| g.shutting_down)
        .unwrap_or(true)
}

fn has_global_audio_running() -> bool {
    global_audio_slot()
        .lock()
        .map(|g| !g.shutting_down && g.child.is_some())
        .unwrap_or(false)
}

fn spawn_global_audio(path: &Path) -> std::io::Result<bool> {
    let mut guard = global_audio_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    kill_child(&mut guard.child);
    if guard.shutting_down {
        return Ok(false);
    }
    let child = Command::new("afplay").arg(path).spawn()?;
    guard.child = Some(child);
    Ok(true)
}

fn global_audio_exited() -> bool {
    if let Ok(mut guard) = global_audio_slot().lock() {
        if guard.shutting_down {
            kill_child(&mut guard.child);
            return true;
        }
        match guard.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(Some(_))),
            None => true,
        }
    } else {
        true
    }
}

use gpui::{
    AnyElement, AsyncApp, ClickEvent, ClipboardItem, Context, FocusHandle, Focusable, KeyDownEvent,
    SharedString, WeakEntity, Window, div, prelude::*, px,
};
use learn_core::{
    AnswerConfidence, AnswerResult, AudioDemoSpec, ClarifyingReply, ConceptNode, ConceptStatus,
    DEFAULT_BUFFER_TARGET, DEFAULT_OPEN_QUEUE_TARGET, DetectedRunner, EdgeKind, ExampleMedia,
    KnowledgeMap, LearnService, MediaReveal, QuestionIntent, RunnerKind, SkillLevel,
    StoredQuestion, TopicSummary, User, synthesize_audio_wav_bars,
};

use crate::text_buffer::TextBuffer;
use crate::{CloseWindow, QuitApp, theme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Dashboard,
    Quiz,
    KnowledgeMap,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveField {
    None,
    AuthUsername,
    AuthPassword,
    NewTopic,
    QuizClarify,
    GeminiKey,
    OpenaiKey,
}

pub struct LearnApp {
    service: LearnService,
    focus_handle: FocusHandle,

    // Auth state
    user: Option<User>,
    known_usernames: Vec<String>,
    is_register_mode: bool,
    remember_me: bool,
    auth_username: TextBuffer,
    auth_password: TextBuffer,

    // Navigation & form state
    route: Route,
    active_field: ActiveField,
    new_topic_title: TextBuffer,
    new_topic_level: SkillLevel,
    gemini_key_input: TextBuffer,
    openai_key_input: TextBuffer,

    // Domain state
    runners: Vec<DetectedRunner>,
    topics: Vec<TopicSummary>,
    active_topic_id: Option<String>,
    knowledge_map: Option<KnowledgeMap>,
    current_question: Option<StoredQuestion>,
    selected_choice_id: Option<String>,
    hint_revealed: bool,
    last_answer: Option<AnswerResult>,

    // Audio preview playback, looping, & realtime step playhead
    audio_status: Option<String>,
    audio_showing_after: bool,
    audio_preview_spec: Option<AudioDemoSpec>,
    audio_preview_tag: Option<String>,
    audio_looping: bool,
    audio_wav_path: Option<PathBuf>,
    audio_started_at: Option<Instant>,
    audio_bpm: f32,
    audio_steps_per_bar: usize,
    audio_active_step: Option<usize>,
    audio_playhead_running: bool,
    media_busy: bool,

    // Live UI feedback
    busy_message: Option<String>,
    prefetching: bool,
    prefetch_topic_id: Option<String>,
    buffered_ready: usize,
    clarify_input: TextBuffer,
    clarify_busy: bool,
    clarify_replies: Vec<ClarifyingReply>,
    anim_tick: usize,
    anim_running: bool,
    error_banner: Option<String>,
    info_banner: Option<String>,
}

impl Drop for LearnApp {
    fn drop(&mut self) {
        shutdown_global_audio();
    }
}

impl Focusable for LearnApp {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl LearnApp {
    pub fn new(service: LearnService, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle);

        let known_usernames = service.list_usernames().unwrap_or_default();
        let is_register_mode = known_usernames.is_empty();
        let restored_user = service.restore_remembered_user().ok().flatten();
        let runners = service.detect_runners(restored_user.as_ref());
        let (topics, gemini_key_input, openai_key_input, active_field) = match &restored_user {
            Some(u) => (
                service.list_topics(&u.id).unwrap_or_default(),
                TextBuffer::from_string_or_empty(&u.gemini_api_key),
                TextBuffer::from_string_or_empty(&u.openai_api_key),
                ActiveField::NewTopic,
            ),
            None => (
                Vec::new(),
                TextBuffer::new(),
                TextBuffer::new(),
                ActiveField::AuthUsername,
            ),
        };
        let first_topic_id = topics.first().map(|t| t.id.clone());

        let mut app = Self {
            service,
            focus_handle,
            user: restored_user,
            known_usernames,
            is_register_mode,
            remember_me: true,
            auth_username: TextBuffer::new(),
            auth_password: TextBuffer::new(),
            route: Route::Dashboard,
            active_field,
            new_topic_title: TextBuffer::new(),
            new_topic_level: SkillLevel::Beginner,
            gemini_key_input,
            openai_key_input,
            runners,
            topics,
            active_topic_id: None,
            knowledge_map: None,
            current_question: None,
            selected_choice_id: None,
            hint_revealed: false,
            last_answer: None,
            audio_status: None,
            audio_showing_after: false,
            audio_preview_spec: None,
            audio_preview_tag: None,
            audio_looping: true,
            audio_wav_path: None,
            audio_started_at: None,
            audio_bpm: 128.0,
            audio_steps_per_bar: 4,
            audio_active_step: None,
            audio_playhead_running: false,
            media_busy: false,
            busy_message: None,
            prefetching: false,
            prefetch_topic_id: None,
            buffered_ready: 0,
            clarify_input: TextBuffer::new(),
            clarify_busy: false,
            clarify_replies: Vec::new(),
            anim_tick: 0,
            anim_running: false,
            error_banner: None,
            info_banner: None,
        };

        if let Some(tid) = first_topic_id {
            app.start_buffer_refill_for(tid, DEFAULT_BUFFER_TARGET, cx);
        }

        app
    }

    // ---------- Auth & Navigation Actions ----------

    fn submit_auth(&mut self, cx: &mut Context<Self>) {
        self.error_banner = None;
        self.info_banner = None;

        let res = if self.is_register_mode {
            self.service
                .register(self.auth_username.as_str(), self.auth_password.as_str())
        } else {
            self.service
                .login(self.auth_username.as_str(), self.auth_password.as_str())
        };

        match res {
            Ok(user) => {
                if self.remember_me {
                    let _ = self.service.remember_user(&user.id);
                } else {
                    let _ = self.service.clear_remembered_user();
                }
                self.gemini_key_input.set_text(user.gemini_api_key.clone());
                self.openai_key_input.set_text(user.openai_api_key.clone());
                self.runners = self.service.detect_runners(Some(&user));
                self.topics = self.service.list_topics(&user.id).unwrap_or_default();
                self.known_usernames = self.service.list_usernames().unwrap_or_default();
                self.auth_password.clear();
                self.user = Some(user);
                self.route = Route::Dashboard;
                self.active_field = ActiveField::NewTopic;
                if let Some(first_topic) = self.topics.first() {
                    self.start_buffer_refill_for(first_topic.id.clone(), DEFAULT_BUFFER_TARGET, cx);
                }
            }
            Err(err) => {
                self.error_banner = Some(err.to_string());
            }
        }
        cx.notify();
    }

    fn sign_out(&mut self, cx: &mut Context<Self>) {
        let _ = self.service.clear_remembered_user();
        self.user = None;
        self.topics.clear();
        self.active_topic_id = None;
        self.knowledge_map = None;
        self.current_question = None;
        self.selected_choice_id = None;
        self.hint_revealed = false;
        self.last_answer = None;
        self.media_busy = false;
        self.stop_audio_quiet();
        self.busy_message = None;
        self.prefetching = false;
        self.prefetch_topic_id = None;
        self.buffered_ready = 0;
        self.error_banner = None;
        self.info_banner = None;
        self.route = Route::Dashboard;
        self.is_register_mode = self.known_usernames.is_empty();
        self.active_field = ActiveField::AuthUsername;
        cx.notify();
    }

    fn refresh_topics(&mut self) {
        if let Some(user) = &self.user {
            self.topics = self.service.list_topics(&user.id).unwrap_or_default();
            if let Some(topic_id) = &self.active_topic_id {
                self.knowledge_map = self.service.get_knowledge_map(&user.id, topic_id).ok();
                self.refresh_buffer_count();
            }
        }
    }

    fn refresh_buffer_count(&mut self) {
        let Some(topic_id) = &self.active_topic_id else {
            self.buffered_ready = 0;
            return;
        };
        let open = self.service.count_open_questions(topic_id).unwrap_or(0);
        let active_open = usize::from(self.current_question.is_some() && self.last_answer.is_none());
        self.buffered_ready = open.saturating_sub(active_open).min(DEFAULT_BUFFER_TARGET);
    }

    fn desired_open_queue_size(&self, topic_id: &str) -> usize {
        if self.active_topic_id.as_deref() == Some(topic_id)
            && self.current_question.is_some()
            && self.last_answer.is_none()
        {
            DEFAULT_OPEN_QUEUE_TARGET
        } else {
            DEFAULT_BUFFER_TARGET
        }
    }

    fn is_ai_active(&self) -> bool {
        self.busy_message.is_some() || self.prefetching || self.clarify_busy || self.media_busy
    }

    /// Runs a lightweight 120ms pulse timer ONLY while AI generation/prefetch/tutor is active.
    /// Automatically exits as soon as all AI work completes (0% CPU when idle).
    fn ensure_anim_loop(&mut self, cx: &mut Context<Self>) {
        if self.anim_running || !self.is_ai_active() {
            return;
        }
        self.anim_running = true;

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(120))
                    .await;

                let mut keep_running = false;
                let updated = this.update(cx, |view, cx| {
                    if view.is_ai_active() {
                        view.anim_tick = view.anim_tick.wrapping_add(1);
                        keep_running = true;
                        cx.notify();
                    } else {
                        view.anim_running = false;
                    }
                });

                if updated.is_err() || !keep_running {
                    break;
                }
            }
        })
        .detach();
    }

    // ---------- Async AI Operations (off UI thread!) ----------

    fn create_topic(&mut self, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        let title = self.new_topic_title.as_str().trim().to_string();
        if title.is_empty() {
            self.error_banner = Some("Enter a topic you want to learn first.".into());
            cx.notify();
            return;
        }
        if self.busy_message.is_some() {
            return;
        }

        let level = self.new_topic_level;
        let service = self.service.clone();
        self.error_banner = None;
        self.info_banner = None;
        self.busy_message = Some(format!(
            "Asking {} to build your Cognitive Mesh Knowledge Map for \"{}\"...",
            user.runner.label(),
            title
        ));
        self.ensure_anim_loop(cx);
        cx.notify();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let result = cx
                .background_executor()
                .spawn(async move {
                    let topic = service.create_topic_with_map(&user, &title, level)?;
                    let first_q = service.get_or_generate_next_question(&user, &topic.id)?;
                    let map = service.get_knowledge_map(&user.id, &topic.id)?;
                    Ok::<_, anyhow::Error>((topic, first_q, map))
                })
                .await;

            this.update(cx, |view, cx| {
                view.busy_message = None;
                match result {
                    Ok((topic, question, map)) => {
                        view.new_topic_title.clear();
                        view.active_topic_id = Some(topic.id.clone());
                        view.knowledge_map = Some(map);
                        view.current_question = Some(question);
                        view.selected_choice_id = None;
                        view.hint_revealed = false;
                        view.last_answer = None;
                        view.stop_audio_quiet();
                        view.route = Route::Quiz;
                        view.active_field = ActiveField::None;
                        view.refresh_topics();
                        view.trigger_background_prefetch(cx);
                    }
                    Err(err) => {
                        view.error_banner = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn open_topic_quiz(&mut self, topic_id: String, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        if self.busy_message.is_some() {
            return;
        }
        self.active_topic_id = Some(topic_id.clone());
        self.route = Route::Quiz;
        self.active_field = ActiveField::None;
        self.selected_choice_id = None;
        self.hint_revealed = false;
        self.last_answer = None;
        self.media_busy = false;
        self.stop_audio_quiet();
        self.clarify_input.clear();
        self.clarify_busy = false;
        self.clarify_replies.clear();
        self.error_banner = None;
        self.knowledge_map = self.service.get_knowledge_map(&user.id, &topic_id).ok();

        if let Ok(Some(ready_q)) = self.service.peek_buffered_question(&topic_id) {
            self.current_question = Some(ready_q);
            self.refresh_buffer_count();
            self.trigger_background_prefetch(cx);
            cx.notify();
            return;
        }

        self.current_question = None;
        self.buffered_ready = 0;
        self.busy_message = Some(format!(
            "Preparing next tailored question via {}...",
            user.runner.label()
        ));
        self.ensure_anim_loop(cx);
        cx.notify();

        let service = self.service.clone();
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = cx
                .background_executor()
                .spawn(async move { service.get_or_generate_next_question(&user, &topic_id) })
                .await;

            this.update(cx, |view, cx| {
                view.busy_message = None;
                match res {
                    Ok(q) => {
                        view.current_question = Some(q);
                        view.refresh_buffer_count();
                        view.trigger_background_prefetch(cx);
                    }
                    Err(err) => {
                        view.error_banner = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn open_topic_map(&mut self, topic_id: String, cx: &mut Context<Self>) {
        let Some(user) = &self.user else { return };
        self.active_topic_id = Some(topic_id.clone());
        self.knowledge_map = self.service.get_knowledge_map(&user.id, &topic_id).ok();
        self.route = Route::KnowledgeMap;
        self.active_field = ActiveField::None;
        self.error_banner = None;
        cx.notify();
    }

    fn delete_topic(&mut self, topic_id: String, cx: &mut Context<Self>) {
        let Some(user) = &self.user else { return };
        if let Err(err) = self.service.delete_topic(&user.id, &topic_id) {
            self.error_banner = Some(err.to_string());
        } else {
            if self.active_topic_id.as_deref() == Some(&topic_id) {
                self.active_topic_id = None;
                self.knowledge_map = None;
                self.current_question = None;
                self.last_answer = None;
                self.route = Route::Dashboard;
            }
            self.refresh_topics();
        }
        cx.notify();
    }

    fn pick_choice(&mut self, choice_id: &str, cx: &mut Context<Self>) {
        if self.busy_message.is_some() {
            return;
        }
        // Before answering, highlight the picked option; after answering, still allow auditioning any choice!
        if self.last_answer.is_none() {
            self.selected_choice_id = Some(choice_id.to_string());
            self.error_banner = None;
        }

        // If this question has an Audio example, immediately play back what this choice sounds like!
        if let Some(q) = self.current_question.clone() {
            if let Some(ExampleMedia::Audio { spec, .. }) = &q.media {
                if let Some(choice) = q.choices.iter().find(|c| c.id == choice_id) {
                    if let Some(effect) = &choice.audio_effect {
                        let audition_spec = spec.with_choice_effect(effect);
                        let label = format!("Choice [{}]", choice.id);
                        self.play_audio_override(audition_spec, label, cx);
                        return;
                    } else if self.last_answer.is_some() && choice.id == q.correct_choice_id {
                        let label = format!("Choice [{}] (Correct)", choice.id);
                        self.play_audio_demo(spec.clone(), false, label, cx);
                        return;
                    }
                }
            }
        }

        cx.notify();
    }

    fn submit_selected_answer(&mut self, confidence: AnswerConfidence, cx: &mut Context<Self>) {
        if self.last_answer.is_some() || self.busy_message.is_some() {
            return;
        }
        let Some(choice_id) = self.selected_choice_id.clone() else {
            self.error_banner = Some("Pick an option (A–D) first, then submit as Guess or Sure!".into());
            cx.notify();
            return;
        };
        let Some(user) = self.user.clone() else { return };
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        let Some(question) = self.current_question.clone() else { return };

        match self.service.submit_answer(
            &user.id,
            &topic_id,
            &question.id,
            &choice_id,
            confidence,
            self.hint_revealed,
        ) {
            Ok(result) => {
                self.last_answer = Some(result);
                self.error_banner = None;
                self.refresh_topics();
                self.trigger_background_prefetch(cx);
            }
            Err(err) => {
                self.error_banner = Some(err.to_string());
            }
        }
        cx.notify();
    }

    fn stop_audio_quiet(&mut self) {
        stop_global_audio();
        self.audio_status = None;
        self.audio_showing_after = false;
        self.audio_preview_spec = None;
        self.audio_preview_tag = None;
        self.audio_started_at = None;
        self.audio_active_step = None;
        self.audio_wav_path = None;
    }

    fn ensure_audio_playhead_loop(&mut self, cx: &mut Context<Self>) {
        if self.audio_playhead_running || !has_global_audio_running() {
            return;
        }
        self.audio_playhead_running = true;

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(33))
                    .await;

                let mut keep_running = false;
                let updated = this.update(cx, |view, cx| {
                    let exited = global_audio_exited();

                    if exited {
                        if !is_audio_shutting_down() && view.audio_looping {
                            if let Some(wav) = view.audio_wav_path.clone() {
                                if matches!(spawn_global_audio(&wav), Ok(true)) {
                                    view.audio_started_at = Some(Instant::now());
                                    view.audio_active_step = Some(0);
                                    keep_running = true;
                                    cx.notify();
                                    return;
                                }
                            }
                        }
                        stop_global_audio();
                        view.audio_started_at = None;
                        view.audio_active_step = None;
                        view.audio_status = None;
                        view.audio_playhead_running = false;
                        cx.notify();
                        return;
                    }

                    if let Some(started) = view.audio_started_at {
                        let bpm = view.audio_bpm.clamp(70.0, 170.0);
                        let steps = view.audio_steps_per_bar.max(1);
                        let bar_sec = (60.0 / bpm) * 4.0;
                        let step_sec = bar_sec / (steps as f32);
                        let step_idx =
                            ((started.elapsed().as_secs_f32() / step_sec).floor() as usize) % steps;
                        if view.audio_active_step != Some(step_idx) {
                            view.audio_active_step = Some(step_idx);
                            cx.notify();
                        }
                        keep_running = true;
                    } else {
                        view.audio_playhead_running = false;
                    }
                });

                if updated.is_err() || !keep_running {
                    break;
                }
            }
        })
        .detach();
    }

    fn generate_supporting_example(&mut self, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        let Some(question) = self.current_question.clone() else { return };
        if self.media_busy {
            return;
        }
        let already_answered = self.last_answer.is_some();
        let service = self.service.clone();
        self.media_busy = true;
        self.error_banner = None;
        self.ensure_anim_loop(cx);
        cx.notify();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = cx
                .background_executor()
                .spawn(async move {
                    service.generate_question_example(&user, &topic_id, &question.id, already_answered)
                })
                .await;

            this.update(cx, |view, cx| {
                view.media_busy = false;
                match res {
                    Ok(updated_q) => {
                        if view
                            .current_question
                            .as_ref()
                            .map(|q| q.id == updated_q.id)
                            .unwrap_or(false)
                        {
                            view.current_question = Some(updated_q);
                        }
                    }
                    Err(err) => {
                        view.error_banner = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn play_audio_override(&mut self, spec: AudioDemoSpec, label: String, cx: &mut Context<Self>) {
        self.play_audio_internal(spec.clone(), false, Some(spec), Some(label.clone()), label, cx);
    }

    fn play_audio_demo(
        &mut self,
        spec: AudioDemoSpec,
        dry_before: bool,
        label: String,
        cx: &mut Context<Self>,
    ) {
        self.play_audio_internal(spec, dry_before, None, None, label, cx);
    }

    fn play_audio_internal(
        &mut self,
        spec: AudioDemoSpec,
        dry_before: bool,
        preview_override: Option<AudioDemoSpec>,
        preview_tag: Option<String>,
        label: String,
        cx: &mut Context<Self>,
    ) {
        self.stop_audio_quiet();
        let tag = if preview_override.is_some() {
            "choice"
        } else if dry_before {
            "original"
        } else {
            "target"
        };
        let wav_path = std::env::temp_dir().join(format!("learn-demo-{tag}.wav"));
        let bars = if self.audio_looping { 16 } else { 2 };
        let active_levels = if !dry_before && !spec.after_step_levels.is_empty() {
            &spec.after_step_levels
        } else {
            &spec.step_levels
        };
        let steps_per_bar = match active_levels.len() {
            4 => 4,
            8 => 8,
            _ => 16,
        };

        match synthesize_audio_wav_bars(&spec, dry_before, bars, &wav_path) {
            Ok(path) => match spawn_global_audio(&path) {
                Ok(true) => {
                    self.audio_wav_path = Some(path);
                    self.audio_showing_after = !dry_before && preview_override.is_none();
                    self.audio_preview_spec = preview_override;
                    self.audio_preview_tag = preview_tag;
                    self.audio_bpm = spec.bpm.clamp(70.0, 170.0);
                    self.audio_steps_per_bar = steps_per_bar;
                    self.audio_started_at = Some(Instant::now());
                    self.audio_active_step = Some(0);
                    self.audio_status = Some(format!(
                        "{}: {}",
                        if self.audio_looping { "LOOPING" } else { "PLAYING" },
                        label
                    ));
                    self.error_banner = None;
                    self.ensure_audio_playhead_loop(cx);
                }
                Ok(false) => {}
                Err(err) => {
                    self.error_banner = Some(format!("Audio playback unavailable (`afplay`): {err}"));
                }
            },
            Err(err) => {
                self.error_banner = Some(format!("Failed to synthesize audio preview: {err}"));
            }
        }
        cx.notify();
    }

    fn launch_interactive_html(&mut self, question_id: &str, html: &str, cx: &mut Context<Self>) {
        let dir = std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".learn-iocore").join("labs"))
            .unwrap_or_else(|_| std::env::temp_dir().join("learn-labs"));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            self.error_banner = Some(format!("Could not create labs directory: {err}"));
            cx.notify();
            return;
        }
        let path = dir.join(format!("{question_id}.html"));
        if let Err(err) = std::fs::write(&path, html) {
            self.error_banner = Some(format!("Could not write interactive lab: {err}"));
            cx.notify();
            return;
        }
        if let Err(err) = Command::new("open").arg(&path).spawn() {
            self.error_banner = Some(format!("Could not open interactive lab: {err}"));
        } else {
            self.info_banner = Some("Opened interactive lab in your default browser.".into());
        }
        cx.notify();
    }

    fn load_next_question(&mut self, cx: &mut Context<Self>) {
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        self.open_topic_quiz(topic_id, cx);
    }

    fn submit_clarifying_question(&mut self, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        let Some(question) = self.current_question.clone() else { return };
        let prompt_text = self.clarify_input.as_str().trim().to_string();
        if prompt_text.is_empty() || self.clarify_busy {
            return;
        }

        let already_answered = self.last_answer.is_some();
        let service = self.service.clone();
        self.clarify_busy = true;
        self.clarify_input.clear();
        self.active_field = ActiveField::None;
        self.error_banner = None;
        self.ensure_anim_loop(cx);
        cx.notify();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = cx
                .background_executor()
                .spawn(async move {
                    service.ask_clarifying_question(
                        &user,
                        &topic_id,
                        &question.id,
                        &prompt_text,
                        already_answered,
                    )
                })
                .await;

            this.update(cx, |view, cx| {
                view.clarify_busy = false;
                match res {
                    Ok(reply) => {
                        view.clarify_replies.push(reply);
                        view.refresh_topics();
                        // Immediately refill buffer so a follow-up question on the gap can be scheduled
                        view.trigger_background_prefetch(cx);
                    }
                    Err(err) => {
                        view.error_banner = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn trigger_background_prefetch(&mut self, cx: &mut Context<Self>) {
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        let target_open = self.desired_open_queue_size(&topic_id);
        self.start_buffer_refill_for(topic_id, target_open, cx);
    }

    fn start_buffer_refill_for(
        &mut self,
        topic_id: String,
        initial_target_open: usize,
        cx: &mut Context<Self>,
    ) {
        let Some(user) = self.user.clone() else { return };
        if self.prefetching && self.prefetch_topic_id.as_deref() == Some(topic_id.as_str()) {
            return;
        }

        let current_open = self.service.count_open_questions(&topic_id).unwrap_or(0);
        if current_open >= initial_target_open {
            self.refresh_buffer_count();
            return;
        }

        let service = self.service.clone();
        self.prefetching = true;
        self.prefetch_topic_id = Some(topic_id.clone());
        self.ensure_anim_loop(cx);

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut target_open = initial_target_open;

            loop {
                let svc_step = service.clone();
                let user_step = user.clone();
                let tid_step = topic_id.clone();

                let step_result = cx
                    .background_executor()
                    .spawn(async move {
                        svc_step.prefetch_one_into_buffer(&user_step, &tid_step, target_open)
                    })
                    .await;

                let mut keep_going = false;
                let updated = this.update(cx, |view, cx| {
                    view.refresh_buffer_count();

                    let same_user = view
                        .user
                        .as_ref()
                        .map(|u| u.id == user.id && u.runner == user.runner)
                        .unwrap_or(false);
                    let still_relevant = view.active_topic_id.is_none()
                        || view.active_topic_id.as_deref() == Some(topic_id.as_str());

                    if same_user && still_relevant {
                        target_open = view.desired_open_queue_size(&topic_id);
                        let open_now = view.service.count_open_questions(&topic_id).unwrap_or(0);
                        if matches!(step_result, Ok(true)) && open_now < target_open {
                            keep_going = true;
                        }
                    }

                    if !keep_going && view.prefetch_topic_id.as_deref() == Some(topic_id.as_str()) {
                        view.prefetching = false;
                        view.prefetch_topic_id = None;
                    }
                    cx.notify();
                });

                if updated.is_err() || !keep_going {
                    break;
                }
            }
        })
        .detach();
    }

    fn expand_map(&mut self, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        let Some(topic_id) = self.active_topic_id.clone() else { return };
        if self.busy_message.is_some() {
            return;
        }
        let service = self.service.clone();
        self.error_banner = None;
        self.busy_message = Some(format!(
            "Asking {} to broaden missing pillars & deepen your Cognitive Mesh...",
            user.runner.label()
        ));
        self.ensure_anim_loop(cx);
        cx.notify();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = cx
                .background_executor()
                .spawn(async move { service.expand_knowledge_map(&user, &topic_id) })
                .await;

            this.update(cx, |view, cx| {
                view.busy_message = None;
                match res {
                    Ok(map) => {
                        view.knowledge_map = Some(map);
                        view.current_question = None;
                        view.selected_choice_id = None;
                        view.last_answer = None;
                        view.refresh_topics();
                        view.trigger_background_prefetch(cx);
                        view.info_banner = Some(
                            "Broadened your Cognitive Mesh with new pillars & refreshed the question queue.".into(),
                        );
                    }
                    Err(err) => {
                        view.error_banner = Some(err.to_string());
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn select_runner(&mut self, kind: RunnerKind, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        match self.service.update_runner(
            &user.id,
            kind,
            Some(self.gemini_key_input.as_str()),
            Some(self.openai_key_input.as_str()),
        ) {
            Ok(updated) => {
                self.runners = self.service.detect_runners(Some(&updated));
                self.user = Some(updated);
                self.prefetching = false;
                self.prefetch_topic_id = None;
                self.refresh_buffer_count();
                self.trigger_background_prefetch(cx);
                self.error_banner = None;
                self.info_banner = Some(format!("Switched active AI runner to {}.", kind.label()));
            }
            Err(err) => {
                self.error_banner = Some(err.to_string());
            }
        }
        cx.notify();
    }

    fn save_api_keys(&mut self, cx: &mut Context<Self>) {
        let Some(user) = self.user.clone() else { return };
        match self.service.update_runner(
            &user.id,
            user.runner,
            Some(self.gemini_key_input.as_str()),
            Some(self.openai_key_input.as_str()),
        ) {
            Ok(updated) => {
                self.runners = self.service.detect_runners(Some(&updated));
                self.user = Some(updated);
                self.error_banner = None;
                self.info_banner = Some("Saved API keys and refreshed runner availability.".into());
            }
            Err(err) => {
                self.error_banner = Some(err.to_string());
            }
        }
        cx.notify();
    }

    fn rescan_runners(&mut self, cx: &mut Context<Self>) {
        self.runners = self.service.detect_runners(self.user.as_ref());
        self.info_banner = Some("Rescanned local PATH for AI CLIs.".into());
        cx.notify();
    }

    // ---------- Keyboard Input Handling ----------

    fn active_field_mut(&mut self) -> Option<&mut TextBuffer> {
        self.field_buffer_mut(self.active_field)
    }

    fn field_buffer_mut(&mut self, field: ActiveField) -> Option<&mut TextBuffer> {
        match field {
            ActiveField::None => None,
            ActiveField::AuthUsername => Some(&mut self.auth_username),
            ActiveField::AuthPassword => Some(&mut self.auth_password),
            ActiveField::NewTopic => Some(&mut self.new_topic_title),
            ActiveField::QuizClarify => Some(&mut self.clarify_input),
            ActiveField::GeminiKey => Some(&mut self.gemini_key_input),
            ActiveField::OpenaiKey => Some(&mut self.openai_key_input),
        }
    }

    fn is_active_field_masked(&self) -> bool {
        matches!(
            self.active_field,
            ActiveField::AuthPassword | ActiveField::GeminiKey | ActiveField::OpenaiKey
        )
    }

    fn handle_key_down(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let ks = &event.keystroke;
        let cmd = ks.modifiers.platform;
        let ctrl = ks.modifiers.control;
        let alt = ks.modifiers.alt;
        let shift = ks.modifiers.shift;
        let cmd_or_ctrl = cmd || ctrl;

        if cmd_or_ctrl && ks.key.eq_ignore_ascii_case("q") {
            shutdown_global_audio();
            cx.quit();
            return;
        }
        if cmd && ks.key.eq_ignore_ascii_case("w") {
            shutdown_global_audio();
            window.remove_window();
            return;
        }

        // Standard text-field shortcuts when an input is focused
        if self.active_field != ActiveField::None {
            if cmd_or_ctrl && ks.key.eq_ignore_ascii_case("a") {
                if let Some(buf) = self.active_field_mut() {
                    buf.select_all();
                    cx.notify();
                }
                return;
            }

            if cmd_or_ctrl && ks.key.eq_ignore_ascii_case("c") {
                if !self.is_active_field_masked() {
                    if let Some(buf) = self.active_field_mut() {
                        if buf.has_selection() {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                buf.selected_text().to_string(),
                            ));
                        }
                    }
                }
                return;
            }

            if cmd_or_ctrl && ks.key.eq_ignore_ascii_case("x") {
                let masked = self.is_active_field_masked();
                if let Some(buf) = self.active_field_mut() {
                    if buf.has_selection() {
                        if !masked {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                buf.selected_text().to_string(),
                            ));
                        }
                        buf.delete_selection();
                        cx.notify();
                    }
                }
                return;
            }

            if cmd_or_ctrl && ks.key.eq_ignore_ascii_case("v") {
                if let Some(clip) = cx.read_from_clipboard().and_then(|c| c.text()) {
                    let clean = clip.replace(['\r', '\n'], " ");
                    if let Some(buf) = self.active_field_mut() {
                        buf.insert_str(&clean);
                        cx.notify();
                    }
                }
                return;
            }

            // Ctrl+U / Ctrl+W / Ctrl+K unix/macOS line & word deletion shortcuts
            if ctrl && !cmd && ks.key.eq_ignore_ascii_case("w") {
                if let Some(buf) = self.active_field_mut() {
                    buf.delete_word_backward();
                    cx.notify();
                }
                return;
            }
            if ctrl && !cmd && ks.key.eq_ignore_ascii_case("u") {
                if let Some(buf) = self.active_field_mut() {
                    buf.delete_to_start();
                    cx.notify();
                }
                return;
            }
            if ctrl && !cmd && ks.key.eq_ignore_ascii_case("k") {
                if let Some(buf) = self.active_field_mut() {
                    buf.delete_to_end();
                    cx.notify();
                }
                return;
            }

            match ks.key.as_str() {
                "backspace" => {
                    if let Some(buf) = self.active_field_mut() {
                        if cmd {
                            buf.delete_to_start();
                        } else if alt || ctrl {
                            buf.delete_word_backward();
                        } else {
                            buf.backspace();
                        }
                        cx.notify();
                    }
                    return;
                }
                "delete" => {
                    if let Some(buf) = self.active_field_mut() {
                        buf.delete_forward();
                        cx.notify();
                    }
                    return;
                }
                "left" => {
                    if let Some(buf) = self.active_field_mut() {
                        if cmd {
                            buf.move_to_start(shift);
                        } else if alt {
                            buf.move_word_left(shift);
                        } else {
                            buf.move_left(shift);
                        }
                        cx.notify();
                    }
                    return;
                }
                "right" => {
                    if let Some(buf) = self.active_field_mut() {
                        if cmd {
                            buf.move_to_end(shift);
                        } else if alt {
                            buf.move_word_right(shift);
                        } else {
                            buf.move_right(shift);
                        }
                        cx.notify();
                    }
                    return;
                }
                "home" => {
                    if let Some(buf) = self.active_field_mut() {
                        buf.move_to_start(shift);
                        cx.notify();
                    }
                    return;
                }
                "end" => {
                    if let Some(buf) = self.active_field_mut() {
                        buf.move_to_end(shift);
                        cx.notify();
                    }
                    return;
                }
                _ => {}
            }
        }

        if self.user.is_some() && self.route == Route::Quiz && self.active_field == ActiveField::None {
            if self.last_answer.is_none() {
                match ks.key.as_str() {
                    "1" | "a" => {
                        self.pick_choice("A", cx);
                        return;
                    }
                    "2" | "b" => {
                        self.pick_choice("B", cx);
                        return;
                    }
                    "3" | "c" => {
                        self.pick_choice("C", cx);
                        return;
                    }
                    "4" | "d" => {
                        self.pick_choice("D", cx);
                        return;
                    }
                    "h" => {
                        self.hint_revealed = true;
                        cx.notify();
                        return;
                    }
                    "g" => {
                        if self.selected_choice_id.is_some() {
                            self.submit_selected_answer(AnswerConfidence::Guessing, cx);
                        }
                        return;
                    }
                    "enter" => {
                        if self.selected_choice_id.is_some() {
                            self.submit_selected_answer(AnswerConfidence::Confident, cx);
                        }
                        return;
                    }
                    _ => {}
                }
            } else if ks.key == "enter" || ks.key == "space" {
                self.load_next_question(cx);
                return;
            }
        }

        if self.user.is_none() && ks.key == "tab" {
            self.active_field = match self.active_field {
                ActiveField::AuthUsername => ActiveField::AuthPassword,
                _ => ActiveField::AuthUsername,
            };
            if let Some(buf) = self.active_field_mut() {
                buf.select_all();
            }
            cx.notify();
            return;
        }

        if ks.key == "enter" {
            match self.active_field {
                ActiveField::AuthUsername | ActiveField::AuthPassword => self.submit_auth(cx),
                ActiveField::NewTopic => self.create_topic(cx),
                ActiveField::QuizClarify => self.submit_clarifying_question(cx),
                ActiveField::GeminiKey | ActiveField::OpenaiKey => self.save_api_keys(cx),
                ActiveField::None => {}
            }
            return;
        }

        if ks.key == "escape" {
            self.active_field = ActiveField::None;
            cx.notify();
            return;
        }

        if !cmd_or_ctrl && !alt {
            if let Some(ch) = &ks.key_char {
                if !ch.is_empty() && !ch.chars().all(|c| c.is_control()) {
                    if let Some(buf) = self.active_field_mut() {
                        buf.insert_str(ch);
                        cx.notify();
                    }
                }
            } else if ks.key == "space" {
                if let Some(buf) = self.active_field_mut() {
                    buf.insert_str(" ");
                    cx.notify();
                }
            }
        }
    }

    // ---------- Reusable UI Primitives ----------

    fn render_input_box(
        &self,
        id: &'static str,
        label: &str,
        buffer: &TextBuffer,
        placeholder: &str,
        masked: bool,
        field: ActiveField,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let focused = self.active_field == field;
        let is_empty = buffer.is_empty();
        let (before, selected, after) = buffer.display_parts(masked);
        let has_sel = !selected.is_empty();

        let text_content: AnyElement = if is_empty {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap_1()
                .text_xs()
                .when(focused, |el| {
                    el.child(div().text_color(theme::accent_hover()).child("█"))
                })
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .text_color(theme::text_muted())
                        .child(placeholder.to_string()),
                )
                .into_any_element()
        } else if focused {
            let mut row = div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_wrap()
                .items_center()
                .text_xs()
                .text_color(theme::text_primary());

            if !before.is_empty() {
                row = row.child(before);
            }
            if has_sel {
                row = row.child(
                    div()
                        .bg(theme::accent())
                        .text_color(theme::pure_white())
                        .child(selected),
                );
            } else {
                row = row.child(div().text_color(theme::accent_hover()).child("█"));
            }
            if !after.is_empty() {
                row = row.child(after);
            }
            row.into_any_element()
        } else {
            div()
                .flex_1()
                .min_w_0()
                .text_xs()
                .text_color(theme::text_primary())
                .child(format!("{before}{selected}{after}"))
                .into_any_element()
        };

        div()
            .flex()
            .flex_col()
            .gap_1()
            .w_full()
            .min_w_0()
            .child(
                div()
                    .w_full()
                    .text_xs()
                    .text_color(theme::text_muted())
                    .child(format!("// {}", label.to_uppercase())),
            )
            .child(
                div()
                    .id(id)
                    .w_full()
                    .min_w_0()
                    .min_h(px(38.0))
                    .px_3()
                    .py_2()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .overflow_hidden()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(if focused {
                        theme::border_focus()
                    } else {
                        theme::border_subtle()
                    })
                    .cursor_text()
                    .on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
                        let was_focused = this.active_field == field;
                        this.active_field = field;
                        window.focus(&this.focus_handle);
                        if let Some(buf) = this.field_buffer_mut(field) {
                            if event.click_count() >= 2 {
                                buf.select_all();
                            } else if !was_focused {
                                buf.collapse_to_end();
                            } else if buf.has_selection() {
                                buf.collapse_to_end();
                            }
                        }
                        cx.notify();
                    }))
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(if focused {
                                theme::accent_hover()
                            } else {
                                theme::text_muted()
                            })
                            .child(">"),
                    )
                    .child(text_content)
                    .when(!is_empty, |el| {
                        el.child(
                            div()
                                .id((id, 999usize))
                                .flex_shrink_0()
                                .px_1p5()
                                .py_0p5()
                                .rounded_sm()
                                .hover(|s| s.bg(theme::bg_card_hover()))
                                .cursor_pointer()
                                .text_xs()
                                .text_color(theme::text_muted())
                                .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                    this.active_field = field;
                                    window.focus(&this.focus_handle);
                                    if let Some(buf) = this.field_buffer_mut(field) {
                                        buf.clear();
                                    }
                                    cx.notify();
                                }))
                                .child("×"),
                        )
                    }),
            )
    }

    /// 3x3 sci-fi telemetry square matrix with a sweeping crimson + white pulse wave.
    fn render_scifi_pulse_grid(&self) -> AnyElement {
        // Spiral traversal order across the 3x3 matrix for a sleek radar/compute feel
        const ORDER: [[usize; 3]; 3] = [
            [0, 1, 2],
            [7, 8, 3],
            [6, 5, 4],
        ];
        let head = self.anim_tick % 9;
        let mut grid = div()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .gap(px(3.0))
            .p(px(4.0))
            .rounded_xs()
            .bg(theme::bg_input())
            .border_1()
            .border_color(theme::border_subtle());

        for row_idx in 0..3 {
            let mut row = div().flex().gap(px(3.0));
            for col_idx in 0..3 {
                let pos = ORDER[row_idx][col_idx];
                let dist = (head + 9 - pos) % 9;
                let color = match dist {
                    0 => theme::accent_hover(), // Hot laser red head
                    1 => theme::pure_white(),   // Crisp white spark
                    2 => theme::accent(),       // Deep crimson trail
                    3 => theme::text_muted(),   // Fading titanium grey
                    _ => theme::border_subtle(),
                };
                row = row.child(div().size(px(6.0)).rounded(px(1.0)).bg(color));
            }
            grid = grid.child(row);
        }

        grid.into_any_element()
    }

    /// Compact 4-square horizontal scanner for inline status pills (buffer refill / Ask Tutor).
    fn render_scifi_mini_scanner(&self) -> AnyElement {
        const BOUNCE: [usize; 6] = [0, 1, 2, 3, 2, 1];
        let active = BOUNCE[self.anim_tick % BOUNCE.len()];
        let prev = BOUNCE[(self.anim_tick + BOUNCE.len() - 1) % BOUNCE.len()];

        let mut strip = div().flex().items_center().gap(px(3.0)).flex_shrink_0();
        for i in 0..4 {
            let color = if i == active {
                theme::accent_hover()
            } else if i == prev {
                theme::pure_white()
            } else {
                theme::border_subtle()
            };
            strip = strip.child(div().size(px(5.0)).rounded(px(1.0)).bg(color));
        }
        strip.into_any_element()
    }

    fn render_busy_panel(&self, msg: &str, subtitle: &str) -> AnyElement {
        div()
            .w_full()
            .min_w_0()
            .p_5()
            .rounded_xs()
            .bg(theme::bg_card())
            .border_1()
            .border_color(theme::accent())
            .flex()
            .items_center()
            .gap_4()
            .child(self.render_scifi_pulse_grid())
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_primary())
                            .child(msg.to_string()),
                    )
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_xs()
                            .text_color(theme::text_muted())
                            .child(subtitle.to_string()),
                    ),
            )
            .into_any_element()
    }

    fn render_banner(&self) -> Option<AnyElement> {
        if let Some(err) = &self.error_banner {
            return Some(
                div()
                    .w_full()
                    .min_w_0()
                    .px_4()
                    .py_3()
                    .rounded_xs()
                    .bg(theme::rose_soft())
                    .border_1()
                    .border_color(theme::rose())
                    .text_sm()
                    .text_color(theme::rose_text())
                    .child(err.clone())
                    .into_any_element(),
            );
        }
        if let Some(info) = &self.info_banner {
            return Some(
                div()
                    .w_full()
                    .min_w_0()
                    .px_4()
                    .py_3()
                    .rounded_xs()
                    .bg(theme::emerald_soft())
                    .border_1()
                    .border_color(theme::emerald())
                    .text_sm()
                    .text_color(theme::emerald_text())
                    .child(info.clone())
                    .into_any_element(),
            );
        }
        None
    }

    // ---------- Screen 1: Auth ----------

    fn render_auth(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let detected_cli = self.runners.iter().find(|r| r.is_cli && r.available);
        let subtitle = match detected_cli {
            Some(r) => format!(
                "Local AI detected: {} ({})",
                r.label,
                r.version.as_deref().unwrap_or("ready")
            ),
            None => "Local-first Cognitive Mesh · connect Codex CLI, Antigravity CLI, or API keys"
                .to_string(),
        };

        let mut card = div()
            .w(px(450.0))
            .max_w_full()
            .p_7()
            .rounded_xs()
            .bg(theme::bg_card())
            .border_1()
            .border_color(theme::border_subtle())
            .flex()
            .flex_col()
            .gap_4()
            .child(div().w_full().h(px(2.0)).bg(theme::accent()))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .w_full()
                    .min_w_0()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(div().size(px(8.0)).rounded_full().bg(theme::accent()))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::accent_text())
                                    .child("LEARN // COGNITIVE MESH"),
                            ),
                    )
                    .child(
                        div()
                            .text_2xl()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(theme::text_primary())
                            .child("OPERATOR ACCESS"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::text_secondary())
                            .child("Adaptive neural-symbolic training · local-first intelligence"),
                    )
                    .child(
                        div()
                            .mt_1()
                            .text_xs()
                            .text_color(if detected_cli.is_some() {
                                theme::emerald_text()
                            } else {
                                theme::accent_text()
                            })
                            .child(subtitle),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .id("tab-signin")
                            .flex_1()
                            .py_2()
                            .rounded_xs()
                            .flex()
                            .justify_center()
                            .cursor_pointer()
                            .bg(if !self.is_register_mode {
                                theme::accent_soft()
                            } else {
                                theme::bg_input()
                            })
                            .border_1()
                            .border_color(if !self.is_register_mode {
                                theme::accent()
                            } else {
                                theme::border_subtle()
                            })
                            .text_sm()
                            .text_color(if !self.is_register_mode {
                                theme::accent_text()
                            } else {
                                theme::text_secondary()
                            })
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.is_register_mode = false;
                                this.error_banner = None;
                                cx.notify();
                            }))
                            .child("Sign In"),
                    )
                    .child(
                        div()
                            .id("tab-register")
                            .flex_1()
                            .py_2()
                            .rounded_xs()
                            .flex()
                            .justify_center()
                            .cursor_pointer()
                            .bg(if self.is_register_mode {
                                theme::accent_soft()
                            } else {
                                theme::bg_input()
                            })
                            .border_1()
                            .border_color(if self.is_register_mode {
                                theme::accent()
                            } else {
                                theme::border_subtle()
                            })
                            .text_sm()
                            .text_color(if self.is_register_mode {
                                theme::accent_text()
                            } else {
                                theme::text_secondary()
                            })
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.is_register_mode = true;
                                this.error_banner = None;
                                cx.notify();
                            }))
                            .child("Create Account"),
                    ),
            );

        if !self.known_usernames.is_empty() && !self.is_register_mode {
            let mut chips = div().flex().flex_wrap().gap_2();
            for (idx, uname) in self.known_usernames.iter().take(6).enumerate() {
                let u_clone = uname.clone();
                chips = chips.child(
                    div()
                        .id(("known-user", idx))
                        .px_3()
                        .py_1()
                        .rounded_xs()
                        .bg(theme::bg_input())
                        .border_1()
                        .border_color(theme::border_subtle())
                        .hover(|s| s.border_color(theme::accent()))
                        .cursor_pointer()
                        .text_xs()
                        .text_color(theme::text_primary())
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.auth_username.set_text(u_clone.clone());
                            this.active_field = ActiveField::AuthPassword;
                            cx.notify();
                        }))
                        .child(uname.clone()),
                );
            }
            card = card.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::text_muted())
                            .child("Existing local profiles:"),
                    )
                    .child(chips),
            );
        }

        if let Some(banner) = self.render_banner() {
            card = card.child(banner);
        }

        card = card
            .child(self.render_input_box(
                "input-username",
                "Username",
                &self.auth_username,
                "e.g. alex",
                false,
                ActiveField::AuthUsername,
                cx,
            ))
            .child(self.render_input_box(
                "input-password",
                "Password (4+ chars)",
                &self.auth_password,
                "Enter password",
                true,
                ActiveField::AuthPassword,
                cx,
            ))
            .child(
                div()
                    .id("toggle-remember-me")
                    .flex()
                    .items_center()
                    .gap_2()
                    .cursor_pointer()
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.remember_me = !this.remember_me;
                        cx.notify();
                    }))
                    .child(
                        div()
                            .w(px(16.0))
                            .h(px(16.0))
                            .rounded_xs()
                            .bg(if self.remember_me {
                                theme::accent()
                            } else {
                                theme::bg_input()
                            })
                            .border_1()
                            .border_color(if self.remember_me {
                                theme::accent()
                            } else {
                                theme::border_subtle()
                            })
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_xs()
                            .text_color(theme::pure_white())
                            .child(if self.remember_me { "✓" } else { "" }),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::text_secondary())
                            .child("Remember me on this device (auto-login on launch)"),
                    ),
            )
            .child(
                div()
                    .id("btn-auth-submit")
                    .mt_1()
                    .w_full()
                    .h(px(42.0))
                    .rounded_xs()
                    .bg(theme::accent())
                    .hover(|s| s.bg(theme::accent_hover()))
                    .cursor_pointer()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::pure_white())
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.submit_auth(cx);
                    }))
                    .child(if self.is_register_mode {
                        "Create Local Account"
                    } else {
                        "Sign In & Resume Learning"
                    }),
            )
            .child(
                div()
                    .w_full()
                    .text_xs()
                    .text_color(theme::text_muted())
                    .child("Stored locally in ~/.learn-iocore/learn.db · Tab switches fields · Enter submits"),
            );

        div()
            .id("auth-scroll")
            .size_full()
            .overflow_y_scroll()
            .p_6()
            .flex()
            .items_center()
            .justify_center()
            .bg(theme::bg_canvas())
            .child(card)
    }

    // ---------- Main Shell: Sidebar ----------

    fn render_sidebar(&self, user: &User, cx: &mut Context<Self>) -> impl IntoElement {
        let nav_btn = |id: &'static str,
                       label: &'static str,
                       target: Route,
                       active: bool,
                       cx: &mut Context<Self>| {
            div()
                .id(id)
                .w_full()
                .px_3()
                .py_2()
                .rounded_sm()
                .border_l_2()
                .border_color(if active {
                    theme::accent()
                } else {
                    theme::transparent()
                })
                .cursor_pointer()
                .bg(if active {
                    theme::bg_card()
                } else {
                    theme::transparent()
                })
                .hover(|s| s.bg(theme::bg_card_hover()))
                .text_xs()
                .font_weight(if active {
                    gpui::FontWeight::SEMIBOLD
                } else {
                    gpui::FontWeight::NORMAL
                })
                .text_color(if active {
                    theme::text_primary()
                } else {
                    theme::text_secondary()
                })
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                    this.route = target;
                    this.active_field = if target == Route::Dashboard {
                        ActiveField::NewTopic
                    } else {
                        ActiveField::None
                    };
                    this.error_banner = None;
                    cx.notify();
                }))
                .child(label)
        };

        let mut topic_list = div()
            .id("sidebar-topics-scroll")
            .flex()
            .flex_col()
            .gap_1()
            .mt_1()
            .w_full()
            .min_w_0()
            .overflow_y_scroll();

        for (idx, topic) in self.topics.iter().enumerate() {
            let tid = topic.id.clone();
            let is_selected = self.active_topic_id.as_deref() == Some(topic.id.as_str());
            let pct = (topic.average_mastery * 100.0).round() as u32;
            topic_list = topic_list.child(
                div()
                    .id(("sidebar-topic", idx))
                    .w_full()
                    .min_w_0()
                    .px_3()
                    .py_2()
                    .rounded_xs()
                    .cursor_pointer()
                    .bg(if is_selected {
                        theme::bg_card()
                    } else {
                        theme::transparent()
                    })
                    .hover(|s| s.bg(theme::bg_card_hover()))
                    .flex()
                    .items_start()
                    .justify_between()
                    .gap_2()
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.open_topic_quiz(tid.clone(), cx);
                    }))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_xs()
                            .text_color(if is_selected {
                                theme::text_primary()
                            } else {
                                theme::text_secondary()
                            })
                            .child(topic.title.clone()),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_xs()
                            .text_color(theme::emerald_text())
                            .child(format!("{pct}%")),
                    ),
            );
        }

        let mut sidebar = div()
            .w(px(250.0))
            .flex_shrink_0()
            .h_full()
            .overflow_hidden()
            .bg(theme::bg_sidebar())
            .border_r_1()
            .border_color(theme::border_subtle())
            .p_4()
            .flex()
            .flex_col()
            .justify_between()
            .gap_3();

        let mut top = div()
            .flex_1()
            .min_h_0()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .w_full()
                    .min_w_0()
                    .pb_3()
                    .border_b_1()
                    .border_color(theme::border_subtle())
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(theme::text_primary())
                                    .child("LEARN // OS"),
                            )
                            .child(
                                div()
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_xs()
                                    .bg(theme::accent_soft())
                                    .text_xs()
                                    .text_color(theme::accent_text())
                                    .child("LIVE"),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .w_full()
                            .min_w_0()
                            .child(
                                div()
                                    .size(px(6.0))
                                    .flex_shrink_0()
                                    .rounded_full()
                                    .bg(theme::accent()),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_xs()
                                    .text_color(theme::text_secondary())
                                    .child(format!("{} · {}", user.username, user.runner.label())),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .mt_2()
                    .child(nav_btn(
                        "nav-dashboard",
                        "01  TOPICS.INDEX",
                        Route::Dashboard,
                        self.route == Route::Dashboard,
                        cx,
                    ))
                    .when(self.active_topic_id.is_some(), |el| {
                        el.child(nav_btn(
                            "nav-quiz",
                            "02  QUIZ.ARENA",
                            Route::Quiz,
                            self.route == Route::Quiz,
                            cx,
                        ))
                        .child(nav_btn(
                            "nav-map",
                            "03  MESH.GRAPH",
                            Route::KnowledgeMap,
                            self.route == Route::KnowledgeMap,
                            cx,
                        ))
                    })
                    .child(nav_btn(
                        "nav-settings",
                        "04  AI.RUNNERS",
                        Route::Settings,
                        self.route == Route::Settings,
                        cx,
                    )),
            );

        if !self.topics.is_empty() {
            top = top.child(
                div()
                    .mt_3()
                    .flex_1()
                    .min_h_0()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .px_3()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_muted())
                            .child("YOUR TOPICS"),
                    )
                    .child(topic_list),
            );
        }

        sidebar = sidebar.child(top).child(
            div()
                .id("btn-signout")
                .flex_shrink_0()
                .w_full()
                .px_3()
                .py_2()
                .rounded_xs()
                .border_1()
                .border_color(theme::border_subtle())
                .hover(|s| s.bg(theme::bg_card_hover()))
                .cursor_pointer()
                .flex()
                .justify_center()
                .text_xs()
                .text_color(theme::text_secondary())
                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                    this.sign_out(cx);
                }))
                .child("Sign Out"),
        );

        sidebar
    }

    // ---------- Screen 2: Dashboard ----------

    fn render_dashboard(&self, user: &User, cx: &mut Context<Self>) -> impl IntoElement {
        let level_pill = |lvl: SkillLevel, cx: &mut Context<Self>| {
            let active = self.new_topic_level == lvl;
            div()
                .id(("level-pill", lvl as usize))
                .px_3()
                .py_1()
                .rounded_xs()
                .cursor_pointer()
                .bg(if active {
                    theme::accent_soft()
                } else {
                    theme::bg_input()
                })
                .border_1()
                .border_color(if active {
                    theme::accent()
                } else {
                    theme::border_subtle()
                })
                .text_xs()
                .text_color(if active {
                    theme::accent_text()
                } else {
                    theme::text_secondary()
                })
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                    this.new_topic_level = lvl;
                    cx.notify();
                }))
                .child(lvl.label())
        };

        let mut container = div()
            .id("dashboard-scroll")
            .size_full()
            .min_w_0()
            .overflow_y_scroll()
            .p_8()
            .flex()
            .flex_col()
            .gap_6();

        if let Some(banner) = self.render_banner() {
            container = container.child(banner);
        }

        if let Some(msg) = &self.busy_message {
            container = container.child(self.render_busy_panel(
                msg,
                "SYNTHESIZING // Running off-thread so the interface stays responsive.",
            ));
        }

        container = container.child(
            div()
                .w_full()
                .min_w_0()
                .p_6()
                .rounded_xs()
                .bg(theme::bg_card())
                .border_1()
                .border_color(theme::border_subtle())
                .flex()
                .flex_col()
                .gap_4()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .w_full()
                        .min_w_0()
                        .child(
                            div()
                                .text_xl()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(theme::text_primary())
                                .child("What do you want to learn?"),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme::text_secondary())
                                .child(format!(
                                    "Active AI engine: {} — builds a Cognitive Mesh with prerequisite edges, confusion synapses, and live buffered 4-choice questions.",
                                    user.runner.label()
                                )),
                        ),
                )
                .child(self.render_input_box(
                    "input-new-topic",
                    "Topic",
                    &self.new_topic_title,
                    "e.g. Rust async internals, Hypnotic techno production, Options pricing, Organic chemistry...",
                    false,
                    ActiveField::NewTopic,
                    cx,
                ))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .justify_between()
                        .gap_3()
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .items_center()
                                .gap_2()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::text_muted())
                                        .child("Starting level:"),
                                )
                                .child(level_pill(SkillLevel::Beginner, cx))
                                .child(level_pill(SkillLevel::Intermediate, cx))
                                .child(level_pill(SkillLevel::Advanced, cx)),
                        )
                        .child(
                            div()
                                .id("btn-start-topic")
                                .px_5()
                                .py_2()
                                .rounded_xs()
                                .bg(theme::accent())
                                .hover(|s| s.bg(theme::accent_hover()))
                                .cursor_pointer()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::pure_white())
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.create_topic(cx);
                                }))
                                .child("Start Learning"),
                        ),
                ),
        );

        let mut topics_section = div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::text_secondary())
                    .child(format!("Your Learning Tracks ({})", self.topics.len())),
            );

        if self.topics.is_empty() {
            topics_section = topics_section.child(
                div()
                    .w_full()
                    .min_w_0()
                    .p_6()
                    .rounded_xs()
                    .bg(theme::bg_card())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .text_sm()
                    .text_color(theme::text_muted())
                    .child("No topics yet. Enter anything above and press Enter to generate your first Cognitive Mesh Knowledge Map!"),
            );
        } else {
            for (idx, topic) in self.topics.iter().enumerate() {
                let pct = (topic.average_mastery * 100.0).round() as u32;
                let acc = if topic.total_attempts > 0 {
                    ((topic.correct_attempts as f64 / topic.total_attempts as f64) * 100.0).round()
                        as u32
                } else {
                    0
                };
                let tid_quiz = topic.id.clone();
                let tid_map = topic.id.clone();
                let tid_del = topic.id.clone();

                topics_section = topics_section.child(
                    div()
                        .w_full()
                        .min_w_0()
                        .p_5()
                        .rounded_xs()
                        .bg(theme::bg_card())
                        .border_1()
                        .border_color(theme::border_subtle())
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .child(
                                    div()
                                        .flex()
                                        .flex_wrap()
                                        .items_center()
                                        .gap_2()
                                        .min_w_0()
                                        .child(
                                            div()
                                                .text_base()
                                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                                .text_color(theme::text_primary())
                                                .child(topic.title.clone()),
                                        )
                                        .child(
                                            div()
                                                .px_2()
                                                .py_0p5()
                                                .rounded_sm()
                                                .bg(theme::bg_input())
                                                .text_xs()
                                                .text_color(theme::text_secondary())
                                                .child(topic.level.label()),
                                        ),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::emerald_text())
                                        .child(format!(
                                            "{} {pct}% · {}/{} · STRK {}",
                                            theme::ascii_bar(topic.average_mastery, 10),
                                            topic.mastered_count,
                                            topic.concept_count,
                                            topic.current_streak
                                        )),
                                ),
                        )
                        .child(
                            div()
                                .w_full()
                                .h(px(6.0))
                                .rounded_full()
                                .bg(theme::bg_input())
                                .child(
                                    div()
                                        .h_full()
                                        .w(gpui::relative(
                                            (topic.average_mastery as f32).clamp(0.02, 1.0),
                                        ))
                                        .rounded_full()
                                        .bg(theme::emerald()),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::text_muted())
                                        .child(format!(
                                            "{} answered · {}% accuracy · best streak {}",
                                            topic.total_attempts, acc, topic.best_streak
                                        )),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .flex_wrap()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            div()
                                                .id(("topic-continue", idx))
                                                .px_3()
                                                .py_1()
                                                .rounded_xs()
                                                .bg(theme::accent())
                                                .hover(|s| s.bg(theme::accent_hover()))
                                                .cursor_pointer()
                                                .text_xs()
                                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                                .text_color(theme::pure_white())
                                                .on_click(cx.listener(
                                                    move |this, _: &ClickEvent, _, cx| {
                                                        this.open_topic_quiz(tid_quiz.clone(), cx);
                                                    },
                                                ))
                                                .child("Continue Quiz"),
                                        )
                                        .child(
                                            div()
                                                .id(("topic-map", idx))
                                                .px_3()
                                                .py_1()
                                                .rounded_xs()
                                                .bg(theme::bg_input())
                                                .border_1()
                                                .border_color(theme::border_subtle())
                                                .hover(|s| s.border_color(theme::accent()))
                                                .cursor_pointer()
                                                .text_xs()
                                                .text_color(theme::text_primary())
                                                .on_click(cx.listener(
                                                    move |this, _: &ClickEvent, _, cx| {
                                                        this.open_topic_map(tid_map.clone(), cx);
                                                    },
                                                ))
                                                .child("Cognitive Mesh"),
                                        )
                                        .child(
                                            div()
                                                .id(("topic-del", idx))
                                                .px_2()
                                                .py_1()
                                                .rounded_xs()
                                                .hover(|s| s.bg(theme::rose_soft()))
                                                .cursor_pointer()
                                                .text_xs()
                                                .text_color(theme::rose_text())
                                                .on_click(cx.listener(
                                                    move |this, _: &ClickEvent, _, cx| {
                                                        this.delete_topic(tid_del.clone(), cx);
                                                    },
                                                ))
                                                .child("Delete"),
                                        ),
                                ),
                        ),
                );
            }
        }

        container.child(topics_section)
    }

    fn render_example_media(
        &self,
        media: &ExampleMedia,
        question_id: &str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        match media {
            ExampleMedia::Curve {
                title,
                caption,
                values,
                labels,
                ..
            } => {
                let mut bars = div()
                    .w_full()
                    .h(px(76.0))
                    .flex()
                    .items_end()
                    .gap_1p5()
                    .pt_2();
                for (i, val) in values.iter().enumerate() {
                    let v = val.clamp(0.06, 1.0);
                    let lbl = labels.get(i).cloned().unwrap_or_default();
                    bars = bars.child(
                        div()
                            .flex_1()
                            .min_w(px(10.0))
                            .h_full()
                            .flex()
                            .flex_col()
                            .justify_end()
                            .items_center()
                            .gap_1()
                            .child(
                                div()
                                    .w_full()
                                    .h(px(54.0 * v))
                                    .rounded_t_xs()
                                    .bg(if v >= 0.72 {
                                        theme::accent()
                                    } else {
                                        theme::emerald()
                                    }),
                            )
                            .when(!lbl.is_empty(), |el| {
                                el.child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::text_muted())
                                        .child(lbl),
                                )
                            }),
                    );
                }

                div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::sky_text())
                            .child(format!("Visual · {title}")),
                    )
                    .child(bars)
                    .when(!caption.is_empty(), |el| {
                        el.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .text_color(theme::text_secondary())
                                .child(caption.clone()),
                        )
                    })
                    .into_any_element()
            }
            ExampleMedia::Flow {
                title,
                caption,
                steps,
                ..
            } => {
                let mut row = div().w_full().min_w_0().flex().flex_wrap().items_center().gap_2();
                for (idx, step) in steps.iter().enumerate() {
                    if idx > 0 {
                        row = row.child(
                            div()
                                .text_sm()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(theme::accent_text())
                                .child("→"),
                        );
                    }
                    row = row.child(
                        div()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::bg_card())
                            .border_1()
                            .border_color(theme::border_subtle())
                            .flex()
                            .flex_col()
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::text_primary())
                                    .child(step.label.clone()),
                            )
                            .when(!step.note.trim().is_empty(), |el| {
                                el.child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::text_muted())
                                        .child(step.note.clone()),
                                )
                            }),
                    );
                }

                div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::sky_text())
                            .child(format!("Diagram · {title}")),
                    )
                    .child(row)
                    .when(!caption.is_empty(), |el| {
                        el.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .text_color(theme::text_secondary())
                                .child(caption.clone()),
                        )
                    })
                    .into_any_element()
            }
            ExampleMedia::Contrast {
                title,
                caption,
                left_title,
                left_points,
                right_title,
                right_points,
                ..
            } => {
                let make_col = |heading: &str, pts: &[String], accent_fg: gpui::Rgba| {
                    let mut col = div()
                        .flex_1()
                        .min_w(px(220.0))
                        .p_3()
                        .rounded_xs()
                        .bg(theme::bg_card())
                        .border_1()
                        .border_color(theme::border_subtle())
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(accent_fg)
                                .child(heading.to_string()),
                        );
                    for p in pts {
                        col = col.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .text_color(theme::text_secondary())
                                .child(format!("• {p}")),
                        );
                    }
                    col
                };

                div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_2p5()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::sky_text())
                            .child(format!("Contrast · {title}")),
                    )
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .flex()
                            .flex_wrap()
                            .gap_3()
                            .child(make_col(left_title, left_points, theme::amber_text()))
                            .child(make_col(right_title, right_points, theme::emerald_text())),
                    )
                    .when(!caption.is_empty(), |el| {
                        el.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .text_color(theme::text_secondary())
                                .child(caption.clone()),
                        )
                    })
                    .into_any_element()
            }
            ExampleMedia::Audio {
                title,
                caption,
                spec,
                ..
            } => {
                let spec_before = spec.clone();
                let spec_after = spec.clone();
                let before_lbl = spec
                    .before_label
                    .clone()
                    .unwrap_or_else(|| "Hear Original".into());
                let after_lbl = spec
                    .after_label
                    .clone()
                    .unwrap_or_else(|| "Hear Target Result".into());
                let b_status = format!("Original ({before_lbl})");
                let a_status = after_lbl.clone();
                let show_after_btn = self.last_answer.is_some();

                let active_spec = self.audio_preview_spec.as_ref().unwrap_or(spec);
                let fallback_levels: Vec<f32> = if active_spec.step_levels.is_empty() {
                    vec![0.85, 0.35, 0.55, 0.45, 0.85, 0.35, 0.55, 0.45]
                } else {
                    Vec::new()
                };
                let displayed_levels: &[f32] = if let Some(preview) = &self.audio_preview_spec {
                    if !preview.step_levels.is_empty() {
                        &preview.step_levels
                    } else {
                        &fallback_levels
                    }
                } else if self.audio_showing_after && !spec.after_step_levels.is_empty() {
                    &spec.after_step_levels
                } else if !spec.step_levels.is_empty() {
                    &spec.step_levels
                } else {
                    &fallback_levels
                };
                let view_tag = if let Some(tag) = &self.audio_preview_tag {
                    format!("AUDITIONING: {tag}")
                } else if self.audio_showing_after && !spec.after_step_levels.is_empty() {
                    format!("SHOWING: {after_lbl}")
                } else {
                    format!("SHOWING: Original ({before_lbl})")
                };

                let mut card = div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::accent())
                    .flex()
                    .flex_col()
                    .gap_2p5()
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::accent_text())
                                    .child(format!(
                                        "AUDIO + VISUAL // {} ({:.0} BPM · {})",
                                        title, spec.bpm, view_tag
                                    )),
                            )
                            .when_some(self.audio_status.clone(), |el, st| {
                                el.child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::emerald_text())
                                        .child(st),
                                )
                            }),
                    );

                // High-contrast, live-synced drum-machine step meter + realtime playhead LED row
                if !displayed_levels.is_empty() {
                    let max_v = displayed_levels
                        .iter()
                        .copied()
                        .fold(0.0f32, f32::max)
                        .max(0.01);
                    let min_v = displayed_levels
                        .iter()
                        .copied()
                        .fold(1.0f32, f32::min);
                    let has_contrast = (max_v - min_v) >= 0.12;
                    let active_step = self
                        .audio_active_step
                        .map(|s| s % displayed_levels.len().max(1));

                    let mut bars = div()
                        .w_full()
                        .h(px(124.0))
                        .flex()
                        .items_end()
                        .gap_2()
                        .pt_2();
                    for (i, val) in displayed_levels.iter().enumerate() {
                        let raw_v = val.clamp(0.0, 1.0);
                        let scaled_h = (raw_v / max_v).clamp(0.14, 1.0) * 72.0;
                        let is_peak = has_contrast && raw_v >= max_v - 0.05;
                        let is_playing_now = active_step == Some(i);
                        let pct = (raw_v * 100.0).round() as u32;
                        let lbl = spec
                            .step_labels
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("{}", i + 1));

                        let bar_bg = if is_playing_now {
                            theme::accent_hover()
                        } else if is_peak {
                            theme::accent()
                        } else {
                            theme::emerald()
                        };

                        bars = bars.child(
                            div()
                                .flex_1()
                                .max_w(px(68.0))
                                .min_w(px(18.0))
                                .h_full()
                                .px_1()
                                .py_1()
                                .rounded_xs()
                                .bg(if is_playing_now {
                                    theme::accent_soft()
                                } else {
                                    theme::transparent()
                                })
                                .border_1()
                                .border_color(if is_playing_now {
                                    theme::accent_hover()
                                } else {
                                    theme::transparent()
                                })
                                .flex()
                                .flex_col()
                                .justify_end()
                                .items_center()
                                .gap_1()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(if is_playing_now {
                                            theme::pure_white()
                                        } else if is_peak {
                                            theme::accent_text()
                                        } else {
                                            theme::text_muted()
                                        })
                                        .child(format!("{pct}%")),
                                )
                                .child(
                                    div()
                                        .w_full()
                                        .h(px(scaled_h))
                                        .rounded_t_xs()
                                        .bg(bar_bg),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(if is_playing_now || is_peak {
                                            gpui::FontWeight::BOLD
                                        } else {
                                            gpui::FontWeight::NORMAL
                                        })
                                        .text_color(if is_playing_now || is_peak {
                                            theme::text_primary()
                                        } else {
                                            theme::text_muted()
                                        })
                                        .child(lbl),
                                )
                                // Hardware sequencer step LED under each column
                                .child(
                                    div()
                                        .w(px(14.0))
                                        .h(px(3.0))
                                        .rounded_xs()
                                        .bg(if is_playing_now {
                                            theme::accent_hover()
                                        } else {
                                            theme::border_subtle()
                                        }),
                                ),
                        );
                    }
                    card = card.child(bars);
                }

                card.when(!caption.is_empty(), |el| {
                    el.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_xs()
                            .text_color(theme::text_secondary())
                            .child(caption.clone()),
                    )
                })
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .id("btn-audio-before")
                                .px_3()
                                .py_1p5()
                                .rounded_sm()
                                .bg(theme::bg_card())
                                .border_1()
                                .border_color(theme::border_subtle())
                                .hover(|s| s.border_color(theme::accent()))
                                .cursor_pointer()
                                .text_xs()
                                .text_color(theme::text_primary())
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.play_audio_demo(
                                        spec_before.clone(),
                                        true,
                                        b_status.clone(),
                                        cx,
                                    );
                                }))
                                .child(format!("▶ Original ({before_lbl})")),
                        )
                        .when(show_after_btn, |el| {
                            el.child(
                                div()
                                    .id("btn-audio-after")
                                    .px_3()
                                    .py_1p5()
                                    .rounded_sm()
                                    .bg(theme::accent())
                                    .hover(|s| s.bg(theme::accent_hover()))
                                    .cursor_pointer()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::pure_white())
                                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.play_audio_demo(
                                            spec_after.clone(),
                                            false,
                                            a_status.clone(),
                                            cx,
                                        );
                                    }))
                                    .child(format!("▶ {after_lbl}")),
                            )
                        })
                        .child(
                            div()
                                .id("btn-audio-loop-toggle")
                                .px_3()
                                .py_1p5()
                                .rounded_xs()
                                .bg(if self.audio_looping {
                                    theme::accent_soft()
                                } else {
                                    theme::bg_card()
                                })
                                .border_1()
                                .border_color(if self.audio_looping {
                                    theme::accent()
                                } else {
                                    theme::border_subtle()
                                })
                                .cursor_pointer()
                                .text_xs()
                                .text_color(if self.audio_looping {
                                    theme::accent_text()
                                } else {
                                    theme::text_muted()
                                })
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.audio_looping = !this.audio_looping;
                                    cx.notify();
                                }))
                                .child(if self.audio_looping {
                                    "[LOOP: ON]"
                                } else {
                                    "[LOOP: OFF]"
                                }),
                        )
                        .when(has_global_audio_running(), |el| {
                            el.child(
                                div()
                                    .id("btn-audio-stop")
                                    .px_3()
                                    .py_1p5()
                                    .rounded_sm()
                                    .bg(theme::rose_soft())
                                    .border_1()
                                    .border_color(theme::rose())
                                    .cursor_pointer()
                                    .text_xs()
                                    .text_color(theme::rose_text())
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.stop_audio_quiet();
                                        cx.notify();
                                    }))
                                    .child("■ Stop"),
                            )
                        }),
                )
                .into_any_element()
            }
            ExampleMedia::InteractiveHtml {
                title,
                caption,
                html,
                ..
            } => {
                let qid = question_id.to_string();
                let html_src = html.clone();
                div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::accent())
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(220.0))
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::accent_text())
                                    .child(format!("Interactive Lab · {title}")),
                            )
                            .when(!caption.is_empty(), |el| {
                                el.child(
                                    div()
                                        .w_full()
                                        .min_w_0()
                                        .text_xs()
                                        .text_color(theme::text_secondary())
                                        .child(caption.clone()),
                                )
                            }),
                    )
                    .child(
                        div()
                            .id("btn-open-html-lab")
                            .flex_shrink_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::accent())
                            .hover(|s| s.bg(theme::accent_hover()))
                            .cursor_pointer()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::pure_white())
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.launch_interactive_html(&qid, &html_src, cx);
                            }))
                            .child("↗ Launch Interactive Example"),
                    )
                    .into_any_element()
            }
        }
    }

    // ---------- Screen 3: Live Quiz Arena ----------

    fn render_quiz(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut container = div()
            .id("quiz-scroll")
            .size_full()
            .min_w_0()
            .overflow_y_scroll()
            .p_8()
            .flex()
            .flex_col()
            .gap_5();

        if let Some(banner) = self.render_banner() {
            container = container.child(banner);
        }

        let topic_title = self
            .knowledge_map
            .as_ref()
            .map(|m| m.topic.title.clone())
            .unwrap_or_else(|| "Quiz Arena".into());
        let streak = self
            .knowledge_map
            .as_ref()
            .map(|m| m.topic.current_streak)
            .unwrap_or(0);
        let avg_pct = self
            .knowledge_map
            .as_ref()
            .map(|m| (m.topic.average_mastery * 100.0).round() as u32)
            .unwrap_or(0);

        let buffer_full = self.buffered_ready >= DEFAULT_BUFFER_TARGET;
        let buffer_label = if self.prefetching && !buffer_full {
            format!(
                "Refilling buffer: {}/{} ready...",
                self.buffered_ready, DEFAULT_BUFFER_TARGET
            )
        } else {
            format!("Buffer: {}/{} ready", self.buffered_ready, DEFAULT_BUFFER_TARGET)
        };

        container = container.child(
            div()
                .w_full()
                .min_w_0()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap_3()
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xl()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(theme::text_primary())
                                .child(topic_title),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .items_center()
                                .gap_2()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(theme::text_secondary())
                                        .child(format!(
                                            "Overall mastery {avg_pct}% · Current streak {streak}"
                                        )),
                                )
                                .child(
                                    div()
                                        .px_2()
                                        .py_0p5()
                                        .rounded_sm()
                                        .bg(if buffer_full {
                                            theme::emerald_soft()
                                        } else {
                                            theme::accent_soft()
                                        })
                                        .border_1()
                                        .border_color(if buffer_full {
                                            theme::border_subtle()
                                        } else {
                                            theme::accent()
                                        })
                                        .flex()
                                        .items_center()
                                        .gap_2()
                                        .when(self.prefetching && !buffer_full, |el| {
                                            el.child(self.render_scifi_mini_scanner())
                                        })
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(if buffer_full {
                                                    theme::emerald_text()
                                                } else {
                                                    theme::accent_text()
                                                })
                                                .child(buffer_label),
                                        ),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("quiz-view-map")
                        .flex_shrink_0()
                        .px_3()
                        .py_1p5()
                        .rounded_xs()
                        .bg(theme::bg_card())
                        .border_1()
                        .border_color(theme::border_subtle())
                        .hover(|s| s.border_color(theme::accent()))
                        .cursor_pointer()
                        .text_xs()
                        .text_color(theme::text_primary())
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            if let Some(tid) = this.active_topic_id.clone() {
                                this.open_topic_map(tid, cx);
                            }
                        }))
                        .child("View Cognitive Mesh"),
                ),
        );

        if let Some(msg) = &self.busy_message {
            return container.child(self.render_busy_panel(
                msg,
                "PROCESSING // Compiling Cognitive Slice and generating adaptive prompt.",
            ));
        }

        let Some(q) = &self.current_question else {
            return container.child(
                div()
                    .w_full()
                    .min_w_0()
                    .p_6()
                    .rounded_xs()
                    .bg(theme::bg_card())
                    .text_sm()
                    .text_color(theme::text_secondary())
                    .child("Select a topic from the Dashboard to start a live quiz."),
            );
        };

        let (intent_bg, intent_fg) = match q.intent {
            QuestionIntent::ReviewMistake => (theme::amber_soft(), theme::amber_text()),
            QuestionIntent::Reinforce => (theme::sky_soft(), theme::sky_text()),
            QuestionIntent::Introduce => (theme::accent_soft(), theme::accent_text()),
            QuestionIntent::DigDeeper => (theme::emerald_soft(), theme::emerald_text()),
        };

        let node_tuple_label = self
            .knowledge_map
            .as_ref()
            .and_then(|m| m.concepts.iter().find(|c| c.id == q.concept_id))
            .map(|c| {
                format!(
                    "{} · Depth {} · μ {:.2} ± {:.2}",
                    c.title, c.depth, c.mastery, c.uncertainty
                )
            })
            .unwrap_or_else(|| format!("{} · Depth {}", q.concept_title, q.depth));

        let mut q_card = div()
            .w_full()
            .min_w_0()
            .p_6()
            .rounded_xs()
            .bg(theme::bg_card())
            .border_1()
            .border_color(theme::border_subtle())
            .flex()
            .flex_col()
            .gap_5()
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .min_w_0()
                            .child(
                                div()
                                    .px_2p5()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(intent_bg)
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(intent_fg)
                                    .child(q.intent.label()),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(theme::bg_input())
                                    .text_xs()
                                    .text_color(theme::text_secondary())
                                    .child(node_tuple_label),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::text_muted())
                            .child(format!("via {}", q.source.label())),
                    ),
            )
            .when(!q.goal.trim().is_empty(), |el| {
                el.child(
                    div()
                        .w_full()
                        .min_w_0()
                        .px_3p5()
                        .py_2p5()
                        .rounded_xs()
                        .bg(theme::bg_input())
                        .border_l_2()
                        .border_color(theme::accent())
                        .flex()
                        .flex_col()
                        .gap_0p5()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::accent_text())
                                .child("// GOAL & CONTEXT"),
                        )
                        .child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .line_height(px(18.0))
                                .text_color(theme::text_secondary())
                                .child(q.goal.clone()),
                        ),
                )
            })
            .when(!q.primer.trim().is_empty(), |el| {
                el.child(
                    div()
                        .w_full()
                        .min_w_0()
                        .px_3p5()
                        .py_2p5()
                        .rounded_xs()
                        .bg(theme::emerald_soft())
                        .border_l_2()
                        .border_color(theme::emerald())
                        .flex()
                        .flex_col()
                        .gap_0p5()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::emerald_text())
                                .child("// KEY IDEA"),
                        )
                        .child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .line_height(px(19.0))
                                .text_color(theme::text_primary())
                                .child(q.primer.clone()),
                        ),
                )
            })
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .text_lg()
                    .line_height(px(28.0))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::text_primary())
                    .child(q.prompt.clone()),
            );

        // Render AI-chosen supporting example right inside the question card!
        let has_inline_media = q
            .media
            .as_ref()
            .map(|m| m.reveal() == MediaReveal::BeforeAnswer || self.last_answer.is_some())
            .unwrap_or(false);

        if let Some(media) = &q.media {
            if media.reveal() == MediaReveal::BeforeAnswer {
                q_card = q_card.child(self.render_example_media(media, &q.id, cx));
            }
        }

        if self.media_busy {
            q_card = q_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .px_3p5()
                    .py_2p5()
                    .rounded_sm()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::accent())
                    .flex()
                    .items_center()
                    .gap_2p5()
                    .child(self.render_scifi_mini_scanner())
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::accent_text())
                            .child("SYNTHESIZING EXAMPLE // Choosing best visual, audio, or interactive lab..."),
                    ),
            );
        } else if !has_inline_media && self.last_answer.is_none() {
            let has_after_media = q
                .media
                .as_ref()
                .map(|m| m.reveal() == MediaReveal::AfterAnswer)
                .unwrap_or(false);
            q_card = q_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .id("btn-gen-example")
                            .px_3()
                            .py_1p5()
                            .rounded_sm()
                            .bg(theme::bg_input())
                            .border_1()
                            .border_color(theme::border_subtle())
                            .hover(|s| s.border_color(theme::accent()))
                            .cursor_pointer()
                            .text_xs()
                            .text_color(theme::text_primary())
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.generate_supporting_example(cx);
                            }))
                            .child("+ Generate Visual / Audio / Interactive Example"),
                    )
                    .when(has_after_media, |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(theme::text_muted())
                                .child("Comparison example also queued for feedback"),
                        )
                    }),
            );
        }

        // Revealed Hint banner
        if self.hint_revealed && !q.hint.trim().is_empty() && self.last_answer.is_none() {
            q_card = q_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .px_3p5()
                    .py_2p5()
                    .rounded_xs()
                    .bg(theme::amber_soft())
                    .border_1()
                    .border_color(theme::amber())
                    .text_xs()
                    .line_height(px(18.0))
                    .text_color(theme::amber_text())
                    .child(format!("Hint: {}", q.hint)),
            );
        }

        let mut choices_col = div().w_full().min_w_0().flex().flex_col().gap_2p5();
        // Correct/wrong choice badges: crisp high-contrast black text on pure-white correct badge!
        for (idx, choice) in q.choices.iter().enumerate() {
            let cid = choice.id.clone();
            let is_picked = self.selected_choice_id.as_deref() == Some(choice.id.as_str());
            let (bg, border, badge_bg, badge_fg) = if let Some(ans) = &self.last_answer {
                if choice.id == ans.correct_choice_id {
                    (theme::emerald_soft(), theme::emerald(), theme::emerald(), theme::bg_canvas())
                } else if choice.id == ans.selected_choice_id && !ans.is_correct {
                    (theme::rose_soft(), theme::rose(), theme::rose(), theme::pure_white())
                } else {
                    (theme::bg_input(), theme::border_subtle(), theme::bg_card(), theme::text_muted())
                }
            } else if is_picked {
                (theme::accent_soft(), theme::accent(), theme::accent(), theme::pure_white())
            } else {
                (theme::bg_input(), theme::border_subtle(), theme::bg_card(), theme::text_secondary())
            };

            let answered = self.last_answer.is_some();
            let has_audio = matches!(q.media, Some(ExampleMedia::Audio { .. }));
            let has_choice_audio = has_audio && (choice.audio_effect.is_some() || (answered && choice.id == q.correct_choice_id));
            choices_col = choices_col.child(
                div()
                    .id(("choice", idx))
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(bg)
                    .border_1()
                    .border_color(border)
                    .when(!answered || has_choice_audio, |el| {
                        el.cursor_pointer()
                            .hover(|s| s.bg(theme::bg_card_hover()).border_color(theme::accent()))
                    })
                    .flex()
                    .items_start()
                    .gap_3()
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.pick_choice(&cid, cx);
                    }))
                    .child(
                        div()
                            .px_2()
                            .h(px(26.0))
                            .flex_shrink_0()
                            .rounded_xs()
                            .bg(badge_bg)
                            .border_1()
                            .border_color(if is_picked || answered {
                                border
                            } else {
                                theme::border_subtle()
                            })
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(badge_fg)
                            .child(format!("[{}]", choice.id)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_sm()
                            .line_height(px(21.0))
                            .text_color(theme::text_primary())
                            .child(choice.text.clone()),
                    ),
            );
        }

        q_card = q_card.child(choices_col);

        if self.last_answer.is_none() {
            let has_pick = self.selected_choice_id.is_some();
            let has_hint = !q.hint.trim().is_empty() && !self.hint_revealed;

            q_card = q_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .pt_1()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .when(has_hint, |el| {
                                el.child(
                                    div()
                                        .id("btn-show-hint")
                                        .px_3()
                                        .py_1p5()
                                        .rounded_xs()
                                        .bg(theme::bg_input())
                                        .border_1()
                                        .border_color(theme::border_subtle())
                                        .hover(|s| s.border_color(theme::amber()))
                                        .cursor_pointer()
                                        .text_xs()
                                        .text_color(theme::amber_text())
                                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                            this.hint_revealed = true;
                                            cx.notify();
                                        }))
                                        .child("Show Hint (H)"),
                                )
                            })
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(theme::text_muted())
                                    .child(if has_pick {
                                        "Submit as Guess (G) or Sure (Enter)"
                                    } else {
                                        "Pick A–D (or keys 1–4), then submit as Guess or Sure"
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .id("btn-submit-guess")
                                    .px_4()
                                    .py_2()
                                    .rounded_xs()
                                    .bg(if has_pick {
                                        theme::amber_soft()
                                    } else {
                                        theme::bg_input()
                                    })
                                    .border_1()
                                    .border_color(if has_pick {
                                        theme::amber()
                                    } else {
                                        theme::border_subtle()
                                    })
                                    .when(has_pick, |el| el.cursor_pointer())
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(if has_pick {
                                        theme::amber_text()
                                    } else {
                                        theme::text_muted()
                                    })
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        if this.selected_choice_id.is_some() {
                                            this.submit_selected_answer(
                                                AnswerConfidence::Guessing,
                                                cx,
                                            );
                                        }
                                    }))
                                    .child("Submit · Guess (G)"),
                            )
                            .child(
                                div()
                                    .id("btn-submit-sure")
                                    .px_4()
                                    .py_2()
                                    .rounded_xs()
                                    .bg(if has_pick {
                                        theme::accent()
                                    } else {
                                        theme::bg_input()
                                    })
                                    .when(has_pick, |el| {
                                        el.hover(|s| s.bg(theme::accent_hover())).cursor_pointer()
                                    })
                                    .border_1()
                                    .border_color(if has_pick {
                                        theme::accent()
                                    } else {
                                        theme::border_subtle()
                                    })
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(if has_pick {
                                        theme::pure_white()
                                    } else {
                                        theme::text_muted()
                                    })
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        if this.selected_choice_id.is_some() {
                                            this.submit_selected_answer(
                                                AnswerConfidence::Confident,
                                                cx,
                                            );
                                        }
                                    }))
                                    .child("Submit · Sure! (Enter)"),
                            ),
                    ),
            );
        }

        container = container.child(q_card);

        if let Some(ans) = &self.last_answer {
            let (fb_bg, fb_border, title_fg) = if ans.is_correct {
                (theme::emerald_soft(), theme::emerald(), theme::emerald_text())
            } else {
                (theme::rose_soft(), theme::rose(), theme::rose_text())
            };
            let delta_pct = (ans.mastery_delta * 100.0).round() as i32;
            let delta_str = if delta_pct >= 0 {
                format!("+{delta_pct}% μ")
            } else {
                format!("{delta_pct}% μ")
            };

            let mut fb = div()
                .w_full()
                .min_w_0()
                .p_5()
                .rounded_xs()
                .bg(fb_bg)
                .border_1()
                .border_color(fb_border)
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .w_full()
                        .min_w_0()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .justify_between()
                        .gap_2()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_base()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(title_fg)
                                .child(ans.reward_message.clone()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(title_fg)
                                .child(format!(
                                    "{} ({}) · μ {:.2} ± {:.2}",
                                    ans.concept.status.label(),
                                    delta_str,
                                    ans.concept.mastery,
                                    ans.concept.uncertainty
                                )),
                        ),
                )
                .when(!ans.is_correct, |el| {
                    el.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_primary())
                            .child(format!(
                                "Correct answer ({}): {}",
                                ans.correct_choice_id, ans.correct_choice_text
                            )),
                    )
                    .when_some(ans.selected_why_wrong.clone(), |inner, why| {
                        inner.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .px_3()
                                .py_2()
                                .rounded_xs()
                                .bg(theme::bg_input())
                                .border_1()
                                .border_color(theme::rose())
                                .text_xs()
                                .line_height(px(18.0))
                                .text_color(theme::rose_text())
                                .child(format!(
                                    "Why your pick ({}) didn't fit: {}",
                                    ans.selected_choice_id, why
                                )),
                        )
                    })
                })
                .child(
                    div()
                        .w_full()
                        .min_w_0()
                        .text_sm()
                        .line_height(px(21.0))
                        .text_color(theme::text_primary())
                        .child(ans.explanation.clone()),
                );

            // Render AI-chosen media inside feedback if reveal == AfterAnswer
            if let Some(media) = &q.media {
                if media.reveal() == MediaReveal::AfterAnswer {
                    fb = fb.child(self.render_example_media(media, &q.id, cx));
                }
            }

            if !ans.mesh_effects.is_empty() {
                let mut effects_box = div()
                    .w_full()
                    .min_w_0()
                    .p_3()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::accent_text())
                            .child("Cognitive Mesh Updates"),
                    );
                for eff in &ans.mesh_effects {
                    effects_box = effects_box.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_xs()
                            .text_color(theme::text_secondary())
                            .child(format!("• {eff}")),
                    );
                }
                fb = fb.child(effects_box);
            }

            fb = fb.child(
                div()
                    .mt_1()
                    .flex()
                    .justify_end()
                    .child(
                        div()
                            .id("btn-next-question")
                            .px_5()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::accent())
                            .hover(|s| s.bg(theme::accent_hover()))
                            .cursor_pointer()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::pure_white())
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.load_next_question(cx);
                            }))
                            .child("Next Question (Enter)"),
                    ),
            );

            container = container.child(fb);
        }

        // Ask-a-Clarifying-Question bar (always available at the bottom of the Quiz Arena)
        let mut clarify_card = div()
            .w_full()
            .min_w_0()
            .p_5()
            .rounded_xs()
            .bg(theme::bg_card())
            .border_1()
            .border_color(theme::border_subtle())
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_primary())
                            .child("Unsure about a term or concept? Ask the tutor"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::text_muted())
                            .child("Answers without spoiling the quiz—and adds any missing prerequisite to your Cognitive Mesh for future practice."),
                    ),
            );

        for reply in &self.clarify_replies {
            clarify_card = clarify_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .p_3p5()
                    .rounded_xs()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_1p5()
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::accent_text())
                            .child(format!("You asked: {}", reply.question_asked)),
                    )
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_sm()
                            .line_height(px(20.0))
                            .text_color(theme::text_primary())
                            .child(reply.answer.clone()),
                    )
                    .when_some(reply.mesh_note.clone(), |el, note| {
                        el.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .mt_1()
                                .px_2p5()
                                .py_1()
                                .rounded_sm()
                                .bg(theme::emerald_soft())
                                .text_xs()
                                .text_color(theme::emerald_text())
                                .child(format!("+ {note}")),
                        )
                    }),
            );
        }

        if self.clarify_busy {
            clarify_card = clarify_card.child(
                div()
                    .w_full()
                    .min_w_0()
                    .px_3()
                    .py_2()
                    .rounded_sm()
                    .bg(theme::bg_input())
                    .border_1()
                    .border_color(theme::accent())
                    .flex()
                    .items_center()
                    .gap_2p5()
                    .child(self.render_scifi_mini_scanner())
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::accent_text())
                            .child("QUERYING TUTOR // Updating Cognitive Mesh..."),
                    ),
            );
        }

        clarify_card = clarify_card.child(
            div()
                .w_full()
                .min_w_0()
                .flex()
                .items_end()
                .gap_2()
                .child(div().flex_1().min_w_0().child(self.render_input_box(
                    "input-quiz-clarify",
                    "Ask a clarifying question",
                    &self.clarify_input,
                    "e.g. What is a motif? Why does microtiming matter?",
                    false,
                    ActiveField::QuizClarify,
                    cx,
                )))
                .child(
                    div()
                        .id("btn-ask-clarify")
                        .flex_shrink_0()
                        .h(px(40.0))
                        .px_4()
                        .rounded_xs()
                        .bg(theme::accent())
                        .hover(|s| s.bg(theme::accent_hover()))
                        .cursor_pointer()
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::pure_white())
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.submit_clarifying_question(cx);
                        }))
                        .child("Ask Tutor"),
                ),
        );

        container.child(clarify_card)
    }

    // ---------- Screen 4: Cognitive Mesh Knowledge Map ----------

    fn render_knowledge_map(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut container = div()
            .id("map-scroll")
            .size_full()
            .min_w_0()
            .overflow_y_scroll()
            .p_8()
            .flex()
            .flex_col()
            .gap_5();

        if let Some(banner) = self.render_banner() {
            container = container.child(banner);
        }

        if let Some(msg) = &self.busy_message {
            container = container.child(self.render_busy_panel(
                msg,
                "SYNTHESIZING // Running off-thread so the interface stays responsive.",
            ));
        }

        let Some(map) = &self.knowledge_map else {
            return container.child(
                div()
                    .text_sm()
                    .text_color(theme::text_secondary())
                    .child("Open a topic from the Dashboard to inspect its Cognitive Mesh."),
            );
        };

        let stat_card = |label: &'static str, count: usize, color: gpui::Rgba| {
            div()
                .flex_1()
                .min_w(px(105.0))
                .p_3()
                .rounded_xs()
                .bg(theme::bg_card())
                .border_1()
                .border_color(theme::border_subtle())
                .flex()
                .flex_col()
                .gap_0p5()
                .child(div().text_xs().text_color(theme::text_muted()).child(label))
                .child(
                    div()
                        .text_xl()
                        .font_weight(gpui::FontWeight::BOLD)
                        .text_color(color)
                        .child(count.to_string()),
                )
        };

        let req_count = map.edges.iter().filter(|e| e.kind == EdgeKind::Requires).count();
        let conf_count = map.edges.iter().filter(|e| e.kind == EdgeKind::Confuses).count();

        container = container
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .w_full()
                                    .min_w_0()
                                    .text_xl()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(theme::text_primary())
                                    .child(format!("Cognitive Mesh · {}", map.topic.title)),
                            )
                            .child(
                                div()
                                    .w_full()
                                    .min_w_0()
                                    .text_xs()
                                    .text_color(theme::text_secondary())
                                    .child(format!(
                                        "{} concepts · {} prerequisite edges · {} confusion links · {} shadow hypotheses",
                                        map.concepts.len(),
                                        req_count,
                                        conf_count,
                                        map.misconceptions.len()
                                    )),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap_2()
                            .child(
                                div()
                                    .id("btn-expand-map")
                                    .px_3()
                                    .py_2()
                                    .rounded_xs()
                                    .bg(theme::bg_card())
                                    .border_1()
                                    .border_color(theme::border_subtle())
                                    .hover(|s| s.border_color(theme::accent()))
                                    .cursor_pointer()
                                    .text_xs()
                                    .text_color(theme::text_primary())
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.expand_map(cx);
                                    }))
                                    .child("+ Broaden & Deepen Map"),
                            )
                            .child(
                                div()
                                    .id("btn-map-quiz")
                                    .px_4()
                                    .py_2()
                                    .rounded_xs()
                                    .bg(theme::accent())
                                    .hover(|s| s.bg(theme::accent_hover()))
                                    .cursor_pointer()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::pure_white())
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        if let Some(tid) = this.active_topic_id.clone() {
                                            this.open_topic_quiz(tid, cx);
                                        }
                                    }))
                                    .child("Practice Now"),
                            ),
                    ),
            )
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .gap_3()
                    .child(stat_card("Deepened", map.stats.deepened, theme::emerald_text()))
                    .child(stat_card("Mastered", map.stats.mastered, theme::emerald()))
                    .child(stat_card("Learning", map.stats.learning, theme::sky_text()))
                    .child(stat_card("Needs Review", map.stats.needs_review, theme::amber_text()))
                    .child(stat_card("Unseen", map.stats.unseen, theme::text_muted())),
            );

        // Cognitive Mesh Diagnostics Panel (Tension Seams, Shadow Hypotheses, Confuses, Cross-Topic Bridges)
        let has_mesh_signals = !map.cognitive_slice.tension_seams.is_empty()
            || !map.misconceptions.is_empty()
            || !map.cognitive_slice.confusion_pairs.is_empty()
            || !map.cognitive_slice.strong_bridges.is_empty();

        if has_mesh_signals {
            let mut diag = div()
                .w_full()
                .min_w_0()
                .p_5()
                .rounded_xs()
                .bg(theme::bg_card())
                .border_1()
                .border_color(theme::border_subtle())
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::accent_text())
                        .child("Live Cognitive Mesh Signals (fed into ~200-token AI slice)"),
                );

            if !map.cognitive_slice.tension_seams.is_empty() {
                let mut col = div().w_full().min_w_0().flex().flex_col().gap_1p5().child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::amber_text())
                        .child("High-Tension Prerequisite Seams"),
                );
                for seam in &map.cognitive_slice.tension_seams {
                    col = col.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::amber_soft())
                            .text_xs()
                            .text_color(theme::amber_text())
                            .child(format!(
                                "{} · tension {:.0}% ({})",
                                seam.edge,
                                seam.tension * 100.0,
                                seam.note
                            )),
                    );
                }
                diag = diag.child(col);
            }

            if !map.misconceptions.is_empty() {
                let mut col = div().w_full().min_w_0().flex().flex_col().gap_1p5().child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::rose_text())
                        .child("Active Shadow Misconception Hypotheses"),
                );
                for hyp in &map.misconceptions {
                    col = col.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::rose_soft())
                            .text_xs()
                            .text_color(theme::rose_text())
                            .child(format!(
                                "{} · posterior {:.0}% (seen {}x, refuted {}x)",
                                hyp.label,
                                hyp.probability * 100.0,
                                hyp.evidence_count,
                                hyp.refute_count
                            )),
                    );
                }
                diag = diag.child(col);
            }

            if !map.cognitive_slice.confusion_pairs.is_empty() {
                let mut col = div().w_full().min_w_0().flex().flex_col().gap_1p5().child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::amber_text())
                        .child("Inhibitory Confusion Synapses (-w)"),
                );
                for cp in &map.cognitive_slice.confusion_pairs {
                    col = col.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::bg_input())
                            .border_1()
                            .border_color(theme::amber())
                            .text_xs()
                            .text_color(theme::text_primary())
                            .child(format!("{} · interference {:.0}%", cp.pair, cp.weight * 100.0)),
                    );
                }
                diag = diag.child(col);
            }

            if !map.cognitive_slice.strong_bridges.is_empty() {
                let mut col = div().w_full().min_w_0().flex().flex_col().gap_1p5().child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::emerald_text())
                        .child("Cross-Topic Analogy Bridges (+w)"),
                );
                for br in &map.cognitive_slice.strong_bridges {
                    col = col.child(
                        div()
                            .w_full()
                            .min_w_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::emerald_soft())
                            .text_xs()
                            .text_color(theme::emerald_text())
                            .child(format!(
                                "{} (μ {:.2}) → {}",
                                br.from, br.from_mu, br.to
                            )),
                    );
                }
                diag = diag.child(col);
            }

            container = container.child(diag);
        }

        for depth in 1..=4u8 {
            let tier: Vec<&ConceptNode> = map.concepts.iter().filter(|c| c.depth == depth).collect();
            if tier.is_empty() {
                continue;
            }
            let tier_title = match depth {
                1 => "Depth 1 · Foundations",
                2 => "Depth 2 · Core Mechanics",
                3 => "Depth 3 · Applied Synthesis",
                _ => "Depth 4 · Mastery & Edge Cases",
            };

            let mut list = div().w_full().min_w_0().flex().flex_col().gap_2p5();
            for node in tier {
                let (pill_bg, pill_fg, bar_fg) = match node.status {
                    ConceptStatus::Deepened | ConceptStatus::Mastered => {
                        (theme::emerald_soft(), theme::emerald_text(), theme::emerald())
                    }
                    ConceptStatus::NeedsReview => (theme::amber_soft(), theme::amber_text(), theme::amber()),
                    ConceptStatus::Learning => (theme::sky_soft(), theme::sky_text(), theme::accent()),
                    ConceptStatus::Unseen => (theme::bg_input(), theme::text_muted(), theme::border_subtle()),
                };

                let prereqs: Vec<String> = map
                    .edges
                    .iter()
                    .filter(|e| e.kind == EdgeKind::Requires && e.to_slug == node.slug)
                    .map(|e| e.from_label.clone())
                    .collect();

                let mut card = div()
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(theme::bg_card())
                    .border_1()
                    .border_color(theme::border_subtle())
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap_2()
                                    .min_w_0()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .text_color(theme::text_primary())
                                            .child(node.title.clone()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(theme::text_muted())
                                            .child(format!("· {}", node.subtopic)),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(pill_bg)
                                            .text_xs()
                                            .text_color(pill_fg)
                                            .child(node.status.label()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(theme::text_secondary())
                                            .child(format!(
                                                "μ {:.2} ± {:.2} · S {:.1}h ({}/{})",
                                                node.mastery,
                                                node.uncertainty,
                                                node.stability,
                                                node.correct_count,
                                                node.attempts
                                            )),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .text_xs()
                            .line_height(px(18.0))
                            .text_color(theme::text_secondary())
                            .child(node.summary.clone()),
                    )
                    .when(!prereqs.is_empty(), |el| {
                        el.child(
                            div()
                                .w_full()
                                .min_w_0()
                                .text_xs()
                                .text_color(theme::text_muted())
                                .child(format!("Requires: {}", prereqs.join(", "))),
                        )
                    })
                    .child(
                        div()
                            .w_full()
                            .h(px(4.0))
                            .rounded_full()
                            .bg(theme::bg_input())
                            .child(
                                div()
                                    .h_full()
                                    .w(gpui::relative((node.mastery as f32).clamp(0.02, 1.0)))
                                    .rounded_full()
                                    .bg(bar_fg),
                            ),
                    );

                if !node.misconceptions.is_empty() {
                    card = card.child(
                        div()
                            .mt_1()
                            .w_full()
                            .min_w_0()
                            .px_3()
                            .py_2()
                            .rounded_xs()
                            .bg(theme::amber_soft())
                            .text_xs()
                            .text_color(theme::amber_text())
                            .child(format!(
                                "Recent error notes: {}",
                                node.misconceptions.join(" | ")
                            )),
                    );
                }

                list = list.child(card);
            }

            container = container.child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_secondary())
                            .child(tier_title),
                    )
                    .child(list),
            );
        }

        container
    }

    // ---------- Screen 5: AI Runners & Settings ----------

    fn render_settings(&self, user: &User, cx: &mut Context<Self>) -> impl IntoElement {
        let mut container = div()
            .id("settings-scroll")
            .size_full()
            .min_w_0()
            .overflow_y_scroll()
            .p_8()
            .flex()
            .flex_col()
            .gap_5();

        if let Some(banner) = self.render_banner() {
            container = container.child(banner);
        }

        container = container.child(
            div()
                .w_full()
                .min_w_0()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap_3()
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_xl()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(theme::text_primary())
                                .child("AI Runners & Local CLI Connectors"),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme::text_secondary())
                                .child("Use your existing Codex, Antigravity, Gemini, or Claude CLI subscriptions locally—or direct API keys."),
                        ),
                )
                .child(
                    div()
                        .id("btn-rescan")
                        .flex_shrink_0()
                        .px_3()
                        .py_1p5()
                        .rounded_xs()
                        .bg(theme::bg_card())
                        .border_1()
                        .border_color(theme::border_subtle())
                        .hover(|s| s.border_color(theme::accent()))
                        .cursor_pointer()
                        .text_xs()
                        .text_color(theme::text_primary())
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.rescan_runners(cx);
                        }))
                        .child("Rescan Local CLIs"),
                ),
        );

        let mut runner_list = div().w_full().min_w_0().flex().flex_col().gap_2p5();
        for (idx, runner) in self.runners.iter().enumerate() {
            let active = user.runner == runner.kind;
            let kind = runner.kind;
            let status_text = if runner.available {
                match &runner.version {
                    Some(v) => format!("Detected · {v}"),
                    None => "Ready".to_string(),
                }
            } else if runner.is_cli {
                format!(
                    "Not found on PATH (`{}`)",
                    runner.command.as_deref().unwrap_or("")
                )
            } else {
                "Needs API key".to_string()
            };

            runner_list = runner_list.child(
                div()
                    .id(("runner-row", idx))
                    .w_full()
                    .min_w_0()
                    .p_4()
                    .rounded_xs()
                    .bg(if active {
                        theme::accent_soft()
                    } else {
                        theme::bg_card()
                    })
                    .border_1()
                    .border_color(if active {
                        theme::accent()
                    } else {
                        theme::border_subtle()
                    })
                    .hover(|s| s.border_color(theme::accent()))
                    .cursor_pointer()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_4()
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.select_runner(kind, cx);
                    }))
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(260.0))
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .size(px(8.0))
                                            .flex_shrink_0()
                                            .rounded_full()
                                            .bg(if runner.available {
                                                theme::emerald()
                                            } else {
                                                theme::text_muted()
                                            }),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .text_color(theme::text_primary())
                                            .child(runner.label.clone()),
                                    ),
                            )
                            .child(
                                div()
                                    .w_full()
                                    .min_w_0()
                                    .text_xs()
                                    .text_color(theme::text_secondary())
                                    .child(runner.description.clone()),
                            ),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .flex()
                            .flex_col()
                            .items_end()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(if runner.available {
                                        theme::emerald_text()
                                    } else {
                                        theme::text_muted()
                                    })
                                    .child(status_text),
                            )
                            .when(active, |el| {
                                el.child(
                                    div()
                                        .px_2()
                                        .py_0p5()
                                        .rounded_sm()
                                        .bg(theme::accent())
                                        .text_xs()
                                        .text_color(theme::pure_white())
                                        .child("Active"),
                                )
                            }),
                    ),
            );
        }

        container = container.child(runner_list);

        container = container.child(
            div()
                .w_full()
                .min_w_0()
                .p_5()
                .rounded_xs()
                .bg(theme::bg_card())
                .border_1()
                .border_color(theme::border_subtle())
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::text_primary())
                        .child("Optional Direct API Keys (saved locally in SQLite)"),
                )
                .child(self.render_input_box(
                    "input-gemini-key",
                    "Gemini API Key (or set GEMINI_API_KEY env var)",
                    &self.gemini_key_input,
                    "Paste with Cmd+V",
                    true,
                    ActiveField::GeminiKey,
                    cx,
                ))
                .child(self.render_input_box(
                    "input-openai-key",
                    "OpenAI API Key (or set OPENAI_API_KEY env var)",
                    &self.openai_key_input,
                    "Paste with Cmd+V",
                    true,
                    ActiveField::OpenaiKey,
                    cx,
                ))
                .child(
                    div()
                        .flex()
                        .justify_end()
                        .child(
                            div()
                                .id("btn-save-keys")
                                .px_4()
                                .py_2()
                                .rounded_xs()
                                .bg(theme::accent())
                                .hover(|s| s.bg(theme::accent_hover()))
                                .cursor_pointer()
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::pure_white())
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.save_api_keys(cx);
                                }))
                                .child("Save API Keys"),
                        ),
                ),
        );

        container
    }
    fn render_terminal_topbar(&self, user: Option<&User>) -> AnyElement {
        let route_path = match (&self.user, self.route) {
            (None, _) => "~/learn/auth",
            (Some(_), Route::Dashboard) => "~/learn/topics",
            (Some(_), Route::Quiz) => "~/learn/quiz-arena",
            (Some(_), Route::KnowledgeMap) => "~/learn/cognitive-mesh",
            (Some(_), Route::Settings) => "~/learn/runners",
        };
        let right_status = match user {
            Some(u) => format!(
                "OP:{}  |  AI:{}  |  BUF:{}/{}",
                u.username,
                u.runner.as_str(),
                self.buffered_ready,
                DEFAULT_BUFFER_TARGET
            ),
            None => "MODE:LOCAL  |  DB:SQLITE.WAL".to_string(),
        };

        div()
            .w_full()
            .h(px(30.0))
            .flex_shrink_0()
            .px_4()
            .bg(theme::bg_sidebar())
            .border_b_1()
            .border_color(theme::border_subtle())
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .size(px(6.0))
                            .rounded_full()
                            .bg(if self.is_ai_active() {
                                theme::accent_hover()
                            } else {
                                theme::pure_white()
                            }),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::text_primary())
                            .child(route_path),
                    )
                    .when(self.is_ai_active(), |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(theme::accent_text())
                                .child("[BUSY]"),
                        )
                    }),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme::text_muted())
                    .child(right_status),
            )
            .into_any_element()
    }

    fn render_terminal_footer(&self) -> AnyElement {
        let hints = if self.user.is_none() {
            "[TAB] SWITCH FIELD   [CMD+A] SELECT ALL   [CMD+BKSP] CLEAR   [RET] SIGN IN   [CMD+W] EXIT"
        } else {
            match self.route {
                Route::Quiz => {
                    "[1-4 / A-D] PICK   [H] HINT   [G] GUESS   [RET] SURE / NEXT   [CMD+W] EXIT"
                }
                Route::Dashboard => {
                    "[RET] START TOPIC   [CMD+A] SELECT ALL   [CMD+BKSP] CLEAR   [CMD+W] EXIT"
                }
                Route::KnowledgeMap => {
                    "[MESH] μ MASTERY ± σ UNCERTAINTY · S STABILITY   [CMD+W] EXIT"
                }
                Route::Settings => {
                    "[CMD+V] PASTE KEY   [RET] SAVE KEYS   [CMD+W] EXIT"
                }
            }
        };

        div()
            .w_full()
            .h(px(26.0))
            .flex_shrink_0()
            .px_4()
            .bg(theme::bg_sidebar())
            .border_t_1()
            .border_color(theme::border_subtle())
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .child(
                div()
                    .text_xs()
                    .text_color(theme::text_muted())
                    .child(hints),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme::accent_text())
                    .child("LEARN // TUI-NATIVE"),
            )
            .into_any_element()
    }
}

impl Render for LearnApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let user_opt = self.user.clone();
        let content: AnyElement = if let Some(user) = user_opt.clone() {
            let main_view: AnyElement = match self.route {
                Route::Dashboard => self.render_dashboard(&user, cx).into_any_element(),
                Route::Quiz => self.render_quiz(cx).into_any_element(),
                Route::KnowledgeMap => self.render_knowledge_map(cx).into_any_element(),
                Route::Settings => self.render_settings(&user, cx).into_any_element(),
            };

            div()
                .flex_1()
                .min_h_0()
                .w_full()
                .overflow_hidden()
                .flex()
                .bg(theme::bg_canvas())
                .child(self.render_sidebar(&user, cx))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .h_full()
                        .overflow_hidden()
                        .child(main_view),
                )
                .into_any_element()
        } else {
            div()
                .flex_1()
                .min_h_0()
                .w_full()
                .overflow_hidden()
                .child(self.render_auth(cx))
                .into_any_element()
        };

        div()
            .id("learn-root")
            .font_family(theme::mono_font())
            .track_focus(&self.focus_handle)
            .on_action(|_: &CloseWindow, window, _| {
                shutdown_global_audio();
                window.remove_window();
            })
            .on_action(|_: &QuitApp, _, cx| {
                shutdown_global_audio();
                cx.quit();
            })
            .size_full()
            .flex()
            .flex_col()
            .overflow_hidden()
            .bg(theme::bg_canvas())
            .text_color(theme::text_primary())
            .on_key_down(cx.listener(Self::handle_key_down))
            .child(self.render_terminal_topbar(user_opt.as_ref()))
            .child(content)
            .child(self.render_terminal_footer())
    }
}

#[allow(dead_code)]
fn _shared(_s: SharedString) {}
