mod app;
mod theme;

use gpui::{
    App, Application, Bounds, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions, actions,
    prelude::*, px, size,
};
use learn_core::LearnService;

use crate::app::LearnApp;

actions!(learn_app, [CloseWindow, QuitApp]);

fn main() {
    let service = LearnService::open_default().expect("failed to open local Learn SQLite database");

    Application::new().run(move |cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("cmd-w", CloseWindow, None),
            KeyBinding::new("cmd-q", QuitApp, None),
        ]);

        cx.on_action(|_: &QuitApp, cx| {
            cx.quit();
        });

        // On macOS, closing the last window via the red traffic light or Cmd+W must explicitly quit.
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds::centered(None, size(px(1160.0), px(780.0)), cx);
        let svc = service.clone();

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("Learn · Adaptive AI Knowledge System".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| LearnApp::new(svc, window, cx)),
        )
        .expect("failed to open Learn window");

        cx.activate(true);
    });

    // Ensure no detached background CLI prefetch thread keeps the process alive after UI exit.
    std::process::exit(0);
}
