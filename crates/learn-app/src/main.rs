mod app;
mod theme;

use gpui::{
    App, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, prelude::*, px, size,
};
use learn_core::LearnService;

use crate::app::LearnApp;

fn main() {
    let service = LearnService::open_default().expect("failed to open local Learn SQLite database");

    Application::new().run(move |cx: &mut App| {
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
}
