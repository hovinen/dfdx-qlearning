pub(super) fn play() {
    let mut options = eframe::NativeOptions::default();
    options.viewport = egui::ViewportBuilder::default()
        .with_fullscreen(true)
        .with_app_id("tictactoe")
        .with_title("Tic-Tac-Toe");
    eframe::run_native("Tic-Tac-Toe", options, Box::new(|_| Box::new(App::new()))).unwrap();
}

struct App {}

impl App {
    fn new() -> Self {
        App {}
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        todo!()
    }
}
