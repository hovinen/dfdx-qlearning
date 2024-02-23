use crate::{CellState, TicTacToeAction, TicTacToeNetwork, TicTacToeState};
use dfdx_qlearning::{
    abstract_model::AbstractModel,
    actor::{Actor, TrainableActor},
};

pub(super) fn play() {
    let mut options = eframe::NativeOptions::default();
    options.viewport = egui::ViewportBuilder::default()
        .with_fullscreen(true)
        .with_app_id("tictactoe")
        .with_title("Tic-Tac-Toe");
    eframe::run_native("Tic-Tac-Toe", options, Box::new(|_| Box::new(App::new()))).unwrap();
}

struct App {
    state: TicTacToeState,
    actor: TrainableActor<TicTacToeState, TicTacToeAction, CellState, TicTacToeNetwork, 9, 9>,
    outcome: Option<CellState>,
}

impl App {
    fn new() -> Self {
        let state = TicTacToeState::default();
        let actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
            CellState::O,
            AbstractModel::load("models/tictactoe.npz", 10, 0.9, 0.3, 10000).unwrap(),
        );
        App {
            state,
            actor,
            outcome: None,
        }
    }

    fn set_piece(&mut self, i: usize, j: usize) {
        if !matches!(self.state.0[i][j], CellState::Empty) {
            return;
        }

        self.state.0[i][j] = CellState::X;
        if self.state.player_won(CellState::X) {
            self.outcome = Some(CellState::X);
            return;
        }
        self.actor.play_step(&mut self.state);
        self.outcome = if self.state.player_won(CellState::O) {
            Some(CellState::O)
        } else if self.state.draw() {
            Some(CellState::Empty)
        } else {
            None
        };
    }

    fn show_game_over_window(ctx: &egui::Context, text: &'static str) {
        egui::Window::new("Game over")
            .fixed_rect(egui::Rect::from_center_size(
                ctx.available_rect().center(),
                egui::vec2(200.0, 75.0),
            ))
            .show(ctx, |ui| {
                ui.heading(text);
                ui.add_space(14.0);
                if ui.button("Close").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let enabled = if let Some(outcome) = self.outcome.as_ref() {
                match outcome {
                    CellState::X => {
                        Self::show_game_over_window(ctx, "You won!");
                    }
                    CellState::O => {
                        Self::show_game_over_window(ctx, "You lost!");
                    }
                    CellState::Empty => {
                        Self::show_game_over_window(ctx, "The game was a draw.");
                    }
                }
                false
            } else {
                true
            };
            ui.add_enabled_ui(enabled, |ui| {
                let cell_size = ui.clip_rect().width().min(ui.clip_rect().height()) / 3.0;
                let cell_x_start = if ui.clip_rect().width() < ui.clip_rect().height() {
                    cell_size / 2.0
                } else {
                    ui.clip_rect().center().x - cell_size * 3.0 / 2.0 + cell_size / 2.0
                };
                let cell_y_start = if ui.clip_rect().width() < ui.clip_rect().height() {
                    ui.clip_rect().center().y - cell_size * 3.0 / 2.0 + cell_size / 2.0
                } else {
                    cell_size / 2.0
                };
                for i in 0..3 {
                    let mut cell_centre =
                        egui::pos2(cell_x_start, cell_y_start + i as f32 * cell_size);
                    for j in 0..3 {
                        let rect = egui::Rect::from_center_size(
                            cell_centre,
                            egui::vec2(cell_size, cell_size),
                        );
                        let response = ui.allocate_rect(rect, egui::Sense::click());
                        if response.hovered() {
                            ui.painter()
                                .rect_filled(rect, 20.0, egui::Color32::DARK_GRAY);
                        }
                        if response.clicked() {
                            self.set_piece(i, j);
                        }
                        let text = match self.state.0[i][j] {
                            CellState::X => "X",
                            CellState::O => "O",
                            CellState::Empty => " ",
                        };
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            text,
                            egui::FontId::monospace(cell_size - 20.0),
                            egui::Color32::WHITE,
                        );
                        cell_centre.x += cell_size;
                    }
                }
            });
        });
    }
}
