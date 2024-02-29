use crate::{CellState, TicTacToeAction, TicTacToeNetwork, TicTacToeState};
use dfdx_qlearning::{
    abstract_model::{AbstractModel, EncodableAction},
    actor::{Actor, Step, TrainableActor},
};
use std::collections::VecDeque;

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
    actor: TrainableActor<TicTacToeState, TicTacToeAction, CellState, TicTacToeNetwork, 9, 18>,
    outcome: Option<CellState>,
    steps: VecDeque<Step<TicTacToeState, TicTacToeAction>>,
}

impl App {
    fn new() -> Self {
        let state = TicTacToeState::default();
        let actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 18>(
            CellState::O,
            AbstractModel::load("models/tictactoe.npz", 100, 0.7, 0.0, 10000).unwrap(),
        );
        App {
            state,
            actor,
            outcome: None,
            steps: VecDeque::new(),
        }
    }

    fn set_piece(&mut self, i: usize, j: usize) {
        if !matches!(self.state.0[i][j], CellState::Empty) {
            return;
        }

        self.state.0[i][j] = CellState::X;
        if let Some(last_step) = self.steps.front_mut() {
            last_step.new_state = self.state.clone();
        }
        if self.state.player_won(CellState::X) {
            self.outcome = Some(CellState::X);
            return;
        }
        if let Some(step) = self.actor.play_step_to_train(&mut self.state) {
            self.steps.push_front(step);
        }
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

    fn resolve(&mut self) {
        let mut state_action_scores = VecDeque::new();
        for step in self.steps.drain(..) {
            let scores = self.actor.evaluate(&step.old_state);
            state_action_scores.push_front((step.old_state.clone(), step.action.clone(), scores));
            let reward = step.new_state.o_reward();
            self.actor.record(step, reward);
        }
        self.actor.train();
        for (state, action, scores) in state_action_scores {
            let new_scores = self.actor.evaluate_training(&state);
            println!("\n\nState:\n{state}");
            println!("Chosen action: {action:?}");
            println!("Scores:");
            for (i, (old_score, new_score)) in
                scores.into_iter().zip(new_scores.into_iter()).enumerate()
            {
                let action = TicTacToeAction::decode(i, CellState::O);
                println!("{action:?}: old={old_score:>10.4}, new={new_score:>10.4}");
            }
        }
        self.actor.1.save("models/tictactoe.npz").unwrap();
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
                self.resolve();
                false
            } else {
                true
            };
            ui.add_enabled_ui(enabled, |ui| {
                let (ui_size, cell_x_start, cell_y_start) =
                    if ui.clip_rect().width() < ui.clip_rect().height() {
                        (
                            ui.clip_rect().width(),
                            0.0,
                            ui.clip_rect().center().y - ui.clip_rect().width() / 3.0,
                        )
                    } else {
                        let ui_size = ui.clip_rect().height() * 0.75;
                        (
                            ui_size,
                            ui.clip_rect().center().x - ui_size / 3.0,
                            ui.clip_rect().center().y - ui_size / 3.0,
                        )
                    };
                let cell_size = ui_size / 3.0;
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
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
