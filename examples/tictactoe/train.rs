use crate::{CellState, TicTacToeNetwork, TicTacToeState};
use dfdx_qlearning::{
    abstract_model::AbstractModel,
    actor::{NaiveActor, TrainableActor},
    game_logger::DisplayGameLogger,
    {actor::Engine, game_logger::TrivialGameLogger},
};
use plotly::{
    common::{AxisSide, Mode, Title},
    layout::Axis,
    Layout, Plot, Scatter,
};

const TRAIN_STEPS: usize = 100;
const CAPACITY: usize = 10000;
const FUTURE_DISCOUNT: f32 = 1.0;
const EPSILON: f32 = 1.0;
const STEPS: usize = 60;
const STEP_GAME_COUNT: usize = 200;
const TEST_GAME_COUNT: usize = 100;

pub(super) fn train() {
    let mut current_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
        CellState::O,
        AbstractModel::new(TRAIN_STEPS, FUTURE_DISCOUNT, EPSILON, CAPACITY),
    );
    let mut previous_actor = current_actor
        .clone()
        .make_untrainable()
        .switch_player(CellState::X);
    let mut naive_actor = NaiveActor::new(CellState::X);
    let engine = Engine::new(TrivialGameLogger);

    let mut x_wins = 0;
    let mut o_wins = 0;
    let mut draws = 0;
    for _ in 0..TEST_GAME_COUNT {
        match engine.play_once(&[
            (&CellState::X, &naive_actor),
            (&CellState::O, &current_actor),
        ]) {
            Some(CellState::X) => x_wins += 1,
            Some(CellState::O) => o_wins += 1,
            None | Some(CellState::Empty) => draws += 1,
        }
    }
    let mut stats = vec![Stats {
        game_count: 0,
        x_win_proportion: x_wins as f64 / TEST_GAME_COUNT as f64,
        o_win_proportion: o_wins as f64 / TEST_GAME_COUNT as f64,
        draw_proportion: draws as f64 / TEST_GAME_COUNT as f64,
        epsilon: EPSILON,
        tau: current_actor.1.tau(),
    }];

    'training_loop: for step in 0..STEPS {
        let existing_actor = current_actor
            .clone()
            .make_untrainable()
            .switch_player(CellState::X);
        println!(
            "[train_steps={TRAIN_STEPS}, capacity={CAPACITY}, future_discount={FUTURE_DISCOUNT}, epsilon={:.3}] Training games {:4} - {:4}",
            current_actor.1.epsilon(),
            step * STEP_GAME_COUNT,
            (step + 1) * STEP_GAME_COUNT - 1
        );
        engine.train_players(
            &mut [
                (&CellState::X, &mut previous_actor),
                (&CellState::O, &mut current_actor),
            ],
            STEP_GAME_COUNT,
        );
        engine.train_players(
            &mut [
                (&CellState::X, &mut naive_actor),
                (&CellState::O, &mut current_actor),
            ],
            STEP_GAME_COUNT,
        );
        previous_actor = existing_actor;

        for _ in 0..1 {
            x_wins = 0;
            o_wins = 0;
            draws = 0;

            for _ in 0..TEST_GAME_COUNT {
                match engine.play_once(&[
                    (&CellState::X, &naive_actor),
                    (&CellState::O, &current_actor),
                ]) {
                    Some(CellState::X) => x_wins += 1,
                    Some(CellState::O) => o_wins += 1,
                    None | Some(CellState::Empty) => draws += 1,
                }
            }

            stats.push(Stats {
                game_count: (step + 1) * STEP_GAME_COUNT,
                x_win_proportion: x_wins as f64 / TEST_GAME_COUNT as f64,
                o_win_proportion: o_wins as f64 / TEST_GAME_COUNT as f64,
                draw_proportion: draws as f64 / TEST_GAME_COUNT as f64,
                epsilon: current_actor.1.epsilon(),
                tau: current_actor.1.tau(),
            });

            if o_wins as f32 / TEST_GAME_COUNT as f32 > 0.95 {
                println!("Model seems good enough, ending training");
                break 'training_loop;
            }
        }
    }

    println!("Sample game against naive actor:");
    Engine::new(DisplayGameLogger).play_once(&[
        (&CellState::X, &naive_actor),
        (&CellState::O, &current_actor),
    ]);
    println!("Sample game against previous trained actor:");
    Engine::new(DisplayGameLogger).play_once(&[
        (&CellState::X, &previous_actor),
        (&CellState::O, &current_actor),
    ]);

    write_stats(&stats);

    current_actor.1.save("models/tictactoe.npz").unwrap();
    println!("Saved model to models/tictactoe.npz");
}

struct Stats {
    game_count: usize,
    x_win_proportion: f64,
    o_win_proportion: f64,
    draw_proportion: f64,
    epsilon: f32,
    tau: f32,
}

fn write_stats(stats: &[Stats]) {
    let mut plot = Plot::new();

    let trace_x = Scatter::new(
        stats
            .iter()
            .map(|Stats { game_count, .. }| *game_count)
            .collect::<Vec<_>>(),
        stats
            .iter()
            .map(
                |Stats {
                     x_win_proportion, ..
                 }| *x_win_proportion,
            )
            .collect::<Vec<_>>(),
    )
    .name("X wins")
    .mode(Mode::Lines);
    plot.add_trace(trace_x);

    let trace_o = Scatter::new(
        stats
            .iter()
            .map(|Stats { game_count, .. }| *game_count)
            .collect::<Vec<_>>(),
        stats
            .iter()
            .map(
                |Stats {
                     o_win_proportion, ..
                 }| *o_win_proportion,
            )
            .collect::<Vec<_>>(),
    )
    .name("O wins")
    .mode(Mode::Lines);
    plot.add_trace(trace_o);

    let trace_d = Scatter::new(
        stats
            .iter()
            .map(|Stats { game_count, .. }| *game_count)
            .collect::<Vec<_>>(),
        stats
            .iter()
            .map(
                |Stats {
                     draw_proportion, ..
                 }| *draw_proportion,
            )
            .collect::<Vec<_>>(),
    )
    .name("draws")
    .mode(Mode::Lines);
    plot.add_trace(trace_d);

    let trace_epsilon = Scatter::new(
        stats
            .iter()
            .map(|Stats { game_count, .. }| *game_count)
            .collect::<Vec<_>>(),
        stats
            .iter()
            .map(|Stats { epsilon, .. }| *epsilon as f64)
            .collect::<Vec<_>>(),
    )
    .name("epsilon")
    .mode(Mode::Lines)
    .y_axis("y2");
    plot.add_trace(trace_epsilon);

    let trace_tau = Scatter::new(
        stats
            .iter()
            .map(|Stats { game_count, .. }| *game_count)
            .collect::<Vec<_>>(),
        stats
            .iter()
            .map(|Stats { tau, .. }| *tau as f64)
            .collect::<Vec<_>>(),
    )
    .name("tau")
    .mode(Mode::Lines)
    .y_axis("y2");
    plot.add_trace(trace_tau);

    let layout = Layout::new()
        .y_axis(Axis::new().title(Title::new("Proportion")))
        .y_axis2(
            Axis::new()
                .title(Title::new("Epsilon / Tau"))
                .overlaying("y")
                .side(AxisSide::Right),
        );
    plot.set_layout(layout);

    const DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/stats/tictactoe");
    std::fs::create_dir_all(DIRECTORY).unwrap();
    println!("Writing graph of results to {DIRECTORY}/stats.html");
    plot.write_html(format!("{DIRECTORY}/stats.html"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TicTacToeAction, TicTacToeNetwork, TicTacToeState};
    use dfdx_qlearning::actor::{Actor, ActorState};
    use googletest::prelude::*;
    use rstest::rstest;

    #[test]
    fn trained_x_model_wins_over_untrained_o_model() -> Result<()> {
        let x_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
            CellState::X,
            AbstractModel::load("models/tictactoe.npz", 10, 0.9, 0.3, 10000)?,
        );
        let naive_actor = NaiveActor::new(CellState::O);

        verify_that!(
            win_count_after_n_games(
                100,
                &[(&CellState::X, &x_actor), (&CellState::O, &naive_actor)],
                CellState::X
            ),
            ge(95)
        )
    }

    #[test]
    fn trained_o_model_wins_over_untrained_x_model() -> Result<()> {
        let o_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
            CellState::O,
            AbstractModel::load("models/tictactoe.npz", 10, 0.9, 0.3, 10000)?,
        );
        let naive_actor = NaiveActor::new(CellState::X);

        verify_that!(
            win_count_after_n_games(
                100,
                &[(&CellState::X, &naive_actor), (&CellState::O, &o_actor)],
                CellState::O
            ),
            ge(80)
        )
    }

    #[rstest]
    #[case([['X', 'X', ' '], ['O', ' ', ' '], [' ', 'O', ' ']], CellState::X, 0, 2)]
    #[case([['O', ' ', ' '], ['X', 'X', ' '], [' ', 'O', ' ']], CellState::X, 1, 2)]
    #[case([['O', ' ', ' '], [' ', 'O', ' '], ['X', 'X', ' ']], CellState::X, 2, 2)]
    #[case([['X', 'X', ' '], ['O', ' ', ' '], [' ', 'O', ' ']], CellState::O, 0, 2)]
    #[case([['O', ' ', ' '], ['X', 'X', ' '], [' ', 'O', ' ']], CellState::O, 1, 2)]
    #[case([['O', ' ', ' '], [' ', 'O', ' '], ['X', 'X', ' ']], CellState::O, 2, 2)]
    #[case([['X', ' ', ' '], ['O', 'X', 'O'], [' ', ' ', ' ']], CellState::X, 2, 2)]
    #[case([['O', 'X', ' '], ['X', 'O', 'X'], [' ', 'X', ' ']], CellState::O, 2, 2)]
    #[case([['O', 'O', ' '], [' ', ' ', 'X'], [' ', 'X', ' ']], CellState::O, 0, 2)]
    #[case([['X', 'X', ' '], ['O', ' ', ' '], [' ', 'O', ' ']], CellState::O, 0, 2)]
    #[case([['O', 'O', ' '], [' ', ' ', 'X'], [' ', 'X', ' ']], CellState::X, 0, 2)]
    fn winning_or_blocking_move_is_chosen(
        #[case] board_state: [[char; 3]; 3],
        #[case] player: CellState,
        #[case] chosen_row: u8,
        #[case] chosen_col: u8,
    ) -> Result<()> {
        let model: AbstractModel<TicTacToeState, _, _, TicTacToeNetwork, 9, 9> =
            AbstractModel::load("models/tictactoe.npz", 10, 0.9, 0.3, 10000)?;
        let state = TicTacToeState(board_state.map(|row| row.map(|cell| cell.into())));
        let candidates = state.available_actions();

        let value = model.choose_from_model_only(&state, player.clone(), &candidates);

        dbg!(model.evaluate(&state, player));

        verify_that!(
            value,
            matches_pattern!(TicTacToeAction(eq(chosen_row), eq(chosen_col)))
        )
    }

    fn win_count_after_n_games(
        game_count: u32,
        actors: &[(&CellState, &dyn Actor<TicTacToeState, TicTacToeAction>)],
        count_player: CellState,
    ) -> u32 {
        let mut wins = 0;
        let engine = Engine::new(TrivialGameLogger);
        for _ in 0..game_count {
            match engine.play_once(actors) {
                Some(c) if c == count_player => {
                    wins += 1;
                }
                _ => {}
            }
        }
        wins
    }
}
