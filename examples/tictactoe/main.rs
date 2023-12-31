use dfdx::{
    nn::{builders::Linear, modules::Dropout},
    prelude::ReLU,
    shapes::Rank1,
    tensor::{Tensor, TensorFrom},
    tensor_ops::Device,
};
use dfdx_qlearning::{
    abstract_model::AbstractModel,
    actor::{NaiveActor, TrainableActor},
    game_logger::DisplayGameLogger,
    {
        abstract_model::{EncodableAction, EncodableState, Reward},
        actor::{ActorState, Engine, EnumerablePlayer},
        game_logger::TrivialGameLogger,
    },
};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

const TRAIN_STEPS: usize = 100;
const CAPACITY: usize = 10000;
const FUTURE_DISCOUNT: f32 = 1.0;
const EPSILON: f32 = 1.0;
const STEPS: usize = 30;
const STEP_GAME_COUNT: usize = 200;
const TEST_GAME_COUNT: usize = 100;

fn main() {
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
    let mut stats = vec![(
        0,
        x_wins as f64 / TEST_GAME_COUNT as f64,
        o_wins as f64 / TEST_GAME_COUNT as f64,
        draws as f64 / TEST_GAME_COUNT as f64,
    )];

    'training_loop: for step in 0..STEPS {
        let existing_actor = current_actor
            .clone()
            .make_untrainable()
            .switch_player(CellState::X);
        print!(
            "[train_steps={TRAIN_STEPS}, capacity={CAPACITY}, future_discount={FUTURE_DISCOUNT}, epsilon={EPSILON}] Training games {:4} - {:4}: ",
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

            stats.push((
                (step + 1) * STEP_GAME_COUNT,
                x_wins as f64 / TEST_GAME_COUNT as f64,
                o_wins as f64 / TEST_GAME_COUNT as f64,
                draws as f64 / TEST_GAME_COUNT as f64,
            ));

            if o_wins as f32 / TEST_GAME_COUNT as f32 > 0.95 {
                println!("Model seems good enough, ending training");
                break 'training_loop;
            }
        }
    }

    let mut plot = plotly::Plot::new();
    let trace_x = plotly::Scatter::new(
        stats.iter().map(|(i, _, _, _)| *i).collect::<Vec<_>>(),
        stats.iter().map(|(_, x, _, _)| *x).collect::<Vec<_>>(),
    )
    .name("X wins")
    .mode(plotly::common::Mode::Lines);
    plot.add_trace(trace_x);
    let trace_o = plotly::Scatter::new(
        stats.iter().map(|(i, _, _, _)| *i).collect::<Vec<_>>(),
        stats.iter().map(|(_, _, o, _)| *o).collect::<Vec<_>>(),
    )
    .name("O wins")
    .mode(plotly::common::Mode::Lines);
    plot.add_trace(trace_o);
    let trace_d = plotly::Scatter::new(
        stats.iter().map(|(i, _, _, _)| *i).collect::<Vec<_>>(),
        stats.iter().map(|(_, _, _, d)| *d).collect::<Vec<_>>(),
    )
    .name("draws")
    .mode(plotly::common::Mode::Lines);
    plot.add_trace(trace_d);
    println!("Writing graph of results to stats.html");
    plot.write_html("stats.html");

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

    current_actor.1.save("models/tictactoe.npz").unwrap();
    println!("Saved model to models/tictactoe.npz");
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Hash)]
enum CellState {
    X,
    O,
    #[default]
    Empty,
}

impl EnumerablePlayer for CellState {
    fn all_players() -> Vec<CellState> {
        vec![CellState::X, CellState::O]
    }
}

impl Display for CellState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellState::X => write!(f, "X"),
            CellState::O => write!(f, "O"),
            CellState::Empty => write!(f, "-"),
        }
    }
}

impl From<char> for CellState {
    fn from(c: char) -> Self {
        match c {
            'X' => CellState::X,
            'O' => CellState::O,
            ' ' => CellState::Empty,
            _ => panic!("Invalid cell state {c}"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct TicTacToeState([[CellState; 3]; 3]);

impl TicTacToeState {
    fn x_reward(&self) -> Reward {
        if self.player_won(CellState::X) {
            Reward(100.0)
        } else if self.player_won(CellState::O) {
            Reward(-100.0)
        } else if self.draw() {
            Reward(-10.0)
        } else {
            Reward(0.0)
        }
    }

    fn o_reward(&self) -> Reward {
        if self.player_won(CellState::X) {
            Reward(-100.0)
        } else if self.player_won(CellState::O) {
            Reward(100.0)
        } else if self.draw() {
            Reward(-10.0)
        } else {
            Reward(0.0)
        }
    }

    fn player_won(&self, state: CellState) -> bool {
        for i in 0..3 {
            let mut row_won = true;
            let mut col_won = true;
            for j in 0..3 {
                if self.0[i][j] != state {
                    row_won = false;
                }
                if self.0[j][i] != state {
                    col_won = false;
                }
            }
            if row_won || col_won {
                return true;
            }
        }
        let mut left_diag_won = true;
        let mut right_diag_won = true;
        for i in 0..3 {
            if self.0[i][i] != state {
                right_diag_won = false;
            }
            if self.0[i][2 - i] != state {
                left_diag_won = false;
            }
        }
        left_diag_won || right_diag_won
    }

    fn draw(&self) -> bool {
        !itertools::iproduct!(0..3, 0..3).any(|(i, j)| matches!(self.0[i][j], CellState::Empty))
    }
}

impl ActorState<TicTacToeAction, CellState> for TicTacToeState {
    fn available_actions(&self) -> Vec<TicTacToeAction> {
        itertools::iproduct!(0..3, 0..3)
            .filter(|(i, j)| matches!(self.0[*i][*j], CellState::Empty))
            .map(|(i, j)| TicTacToeAction(i as u8, j as u8))
            .collect()
    }

    fn apply_action(&mut self, action: TicTacToeAction, player: CellState) -> bool {
        if matches!(
            self.0[action.0 as usize][action.1 as usize],
            CellState::Empty
        ) {
            self.0[action.0 as usize][action.1 as usize] = player;
            return true;
        } else {
            return false;
        }
    }

    fn is_game_over(&self) -> bool {
        self.player_won(CellState::X) || self.player_won(CellState::O) || self.draw()
    }

    fn who_won(&self) -> Option<CellState> {
        if self.player_won(CellState::X) {
            Some(CellState::X)
        } else if self.player_won(CellState::O) {
            Some(CellState::O)
        } else {
            None
        }
    }

    fn reward(&self, player: &CellState) -> Reward {
        match player {
            CellState::X => self.x_reward(),
            CellState::O => self.o_reward(),
            CellState::Empty => Reward::default(),
        }
    }
}

impl Display for TicTacToeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..3 {
            write!(f, " ")?;
            for j in 0..3 {
                write!(f, "{}", self.0[i][j])?;
                if j < 2 {
                    write!(f, "|")?;
                }
            }
            if i < 2 {
                write!(f, "\n -----")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl EncodableState<9, CellState> for TicTacToeState {
    fn encode<D: Device<f32>>(&self, context: CellState, device: &D) -> Tensor<Rank1<9>, f32, D> {
        let mut raw_tensor = [0.0; 9];
        let sign = match context {
            CellState::X => 1.0,
            CellState::O => -1.0,
            CellState::Empty => 0.0,
        };
        for i in 0..3 {
            for j in 0..3 {
                raw_tensor[i * 3 + j] = match self.0[i][j] {
                    CellState::X => 1.0 * sign,
                    CellState::O => -1.0 * sign,
                    CellState::Empty => 0.0,
                }
            }
        }
        device.tensor(raw_tensor)
    }
}

#[derive(Clone, Debug)]
struct TicTacToeAction(u8, u8);

impl EncodableAction for TicTacToeAction {
    fn encode(&self) -> usize {
        (self.0 * 3 + self.1) as usize
    }

    fn decode(index: usize) -> Self {
        TicTacToeAction((index / 3) as u8, (index % 3) as u8)
    }
}

type TicTacToeNetwork = (
    (Linear<9, 32>, ReLU),
    Dropout,
    (Linear<32, 32>, ReLU),
    Linear<32, 9>,
);

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx_qlearning::actor::Actor;
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
    #[case([['X', ' ', ' '], ['O', 'X', 'O'], [' ', ' ', ' ']], CellState::X, 2, 2)]
    #[case([['O', 'X', ' '], ['X', 'O', 'X'], [' ', 'X', ' ']], CellState::O, 2, 2)]
    #[case([['O', 'O', ' '], [' ', ' ', 'X'], [' ', 'X', ' ']], CellState::O, 0, 2)]
    #[case([['X', 'X', ' '], ['O', ' ', ' '], [' ', 'O', ' ']], CellState::O, 0, 2)]
    #[case([['O', 'O', ' '], [' ', ' ', 'X'], [' ', 'X', ' ']], CellState::X, 0, 2)]
    fn winning_move_is_chosen(
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
