use dfdx::{
    nn::builders::Linear,
    prelude::ReLU,
    shapes::Rank1,
    tensor::{Tensor, TensorFrom},
    tensor_ops::Device,
};
use dfdx_qlearning::{
    abstract_model::{AbstractModel, EncodableAction, EncodableState, Reward},
    actor::{ActorState, Engine, EnumerablePlayer, NaiveActor, TrainableActor},
    game_logger::DisplayGameLogger,
};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

const TRAIN_STEPS: usize = 100;
const CAPACITY: usize = 10000;
const FUTURE_DISCOUNT: f32 = 1.0;
const EPSILON: f32 = 1.0;
const STEPS: usize = 100;
const STEP_GAME_COUNT: usize = 200;
const TEST_GAME_COUNT: usize = 1000;

fn main() {
    let mut current_actor = TrainableActor::<ConnectFourState, _, _, ConnectFourNetwork, 42, 7>(
        CellState::Blue,
        AbstractModel::new(TRAIN_STEPS, FUTURE_DISCOUNT, EPSILON, CAPACITY),
    );
    let mut previous_actor = current_actor
        .clone()
        .make_untrainable()
        .switch_player(CellState::Red);
    let mut naive_actor = NaiveActor::new(CellState::Red);
    let engine = Engine::new(DisplayGameLogger);

    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut draws = 0;
    for _ in 0..TEST_GAME_COUNT {
        match engine.play_once(&[
            (&CellState::Red, &naive_actor),
            (&CellState::Blue, &current_actor),
        ]) {
            Some(CellState::Red) => red_wins += 1,
            Some(CellState::Blue) => blue_wins += 1,
            None | Some(CellState::Empty) => draws += 1,
        }
    }

    let mut stats = vec![(
        0,
        red_wins as f64 / TEST_GAME_COUNT as f64,
        blue_wins as f64 / TEST_GAME_COUNT as f64,
        draws as f64 / TEST_GAME_COUNT as f64,
    )];

    'training_loop: for step in 0..STEPS {
        let existing_actor = current_actor
            .clone()
            .make_untrainable()
            .switch_player(CellState::Red);
        println!(
            "[train_steps={TRAIN_STEPS}, capacity={CAPACITY}, future_discount={FUTURE_DISCOUNT}, epsilon={EPSILON}] Training games {:4} - {:4}",
            step * STEP_GAME_COUNT,
            (step + 1) * STEP_GAME_COUNT - 1
        );
        engine.train_players(
            &mut [
                (&CellState::Red, &mut previous_actor),
                (&CellState::Blue, &mut current_actor),
            ],
            STEP_GAME_COUNT,
        );
        engine.train_players(
            &mut [
                (&CellState::Red, &mut naive_actor),
                (&CellState::Blue, &mut current_actor),
            ],
            STEP_GAME_COUNT,
        );
        previous_actor = existing_actor;

        for _ in 0..1 {
            red_wins = 0;
            blue_wins = 0;
            draws = 0;

            for _ in 0..TEST_GAME_COUNT {
                match engine.play_once(&[
                    (&CellState::Red, &naive_actor),
                    (&CellState::Blue, &current_actor),
                ]) {
                    Some(CellState::Red) => red_wins += 1,
                    Some(CellState::Blue) => blue_wins += 1,
                    None | Some(CellState::Empty) => draws += 1,
                }
            }

            stats.push((
                (step + 1) * STEP_GAME_COUNT,
                red_wins as f64 / TEST_GAME_COUNT as f64,
                blue_wins as f64 / TEST_GAME_COUNT as f64,
                draws as f64 / TEST_GAME_COUNT as f64,
            ));

            if blue_wins as f32 / TEST_GAME_COUNT as f32 > 0.85 {
                println!("Model seems good enough, ending training");
                break 'training_loop;
            }
        }
    }

    println!("Sample game against naive actor:");
    engine.play_once(&[
        (&CellState::Red, &naive_actor),
        (&CellState::Blue, &current_actor),
    ]);
    println!("Sample game against previous trained actor:");
    engine.play_once(&[
        (&CellState::Red, &previous_actor),
        (&CellState::Blue, &current_actor),
    ]);

    write_stats(&stats);

    println!("Saving model to models/connectfour.npz");
    current_actor.1.save("models/connectfour.npz").unwrap();
}

fn write_stats(stats: &[(usize, f64, f64, f64)]) {
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

    const DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/stats/connectfour");
    std::fs::create_dir_all(DIRECTORY).unwrap();
    println!("Writing graph of results to {DIRECTORY}/stats.html");
    plot.write_html(format!("{DIRECTORY}/stats.html"));
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Hash)]
enum CellState {
    Red,
    Blue,
    #[default]
    Empty,
}

impl EnumerablePlayer for CellState {
    fn all_players() -> Vec<CellState> {
        vec![CellState::Red, CellState::Blue]
    }
}

impl Display for CellState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellState::Red => write!(f, "R"),
            CellState::Blue => write!(f, "B"),
            CellState::Empty => write!(f, " "),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ConnectFourState([[CellState; 6]; 7]);

impl ConnectFourState {
    fn red_reward(&self) -> Reward {
        if self.player_won(CellState::Red) {
            Reward(100.0)
        } else if self.player_won(CellState::Blue) {
            Reward(-100.0)
        } else if self.draw() {
            Reward(-10.0)
        } else {
            Reward(0.0)
        }
    }

    fn blue_reward(&self) -> Reward {
        if self.player_won(CellState::Red) {
            Reward(-100.0)
        } else if self.player_won(CellState::Blue) {
            Reward(100.0)
        } else if self.draw() {
            Reward(-10.0)
        } else {
            Reward(0.0)
        }
    }

    fn player_won(&self, state: CellState) -> bool {
        for i in 0..7 {
            for j in 0..6 {
                let mut won_horiz = true;
                let mut won_vert = true;
                let mut won_diag_up = true;
                let mut won_diag_down = true;
                for k in 0..4 {
                    if i + k >= 7 || self.0[i + k][j] != state {
                        won_horiz = false;
                    }
                    if j + k >= 6 || self.0[i][j + k] != state {
                        won_vert = false;
                    }
                    if i + k >= 7 || j < k || self.0[i + k][j - k] != state {
                        won_diag_up = false;
                    }
                    if i + k >= 7 || j + k >= 6 || self.0[i + k][j + k] != state {
                        won_diag_down = false;
                    }
                }
                if won_horiz || won_vert || won_diag_up || won_diag_down {
                    return true;
                }
            }
        }
        false
    }

    fn draw(&self) -> bool {
        !(0..7).any(|i| matches!(self.0[i][5], CellState::Empty))
    }
}

impl ActorState<ConnectFourAction, CellState> for ConnectFourState {
    fn available_actions(&self) -> Vec<ConnectFourAction> {
        (0..7)
            .filter(|i| matches!(self.0[*i][5], CellState::Empty))
            .map(|i| ConnectFourAction(i as u8))
            .collect()
    }

    fn apply_action(&mut self, action: ConnectFourAction, player: CellState) -> bool {
        if let Some(index) =
            (0..6).find(|i| matches!(self.0[action.0 as usize][*i], CellState::Empty))
        {
            self.0[action.0 as usize][index] = player;
            return true;
        } else {
            return false;
        }
    }

    fn is_game_over(&self) -> bool {
        self.player_won(CellState::Red) || self.player_won(CellState::Blue) || self.draw()
    }

    fn who_won(&self) -> Option<CellState> {
        if self.player_won(CellState::Red) {
            Some(CellState::Red)
        } else if self.player_won(CellState::Blue) {
            Some(CellState::Blue)
        } else {
            None
        }
    }

    fn reward(&self, player: &CellState) -> Reward {
        match player {
            CellState::Red => self.red_reward(),
            CellState::Blue => self.blue_reward(),
            CellState::Empty => Reward::default(),
        }
    }
}

impl Display for ConnectFourState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for j in 0..6 {
            write!(f, " ")?;
            for i in 0..7 {
                write!(f, " {} ", self.0[i][5 - j])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl EncodableState<42, CellState> for ConnectFourState {
    fn encode<D: Device<f32>>(&self, context: CellState, device: &D) -> Tensor<Rank1<42>, f32, D> {
        let mut raw_tensor = [0.0; 42];
        let sign = match context {
            CellState::Red => 1.0,
            CellState::Blue => -1.0,
            CellState::Empty => 0.0,
        };
        for i in 0..7 {
            for j in 0..6 {
                raw_tensor[i * 6 + j] = match self.0[i][j] {
                    CellState::Red => 1.0 * sign,
                    CellState::Blue => -1.0 * sign,
                    CellState::Empty => 0.0,
                }
            }
        }
        device.tensor(raw_tensor)
    }
}

#[derive(Clone, Debug)]
struct ConnectFourAction(u8);

impl EncodableAction for ConnectFourAction {
    fn encode(&self) -> usize {
        self.0 as usize
    }

    fn decode(index: usize) -> Self {
        ConnectFourAction(index as u8)
    }
}

type ConnectFourNetwork = (Linear<42, 96>, ReLU, Linear<96, 96>, ReLU, Linear<96, 7>);

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx_qlearning::{actor::Actor, game_logger::TrivialGameLogger};
    use googletest::prelude::*;

    #[test]
    fn trained_red_model_wins_over_untrained_blue_model() -> Result<()> {
        let red_actor = TrainableActor::<ConnectFourState, _, _, ConnectFourNetwork, 42, 7>(
            CellState::Red,
            AbstractModel::load("models/connectfour.npz", 10, 0.9, 0.3, 10000)?,
        );
        let naive_actor = NaiveActor::new(CellState::Blue);

        verify_that!(
            win_count_after_n_games(
                100,
                &[
                    (&CellState::Red, &red_actor),
                    (&CellState::Blue, &naive_actor)
                ],
                CellState::Red
            ),
            ge(80)
        )
    }

    #[test]
    fn trained_blue_model_wins_over_untrained_red_model() -> Result<()> {
        let blue_actor = TrainableActor::<ConnectFourState, _, _, ConnectFourNetwork, 42, 7>(
            CellState::Blue,
            AbstractModel::load("models/connectfour.npz", 10, 0.9, 0.3, 10000)?,
        );
        let naive_actor = NaiveActor::new(CellState::Red);

        verify_that!(
            win_count_after_n_games(
                100,
                &[
                    (&CellState::Red, &naive_actor),
                    (&CellState::Blue, &blue_actor)
                ],
                CellState::Blue
            ),
            ge(80)
        )
    }

    fn win_count_after_n_games(
        game_count: u32,
        actors: &[(&CellState, &dyn Actor<ConnectFourState, ConnectFourAction>)],
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
