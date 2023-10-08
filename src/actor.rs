use crate::abstract_model::{AbstractModel, EncodableAction, EncodableState, Reward};
use dfdx::{
    prelude::{BuildOnDevice, Module, ModuleMut, TensorCollection},
    shapes::{Const, Rank1},
    tensor::{Cpu, OwnedTape, Tensor},
};
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{
    cell::Cell,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
};

pub trait GameLogger<Player, State> {
    fn log_state(&self, player: &Player, state: &State);

    fn log_wins(&self, player: &Player, count: usize, game_count: usize);

    fn log_draws(&self, count: usize, game_count: usize);
}

struct TrivialGameLogger;

impl<Player, State> GameLogger<Player, State> for TrivialGameLogger {
    fn log_state(&self, _player: &Player, _state: &State) {}

    fn log_wins(&self, _player: &Player, _count: usize, _game_count: usize) {}

    fn log_draws(&self, _count: usize, _game_count: usize) {}
}

struct DisplayGameLogger;

impl<Player: Display, State: Display> GameLogger<Player, State> for DisplayGameLogger {
    fn log_state(&self, player: &Player, state: &State) {
        println!("After {player} move:\n{state}\n");
    }

    fn log_wins(&self, player: &Player, count: usize, game_count: usize) {
        print!("{player} wins = {count:4} / {game_count:4}, ",);
    }

    fn log_draws(&self, count: usize, game_count: usize) {
        println!("draws = {count:4} / {game_count:4}");
    }
}

pub struct Engine<Player, Action, State, Logger> {
    logger: Logger,
    phantom_player: PhantomData<Player>,
    phantom_action: PhantomData<Action>,
    phantom_state: PhantomData<State>,
}

impl<
        Player: EnumerablePlayer + Sized,
        Action,
        State: ActorState<Action, Player> + Default,
        Logger: GameLogger<Player, State>,
    > Engine<Player, Action, State, Logger>
{
    #[allow(unused)]
    pub fn new(logger: Logger) -> Self {
        Self {
            logger,
            phantom_player: PhantomData,
            phantom_action: PhantomData,
            phantom_state: PhantomData,
        }
    }

    #[allow(unused)]
    pub fn play_once(&self, actors: &[(&Player, &dyn Actor<State, Action>)]) -> Option<Player> {
        let mut state = State::default();
        while !state.is_game_over() {
            for (player, actor) in actors {
                actor.play_step(&mut state);
                self.logger.log_state(player, &state);
            }
        }
        state.who_won()
    }
}

impl<
        Player: EnumerablePlayer + Sized + Hash + Eq,
        Action,
        State: ActorState<Action, Player> + Default,
        Logger: GameLogger<Player, State>,
    > Engine<Player, Action, State, Logger>
{
    #[allow(unused)]
    pub fn train_players(
        &self,
        actors: &mut [(&Player, &mut dyn Actor<State, Action>)],
        game_count: usize,
    ) {
        let mut wins = HashMap::new();
        let mut draws = 0;
        for _ in 0..game_count {
            let mut state = State::default();
            while !state.is_game_over() {
                for (player, actor) in actors.iter_mut() {
                    if let Some(step) = actor.play_step_to_train(&mut state) {
                        actor.record(step, state.reward(player));
                    }
                }
            }
            match state.who_won() {
                Some(player) => *wins.entry(player).or_insert(0) += 1,
                None => draws += 1,
            }
            for (_, actor) in actors.iter_mut() {
                actor.train();
            }
        }
        for (player, _) in actors.iter_mut() {
            self.logger
                .log_wins(player, wins.get(player).copied().unwrap_or(0), game_count);
        }
        self.logger.log_draws(draws, game_count);
    }
}

pub trait Actor<State, Action> {
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>>;

    fn play_step(&self, state: &mut State);

    fn record(&mut self, step: Step<State, Action>, reward: Reward);

    fn train(&mut self);
}

pub struct Step<State, Action> {
    old_state: State,
    action: Action,
    new_state: State,
}

pub trait ActorState<Action, Player>: Hash + Eq + Clone {
    fn available_actions(&self) -> Vec<Action>;

    fn apply_action(&mut self, action: Action, player: Player) -> bool;

    fn is_game_over(&self) -> bool;

    fn who_won(&self) -> Option<Player>;

    fn reward(&self, player: &Player) -> Reward;
}

pub trait EnumerablePlayer: Sized {
    fn all_players() -> Vec<Self>;
}

#[derive(Clone, Debug)]
pub struct TrainableActor<
    State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
    Action: EncodableAction,
    Player,
    Model: BuildOnDevice<Cpu, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
>(
    pub Player,
    pub AbstractModel<State, Player, Action, Model, N_FEATURES, N_ACTIONS>,
)
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug + Clone;

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
        Action: EncodableAction,
        Player,
        Model: BuildOnDevice<Cpu, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > TrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug + Clone,
{
    #[allow(unused)]
    pub fn make_untrainable(
        self,
    ) -> UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS> {
        UntrainableActor(self.0, self.1)
    }
}

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
        Action: EncodableAction + Clone,
        Player: Clone,
        Model: BuildOnDevice<Cpu, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > Actor<State, Action> for TrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cpu>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cpu>>
        + ModuleMut<
            Tensor<(usize, Const<N_FEATURES>), f32, Cpu, OwnedTape<f32, Cpu>>,
            Output = Tensor<(usize, Const<N_ACTIONS>), f32, Cpu, OwnedTape<f32, Cpu>>,
        > + TensorCollection<f32, Cpu>,
{
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>> {
        let old_state = state.clone();
        let candidates = state.available_actions();
        if !candidates.is_empty() {
            let action = self
                .1
                .choose_with_epsilon_greedy(&state, self.0.clone(), &candidates);
            state.apply_action(action.clone(), self.0.clone());
            Some(Step {
                old_state,
                action,
                new_state: state.clone(),
            })
        } else {
            None
        }
    }

    fn play_step(&self, state: &mut State) {
        let candidates = state.available_actions();
        if !candidates.is_empty() {
            let action = self
                .1
                .choose_from_model_only(&state, self.0.clone(), &candidates);
            state.apply_action(action, self.0.clone());
        }
    }

    fn record(&mut self, step: Step<State, Action>, reward: Reward) {
        self.1
            .record(step.old_state, step.action, reward, step.new_state);
    }

    fn train(&mut self) {
        self.1.train(self.0.clone());
    }
}

pub struct UntrainableActor<
    State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
    Action: EncodableAction,
    Player,
    Model: BuildOnDevice<Cpu, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
>(
    Player,
    AbstractModel<State, Player, Action, Model, N_FEATURES, N_ACTIONS>,
)
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug + Clone;

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
        Action: EncodableAction,
        Player,
        Model: BuildOnDevice<Cpu, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug + Clone,
{
    #[allow(unused)]
    pub fn switch_player(self, new_player: Player) -> Self {
        Self(new_player, self.1)
    }
}

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
        Action: EncodableAction + Clone,
        Player: Clone,
        Model: BuildOnDevice<Cpu, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > Actor<State, Action> for UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cpu>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cpu>>,
{
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>> {
        self.play_step(state);
        None
    }

    fn play_step(&self, state: &mut State) {
        let candidates = state.available_actions();
        if !candidates.is_empty() {
            let action = self
                .1
                .choose_from_model_only(&state, self.0.clone(), &candidates);
            state.apply_action(action, self.0.clone());
        }
    }

    fn record(&mut self, _step: Step<State, Action>, _reward: Reward) {}

    fn train(&mut self) {}
}

pub struct NaiveActor<Player>(Player, Cell<Option<Xoshiro256PlusPlus>>);

impl<Player> NaiveActor<Player> {
    #[allow(unused)]
    pub fn new(player: Player) -> Self {
        Self(player, Cell::new(None))
    }
}

impl<State: ActorState<Action, Player>, Action: Clone, Player: Clone> Actor<State, Action>
    for NaiveActor<Player>
{
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>> {
        self.play_step(state);
        None
    }

    fn play_step(&self, state: &mut State) {
        let candidates = state.available_actions();
        if !candidates.is_empty() {
            let mut rng = self
                .1
                .take()
                .unwrap_or_else(|| Xoshiro256PlusPlus::from_rng(thread_rng()).unwrap());
            let action = candidates[rng.gen_range(0..candidates.len())].clone();
            state.apply_action(action, self.0.clone());
            self.1.set(Some(rng));
        }
    }

    fn record(&mut self, _step: Step<State, Action>, _reward: Reward) {}

    fn train(&mut self) {}
}

#[cfg(test)]
mod tests {
    mod tictactoe {
        use crate::abstract_model::{AbstractModel, EncodableAction, EncodableState, Reward};
        use crate::actor::{
            Actor, ActorState, DisplayGameLogger, Engine, EnumerablePlayer, NaiveActor,
            TrainableActor,
        };
        use dfdx::{
            nn::builders::Linear,
            prelude::ReLU,
            shapes::Rank1,
            tensor::{Tensor, TensorFrom},
            tensor_ops::Device,
        };
        use googletest::prelude::*;
        use std::{
            fmt::{Debug, Display},
            hash::Hash,
        };

        const GAME_COUNT: usize = 5000;

        #[test]
        fn trained_x_model_wins_over_untrained_o_model() -> Result<()> {
            let mut x_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
                CellState::X,
                AbstractModel::new(10, 0.9, 0.3, 10000),
            );
            let mut naive_actor = NaiveActor::new(CellState::O);
            let engine = Engine::new(DisplayGameLogger);

            engine.train_players(
                &mut [
                    (&CellState::X, &mut x_actor),
                    (&CellState::O, &mut naive_actor),
                ],
                GAME_COUNT,
            );

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
            let mut o_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
                CellState::O,
                AbstractModel::new(10, 0.9, 0.3, 10000),
            );
            let mut naive_actor = NaiveActor::new(CellState::X);
            let engine = Engine::new(DisplayGameLogger);

            engine.train_players(
                &mut [
                    (&CellState::X, &mut naive_actor),
                    (&CellState::O, &mut o_actor),
                ],
                GAME_COUNT,
            );

            verify_that!(
                win_count_after_n_games(
                    100,
                    &[(&CellState::X, &naive_actor), (&CellState::O, &o_actor)],
                    CellState::O
                ),
                ge(80)
            )
        }

        fn win_count_after_n_games(
            game_count: u32,
            actors: &[(&CellState, &dyn Actor<TicTacToeState, TicTacToeAction>)],
            count_player: CellState,
        ) -> u32 {
            let mut wins = 0;
            let engine = Engine::new(DisplayGameLogger);
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

        #[rstest::rstest]
        #[case(5, 10000, 0.9, 0.3)]
        #[case(10, 10000, 0.9, 0.3)]
        #[ignore = "Test for diagnostic purposes only"]
        fn model_improves_with_training(
            #[case] train_steps: usize,
            #[case] capacity: usize,
            #[case] future_discount: f32,
            #[case] epsilon: f32,
        ) -> Result<()> {
            const STEPS: usize = 100;
            const STEP_GAME_COUNT: usize = 50;
            const TEST_GAME_COUNT: usize = 1000;
            let mut current_actor = TrainableActor::<TicTacToeState, _, _, TicTacToeNetwork, 9, 9>(
                CellState::O,
                AbstractModel::new(train_steps, future_discount, epsilon, capacity),
            );
            let mut previous_actor = current_actor
                .clone()
                .make_untrainable()
                .switch_player(CellState::X);
            let mut naive_actor = NaiveActor::new(CellState::X);
            let engine = Engine::new(DisplayGameLogger);

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
            println!("[train_steps={train_steps}, capacity={capacity}, future_discount={future_discount}, epsilon={epsilon}] After    0 games:           X wins = {x_wins:4} / {TEST_GAME_COUNT:4}, O wins = {o_wins:4} / {TEST_GAME_COUNT:4}, draws = {draws:4} / {TEST_GAME_COUNT:4}");

            for step in 0..STEPS {
                let existing_actor = current_actor
                    .clone()
                    .make_untrainable()
                    .switch_player(CellState::X);
                print!(
                    "[train_steps={train_steps}, capacity={capacity}, future_discount={future_discount}, epsilon={epsilon}] Training games {:4} - {:4}: ",
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

                    println!("[train_steps={train_steps}, capacity={capacity}, future_discount={future_discount}, epsilon={epsilon}] After {:4} games:           X wins = {x_wins:4} / {TEST_GAME_COUNT:4}, draws = {draws:4} / {TEST_GAME_COUNT:4}, O wins = {o_wins:4} / {TEST_GAME_COUNT:4}", (step + 1) * STEP_GAME_COUNT);
                }
            }

            println!("Sample game against naive actor:");
            engine.play_once(&[
                (&CellState::X, &naive_actor),
                (&CellState::O, &current_actor),
            ]);
            println!("Sample game against previous trained actor:");
            engine.play_once(&[
                (&CellState::X, &previous_actor),
                (&CellState::O, &current_actor),
            ]);

            verify_that!(o_wins as f32 / TEST_GAME_COUNT as f32, ge(0.80))
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
                !itertools::iproduct!(0..3, 0..3)
                    .any(|(i, j)| matches!(self.0[i][j], CellState::Empty))
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
                        match self.0[i][j] {
                            CellState::X => write!(f, "X")?,
                            CellState::O => write!(f, "O")?,
                            CellState::Empty => write!(f, " ")?,
                        };
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
            fn encode<D: Device<f32>>(
                &self,
                context: CellState,
                device: &D,
            ) -> Tensor<Rank1<9>, f32, D> {
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

        type TicTacToeNetwork = ((Linear<9, 32>, ReLU), Linear<32, 9>);
    }
}
