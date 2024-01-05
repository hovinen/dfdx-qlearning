use crate::{
    abstract_model::{AbstractModel, EncodableAction, EncodableState, Reward},
    game_logger::GameLogger,
};
use dfdx::{
    prelude::{BuildOnDevice, Module, ModuleMut, TensorCollection},
    shapes::{Const, Rank1},
    tensor::{Cuda, OwnedTape, Tensor},
};
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{cell::Cell, collections::HashMap, fmt::Debug, hash::Hash, marker::PhantomData};

pub struct Engine<Player, Action, State, Logger> {
    logger: Logger,
    phantom_player: PhantomData<Player>,
    phantom_action: PhantomData<Action>,
    phantom_state: PhantomData<State>,
}

impl<
        Player: EnumerablePlayer + Sized,
        Action: Clone + Debug,
        State: ActorState<Action, Player> + Default + Clone + Debug,
        Logger: GameLogger<Player, State>,
    > Engine<Player, Action, State, Logger>
{
    pub fn new(logger: Logger) -> Self {
        Self {
            logger,
            phantom_player: PhantomData,
            phantom_action: PhantomData,
            phantom_state: PhantomData,
        }
    }

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
        Action: Clone + Debug,
        State: ActorState<Action, Player> + Default + Clone + Debug,
        Logger: GameLogger<Player, State>,
    > Engine<Player, Action, State, Logger>
{
    pub fn train_players(
        &self,
        actors: &mut [(&Player, &mut dyn Actor<State, Action>)],
        game_count: usize,
    ) {
        let mut wins = HashMap::new();
        let mut draws = 0;
        for _ in 0..game_count {
            let mut state = State::default();
            let mut steps: Vec<Option<Step<State, Action>>> = vec![None; actors.len()];
            while !state.is_game_over() {
                for ((player, actor), step) in actors.iter_mut().zip(steps.iter_mut()) {
                    if let Some(mut step) = step.take() {
                        step.new_state = state.clone();
                        actor.record(step, state.reward(player));
                    }
                    *step = actor.play_step_to_train(&mut state);
                }
            }
            for ((player, actor), step) in actors.iter_mut().zip(steps.iter_mut()) {
                if let Some(mut step) = step.take() {
                    step.new_state = state.clone();
                    actor.record(step, state.reward(player));
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

pub trait Actor<State: Clone + Debug, Action: Clone + Debug> {
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>>;

    fn play_step(&self, state: &mut State);

    fn record(&mut self, step: Step<State, Action>, reward: Reward);

    fn train(&mut self);
}

#[derive(Clone, Debug)]
pub struct Step<State: Clone + Debug, Action: Clone + Debug> {
    pub old_state: State,
    pub action: Action,
    pub new_state: State,
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
    Model: BuildOnDevice<Cuda, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
>(
    pub Player,
    pub AbstractModel<State, Player, Action, Model, N_FEATURES, N_ACTIONS>,
)
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug + Clone;

impl<
        State,
        Action: EncodableAction + Clone,
        Player: Clone,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > TrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    State: ActorState<Action, Player>
        + EncodableState<N_FEATURES, Player>
        + Hash
        + PartialEq
        + Eq
        + Clone,
    <Model as BuildOnDevice<Cpu, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>,
{
    #[allow(unused)]
    pub fn make_untrainable(
        self,
    ) -> UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS> {
        UntrainableActor(self.0, self.1)
    }

    pub fn evaluate(&self, state: &State) -> Vec<f32> {
        self.1.evaluate_training(state, self.0.clone())
    }

    pub fn evaluate_training(&self, state: &State) -> Vec<f32> {
        self.1.evaluate_training(state, self.0.clone())
    }
}

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player> + Debug,
        Action: EncodableAction + Clone + Debug,
        Player: Clone,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > Actor<State, Action> for TrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>
        + ModuleMut<
            Tensor<(usize, Const<N_FEATURES>), f32, Cuda, OwnedTape<f32, Cuda>>,
            Output = Tensor<(usize, Const<N_ACTIONS>), f32, Cuda, OwnedTape<f32, Cuda>>,
        > + TensorCollection<f32, Cuda>,
{
    fn play_step_to_train(&self, state: &mut State) -> Option<Step<State, Action>> {
        let old_state = state.clone();
        let candidates = state.available_actions();
        if !candidates.is_empty() {
            let action = self
                .1
                .choose_with_epsilon_greedy(state, self.0.clone(), &candidates);
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
                .choose_from_model_only(state, self.0.clone(), &candidates);
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
    Model: BuildOnDevice<Cuda, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
>(
    Player,
    pub AbstractModel<State, Player, Action, Model, N_FEATURES, N_ACTIONS>,
)
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug + Clone;

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player>,
        Action: EncodableAction,
        Player,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug + Clone,
{
    #[allow(unused)]
    pub fn switch_player(self, new_player: Player) -> Self {
        Self(new_player, self.1)
    }
}

impl<
        State: ActorState<Action, Player> + EncodableState<N_FEATURES, Player> + Debug,
        Action: EncodableAction + Clone + Debug,
        Player: Clone,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > Actor<State, Action> for UntrainableActor<State, Action, Player, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>,
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
                .choose_from_model_only(state, self.0.clone(), &candidates);
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

impl<State: ActorState<Action, Player> + Debug, Action: Clone + Debug, Player: Clone>
    Actor<State, Action> for NaiveActor<Player>
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
