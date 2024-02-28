use dfdx::{
    nn::modules::*,
    optim::Adam,
    prelude::{mse_loss, BuildOnDevice, DeviceBuildExt, Optimizer, ZeroGrads},
    shapes::{Const, Rank1},
    tensor::{AutoDevice, Cuda, OwnedTape, Tensor, TensorFrom, Trace},
    tensor_ops::{AdamConfig, Backward, Device, WeightDecay},
};
use multimap::MultiMap;
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
};

/// Factor by which epsilon, the probability of choosing a random action rather than using the
/// model, decays in each training iteration.
const EPSILON_DECAY: f32 = 0.99995;

/// The minimum value of epsilon, the probability of choosing a random action rather than using the
/// model. It will decay until it reaches this point.
const MIN_EPSILON: f32 = 0.01;

/// Reduction per iteration of contribution of training model when updating the ground model.
const TAU_DECAY: f32 = 0.9995;

/// Initial contribution of training model when updating the ground model.
const TAU_INIT: f32 = 0.1;

/// Time steps after which we update the ground model with the training model.
const C: usize = 2;

pub trait EncodableState<const N_FEATURES: usize, Context> {
    fn encode<D: Device<f32>>(
        &self,
        context: Context,
        device: &D,
    ) -> Tensor<Rank1<N_FEATURES>, f32, D>;

    fn is_terminal(&self) -> bool;
}

pub trait EncodableAction {
    fn encode(&self) -> usize;

    fn decode(index: usize) -> Self;
}

#[derive(Debug, Clone)]
pub struct AbstractModel<
    State: EncodableState<N_FEATURES, Context> + Hash + PartialEq + Eq + Clone,
    Context,
    Action: EncodableAction,
    Model: BuildOnDevice<Cuda, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
> where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug + Clone,
{
    device: AutoDevice,
    network: NeuralNetwork<Model, N_FEATURES, N_ACTIONS>,
    training_examples: TrainingExamples<State, Action>,
    future_discount: f32,
    epsilon: f32,
    tick: usize,
    phantom: PhantomData<Context>,
}

impl<
        State: EncodableState<N_FEATURES, Context> + Hash + PartialEq + Eq + Clone,
        Context: Clone,
        Action: EncodableAction + Clone,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > AbstractModel<State, Context, Action, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>,
{
    pub fn new(train_steps: usize, future_discount: f32, epsilon: f32, capacity: usize) -> Self {
        let device = AutoDevice::seed_from_u64(thread_rng().gen());
        let network = NeuralNetwork::new(&device, train_steps);
        Self {
            device,
            network,
            training_examples: TrainingExamples::new(capacity),
            future_discount,
            epsilon,
            tick: 0,
            phantom: Default::default(),
        }
    }

    pub fn load<'a>(
        path: &'a str,
        train_steps: usize,
        future_discount: f32,
        epsilon: f32,
        capacity: usize,
    ) -> std::io::Result<Self>
    where
        <Model as BuildOnDevice<Cuda, f32>>::Built: dfdx::nn::LoadFromNpz<f32, Cuda>,
    {
        let device = AutoDevice::seed_from_u64(thread_rng().gen());
        let network = NeuralNetwork::load(path, &device, train_steps)?;
        Ok(Self {
            device,
            network,
            training_examples: TrainingExamples::new(capacity),
            future_discount,
            epsilon,
            tick: 0,
            phantom: Default::default(),
        })
    }

    // TODO: Support multiple actions by, e.g., taking all actions above a threshold.
    pub fn choose_with_epsilon_greedy(
        &self,
        state: &State,
        context: Context,
        candidates: &[Action],
    ) -> Action
    where
        Action: Debug,
    {
        let mut rng = Xoshiro256PlusPlus::from_rng(thread_rng()).unwrap();
        if rng.gen_range(0.0..1.0) < self.epsilon {
            candidates[rng.gen_range(0..candidates.len())].clone()
        } else {
            self.choose_from_model_only(state, context, candidates)
        }
    }

    pub fn choose_from_model_only(
        &self,
        state: &State,
        context: Context,
        candidates: &[Action],
    ) -> Action
    where
        Action: Debug,
    {
        assert!(!candidates.is_empty());
        let candidate_indices = candidates
            .iter()
            .map(|c| c.encode())
            .collect::<HashSet<_>>();
        let scores = self.evaluate(state, context);
        let filtered_scores = scores
            .iter()
            .enumerate()
            .filter(|(i, s)| candidate_indices.contains(i) && **s >= 0.0)
            .collect::<Vec<_>>();
        let total_score = filtered_scores.iter().map(|(_, s)| *s).sum::<f32>();
        let chosen_index = if total_score > 0.0 {
            let mut v = thread_rng().gen_range(0.0..total_score);
            let mut chosen_index = filtered_scores.last().unwrap().0;
            for (index, &score) in filtered_scores.into_iter() {
                if v < score {
                    chosen_index = index;
                    break;
                }
                v -= score;
            }
            chosen_index
        } else if !filtered_scores.is_empty() {
            filtered_scores[thread_rng().gen_range(0..filtered_scores.len())].0
        } else {
            candidate_indices
                .into_iter()
                .max_by(|i1, i2| scores[*i1].total_cmp(&scores[*i2]))
                .unwrap()
        };
        Action::decode(chosen_index)
    }

    #[allow(unused)]
    fn output_scores(&self, scores: &[f32], candidates: &[Action])
    where
        Action: Debug,
    {
        println!("Scores:");
        for candidate in candidates {
            println!("{candidate:?}: {}", scores[candidate.encode()]);
        }
    }

    pub fn evaluate(&self, state: &State, context: Context) -> Vec<f32> {
        let input = state.encode(context, &self.device);
        self.network.evaluate(&input).as_vec()
    }

    pub fn evaluate_training(&self, state: &State, context: Context) -> Vec<f32> {
        let input = state.encode(context, &self.device);
        self.network.evaluate_training(&input).as_vec()
    }

    pub fn record(&mut self, state: State, action: Action, reward: Reward, new_state: State) {
        let is_terminal = new_state.is_terminal();
        self.training_examples
            .add(state, action, reward, new_state, is_terminal)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()>
    where
        <Model as BuildOnDevice<Cuda, f32>>::Built: dfdx::nn::SaveToNpz<f32, Cuda>,
    {
        Ok(self.network.model.save(path)?)
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn tau(&self) -> f32 {
        self.network.tau
    }
}

impl<
        State: EncodableState<N_FEATURES, Context> + Hash + PartialEq + Eq + Clone + Debug,
        Context: Clone,
        Action: EncodableAction + Clone + Debug,
        Model: BuildOnDevice<Cuda, f32>,
        const N_FEATURES: usize,
        const N_ACTIONS: usize,
    > AbstractModel<State, Context, Action, Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>
        + ModuleMut<
            Tensor<(usize, Const<N_FEATURES>), f32, Cuda, OwnedTape<f32, Cuda>>,
            Output = Tensor<(usize, Const<N_ACTIONS>), f32, Cuda, OwnedTape<f32, Cuda>>,
        >,
{
    pub fn train(&mut self, context: Context) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut state_count = 0;
        let mut q_estimates = HashMap::new();
        for (state, examples) in self.training_examples.by_state() {
            x.extend(state.encode(context.clone(), &self.device).as_vec());
            let mut y_state = [0.0; N_ACTIONS];
            for example in examples {
                let q = if example.is_terminal {
                    0.0
                } else {
                    let q_tensor = q_estimates
                        .entry(example.new_state.clone())
                        .or_insert_with(|| self.evaluate(&example.new_state, context.clone()));
                    *q_tensor.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
                };
                y_state[example.action.encode()] = example.reward.0 + self.future_discount * q;
            }
            y.extend(y_state);
            state_count += 1;
        }
        if !x.is_empty() && !y.is_empty() {
            let x_tensor = self.device.tensor((x, (state_count, Const)));
            let y_tensor = self.device.tensor((y, (state_count, Const)));
            self.network.train(&x_tensor, &y_tensor);
        }

        self.tick += 1;
        if self.tick >= C {
            self.network.update_model();
            self.tick = 0;
        }

        self.update_epsilon();
    }

    pub fn train_supervised(&mut self, examples: &[(State, Vec<(Action, f32)>)], context: Context) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (state, action_values) in examples {
            x.extend(state.encode(context.clone(), &self.device).as_vec());
            let mut y_state = [0.0; N_ACTIONS];
            for (action, value) in action_values {
                y_state[action.encode()] = *value;
            }
            y.extend(y_state);
        }
        let x_tensor = self.device.tensor((x, (examples.len(), Const)));
        let y_tensor = self.device.tensor((y, (examples.len(), Const)));
        self.network.train(&x_tensor, &y_tensor);
        self.network.model = self.network.model_training.clone();
    }

    fn update_epsilon(&mut self) {
        self.epsilon = f32::max(MIN_EPSILON, EPSILON_DECAY * self.epsilon);
    }
}

#[derive(Debug, Clone)]
struct NeuralNetwork<
    Model: BuildOnDevice<Cuda, f32>,
    const N_FEATURES: usize,
    const N_ACTIONS: usize,
> where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug + Clone,
{
    model: <Model as BuildOnDevice<Cuda, f32>>::Built,
    model_training: <Model as BuildOnDevice<Cuda, f32>>::Built,
    optimiser: Adam<<Model as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
    train_steps: usize,
    tau: f32,
}

impl<Model: BuildOnDevice<Cuda, f32>, const N_FEATURES: usize, const N_ACTIONS: usize>
    NeuralNetwork<Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + Module<Tensor<Rank1<N_FEATURES>, f32, Cuda>, Output = Tensor<Rank1<N_ACTIONS>, f32, Cuda>>,
{
    pub fn new(device: &AutoDevice, train_steps: usize) -> Self {
        let model = device.build_module::<Model, f32>();
        let model_training = model.clone();
        let optimiser = Adam::new(
            &model,
            AdamConfig {
                weight_decay: Some(WeightDecay::L2(1e-1)),
                ..Default::default()
            },
        );
        Self {
            model,
            model_training,
            optimiser,
            train_steps,
            tau: TAU_INIT,
        }
    }

    pub fn load<'a>(
        path: &'a str,
        device: &AutoDevice,
        train_steps: usize,
    ) -> std::io::Result<Self> {
        let mut model: <Model as BuildOnDevice<Cuda, f32>>::Built =
            device.build_module::<Model, f32>();
        model
            .load(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let model_training = model.clone();
        let optimiser = Adam::new(
            &model,
            AdamConfig {
                weight_decay: Some(WeightDecay::L2(1e-1)),
                ..Default::default()
            },
        );
        Ok(Self {
            model,
            model_training,
            optimiser,
            train_steps,
            tau: TAU_INIT,
        })
    }

    fn evaluate(
        &self,
        input: &Tensor<Rank1<N_FEATURES>, f32, Cuda>,
    ) -> Tensor<Rank1<N_ACTIONS>, f32, Cuda> {
        self.model.forward(input.clone())
    }

    fn evaluate_training(
        &self,
        input: &Tensor<Rank1<N_FEATURES>, f32, Cuda>,
    ) -> Tensor<Rank1<N_ACTIONS>, f32, Cuda> {
        self.model_training.forward(input.clone())
    }
}

impl<Model: BuildOnDevice<Cuda, f32>, const N_FEATURES: usize, const N_ACTIONS: usize>
    NeuralNetwork<Model, N_FEATURES, N_ACTIONS>
where
    <Model as BuildOnDevice<Cuda, f32>>::Built: Debug
        + Clone
        + ModuleMut<
            Tensor<(usize, Const<N_FEATURES>), f32, Cuda, OwnedTape<f32, Cuda>>,
            Output = Tensor<(usize, Const<N_ACTIONS>), f32, Cuda, OwnedTape<f32, Cuda>>,
        > + TensorCollection<f32, Cuda>,
{
    fn train(
        &mut self,
        input: &Tensor<(usize, Const<N_FEATURES>), f32, Cuda>,
        output: &Tensor<(usize, Const<N_ACTIONS>), f32, Cuda>,
    ) {
        let mut gradients = self.model_training.alloc_grads();
        for _ in 0..self.train_steps {
            let q_predicted = self.model_training.forward_mut(input.trace(gradients));
            let loss = mse_loss(q_predicted, output.clone());
            gradients = loss.backward();
            self.optimiser
                .update(&mut self.model_training, &gradients)
                .expect("Unused parameters found");
            self.model_training.zero_grads(&mut gradients);
        }
    }

    fn update_model(&mut self) {
        self.model.ema(&self.model_training, 1.0 - self.tau);
        self.tau *= TAU_DECAY;
    }
}

#[derive(Debug, Clone)]
struct TrainingExamples<State: Hash + PartialEq + Eq + Clone, Action> {
    data: VecDeque<TrainingExample<State, Action>>,
    capacity: usize,
}

impl<State: Hash + PartialEq + Eq + Clone, Action> TrainingExamples<State, Action> {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::new(),
            capacity,
        }
    }

    fn add(
        &mut self,
        state: State,
        action: Action,
        reward: Reward,
        new_state: State,
        is_terminal: bool,
    ) {
        self.data.push_back(TrainingExample {
            state,
            action,
            reward,
            new_state,
            is_terminal,
        });
        while self.data.len() >= self.capacity {
            self.data.pop_front();
        }
    }

    fn by_state(&self) -> impl IntoIterator<Item = (State, Vec<&TrainingExample<State, Action>>)> {
        self.data
            .iter()
            .map(|e| (e.state.clone(), e))
            .collect::<MultiMap<_, _>>()
    }
}

#[derive(Debug, Clone)]
struct TrainingExample<State, Action> {
    state: State,
    action: Action,
    reward: Reward,
    new_state: State,
    is_terminal: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Reward(pub f32);
