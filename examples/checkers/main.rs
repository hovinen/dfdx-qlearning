use clap::{Parser, Subcommand};
use dfdx::prelude::{Device, Rank1, Tensor};
use dfdx_qlearning::{
    abstract_model::{EncodableAction, EncodableState, Reward},
    actor::{ActorState, EnumerablePlayer},
};

const BOARD_SIZE: usize = 8;
const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            eprintln!("Usage: checkers play|train");
        }
        Some(Command::Train) => todo!(),
        Some(Command::Play) => todo!(),
    };
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Train,
    Play,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Player {
    White,
    Black,
}

impl EnumerablePlayer for Player {
    fn all_players() -> Vec<Self> {
        vec![Player::White, Player::Black]
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
enum PositionState {
    #[default]
    Vacant,
    OccupiedMan(Player),
    OccupiedKing(Player),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct State([[PositionState; BOARD_SIZE]; BOARD_SIZE]);

impl State {
    fn new_initial() -> Self {
        let mut positions = [[PositionState::default(); BOARD_SIZE]; BOARD_SIZE];
        for row in 0..3 {
            for j in 0..4 {
                positions[row][j * 2 + row % 2] = PositionState::OccupiedMan(Player::Black);
                positions[BOARD_SIZE - 1 - row][j * 2 + (row + 1) % 2] =
                    PositionState::OccupiedMan(Player::White);
            }
        }
        Self(positions)
    }
}

impl ActorState<Action, Player> for State {
    fn available_actions(&self, player: Player) -> Vec<Action> {
        todo!()
    }

    fn apply_action(&mut self, action: Action, player: Player) -> bool {
        match self.0[action.piece_position.0 as usize][action.piece_position.1 as usize] {
            PositionState::OccupiedMan(cell_player) if cell_player == player => {
                match action.direction {
                    Direction::ForwardLeft => {
                        let row_disp = match player {
                            Player::White => -1,
                            Player::Black => 1,
                        };
                        let new_row = action.piece_position.0 as i32 + row_disp;
                        if new_row < 0 || new_row >= BOARD_SIZE as i32 {
                            false
                        } else {
                            todo!()
                        }
                    }
                    Direction::ForwardRight => todo!(),
                    Direction::BackwardLeft | Direction::BackwardRight => false,
                }
            }
            PositionState::OccupiedKing(cell_player) if cell_player == player => {
                todo!()
            }
            _ => false,
        }
    }

    fn is_game_over(&self) -> bool {
        self.available_actions(Player::White).is_empty()
            || self.available_actions(Player::Black).is_empty()
    }

    fn who_won(&self) -> Option<Player> {
        match (
            self.available_actions(Player::White).is_empty(),
            self.available_actions(Player::Black).is_empty(),
        ) {
            (true, true) | (false, false) => None,
            (true, false) => Some(Player::Black),
            (false, true) => Some(Player::White),
        }
    }

    fn reward(&self, player: &Player) -> Reward {
        todo!()
    }
}

impl EncodableState<BOARD_CELLS, Player> for State {
    fn encode<D: Device<f32>>(
        &self,
        context: Player,
        device: &D,
    ) -> Tensor<Rank1<BOARD_CELLS>, f32, D> {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        self.is_game_over()
    }
}

struct Action {
    piece_position: (u8, u8),
    direction: Direction,
}

enum Direction {
    ForwardLeft,
    ForwardRight,
    BackwardLeft,
    BackwardRight,
}

impl EncodableAction<Player> for State {
    fn encode(&self, _context: Player) -> usize {
        todo!()
    }

    fn decode(index: usize, _context: Player) -> Self {
        todo!()
    }
}
