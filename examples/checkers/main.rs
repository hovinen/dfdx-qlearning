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
        let (old_row, old_col) = (
            action.piece_position.0 as usize,
            action.piece_position.1 as usize,
        );
        let (new_row, new_col) = match self.0[old_row][old_col] {
            PositionState::OccupiedMan(cell_player) if cell_player == player => {
                let row_disp = match player {
                    Player::White => -1,
                    Player::Black => 1,
                };
                let new_row = action.piece_position.0 as i32 + row_disp;
                let col_disp = match action.direction {
                    Direction::ForwardLeft => -1,
                    Direction::ForwardRight => 1,
                    Direction::BackwardLeft | Direction::BackwardRight => {
                        return false;
                    }
                };
                let new_col = action.piece_position.1 as i32 + col_disp;
                (new_row, new_col)
            }
            PositionState::OccupiedKing(cell_player) if cell_player == player => {
                let row_disp = match (player, action.direction) {
                    (Player::White, Direction::ForwardLeft)
                    | (Player::White, Direction::ForwardRight)
                    | (Player::Black, Direction::BackwardLeft)
                    | (Player::Black, Direction::BackwardRight) => -1,
                    _ => 1,
                };
                let new_row = action.piece_position.0 as i32 + row_disp;
                let col_disp = match action.direction {
                    Direction::ForwardLeft | Direction::BackwardLeft => -1,
                    Direction::ForwardRight | Direction::BackwardRight => 1,
                };
                let new_col = action.piece_position.1 as i32 + col_disp;
                (new_row, new_col)
            }
            _ => {
                // Own piece not at the indicated position
                return false;
            }
        };
        if new_row < 0 || new_row >= BOARD_SIZE as i32 {
            // Out of bounds by row
            return false;
        }
        if new_col < 0 || new_col >= BOARD_SIZE as i32 {
            // Out of bounds by column
            return false;
        }
        let (new_row, new_col) = (new_row as usize, new_col as usize);
        match self.0[new_row][new_col] {
            PositionState::OccupiedMan(cell_player) | PositionState::OccupiedKing(cell_player)
                if cell_player != player =>
            {
                todo!("Capturing");
            }
            PositionState::Vacant => {
                self.0[new_row][new_col] = self.0[old_row][old_col];
                self.0[old_row][old_col] = PositionState::Vacant;
            }
            _ => {
                // Blocked by own piece
                return false;
            }
        }
        true
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
