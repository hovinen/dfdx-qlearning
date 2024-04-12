mod play;
mod train;

use clap::{Parser, Subcommand};
use dfdx::{
    nn::{builders::Linear, modules::ReLU},
    shapes::Rank1,
    tensor::{Tensor, TensorFrom},
    tensor_ops::Device,
};
use dfdx_qlearning::{
    abstract_model::{EncodableAction, EncodableState, Reward},
    actor::{ActorState, EnumerablePlayer},
};
use play::play;
use std::fmt::Display;
use train::train;

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            eprintln!("Usage: tictactoe play|train");
        }
        Some(Command::Play) => play(),
        Some(Command::Train) => train(),
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

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy, Hash)]
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
    fn available_actions(&self, _player: CellState) -> Vec<TicTacToeAction> {
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

    fn is_terminal(&self) -> bool {
        self.player_won(CellState::X) || self.player_won(CellState::O) || self.draw()
    }
}

#[derive(Clone, Debug)]
struct TicTacToeAction(u8, u8);

impl EncodableAction<CellState> for TicTacToeAction {
    fn encode(&self, _: CellState) -> usize {
        (self.0 * 3 + self.1) as usize
    }

    fn decode(index: usize, _: CellState) -> Self {
        TicTacToeAction((index / 3) as u8, (index % 3) as u8)
    }
}

type TicTacToeNetwork = (Linear<9, 32>, ReLU, Linear<32, 32>, ReLU, Linear<32, 9>);
