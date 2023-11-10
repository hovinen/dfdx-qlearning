use std::fmt::Display;

pub trait GameLogger<Player, State> {
    fn log_state(&self, player: &Player, state: &State);

    fn log_wins(&self, player: &Player, count: usize, game_count: usize);

    fn log_draws(&self, count: usize, game_count: usize);
}

pub struct TrivialGameLogger;

impl<Player, State> GameLogger<Player, State> for TrivialGameLogger {
    fn log_state(&self, _player: &Player, _state: &State) {}

    fn log_wins(&self, _player: &Player, _count: usize, _game_count: usize) {}

    fn log_draws(&self, _count: usize, _game_count: usize) {}
}

pub struct DisplayGameLogger;

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
