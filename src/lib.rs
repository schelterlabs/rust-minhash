use rand::prelude::SmallRng;
use rand::{thread_rng, SeedableRng};

mod error;
mod minhash;
mod minhash_lsh;

pub use crate::minhash::*;
pub use crate::minhash_lsh::*;

pub fn create_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        _ => match SmallRng::from_rng(thread_rng()) {
            Ok(rng) => rng,
            Err(_) => SmallRng::from_entropy(),
        },
    }
}
