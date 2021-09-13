use rand::prelude::SmallRng;
use rand::{thread_rng, SeedableRng};

pub mod datasketch_minhash;
pub mod datasketch_minhash_lsh;
mod error;
pub mod lsh_rs_minhash;

pub fn create_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        _ => match SmallRng::from_rng(thread_rng()) {
            Ok(rng) => rng,
            Err(_) => SmallRng::from_entropy(),
        },
    }
}
