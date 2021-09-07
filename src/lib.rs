use rand::prelude::SmallRng;
use rand::{SeedableRng, thread_rng};

pub mod lsh_rs_minhash;
pub mod datasketch_minhash;

const _MERSENNE_PRIME: u64 = (1 << 61) - 1;
const _MAX_HASH: u64 = (1 << 32) - 1;
const _HASH_RANGE: u32 = u32::MAX;

pub fn create_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        _ => {
            match SmallRng::from_rng(thread_rng()) {
                Ok(rng) => rng,
                Err(_) => SmallRng::from_entropy(),
            }
        }
    }
}
