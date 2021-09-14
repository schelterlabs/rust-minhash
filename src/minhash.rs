use crate::create_rng;
use crate::error::MinHashingError;
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::Rng;
use std::cmp::min;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const _MERSENNE_PRIME: u64 = (1 << 61) - 1;
const _MAX_HASH: u64 = (1 << 32) - 1;

type Result<T> = std::result::Result<T, MinHashingError>;

#[derive(Clone, Debug)]
pub struct HashValues(pub Vec<u64>);

#[derive(Clone)]
pub struct MinHash {
    seed: Option<u64>,
    num_perm: usize,
    pub hash_values: HashValues,
    permutations: Vec<(u64, u64)>,
}

impl MinHash {
    pub fn new(num_perm: usize, seed: Option<u64>) -> MinHash {
        let hash_values = Self::init_hash_values(num_perm);
        let permutations = Self::init_permutations(num_perm, seed);
        MinHash {
            seed,
            num_perm,
            hash_values,
            permutations,
        }
    }

    fn init_hash_values(num_perm: usize) -> HashValues {
        let vec = vec![_MAX_HASH; num_perm];
        HashValues(vec)
    }

    fn init_permutations(num_perm: usize, seed: Option<u64>) -> Vec<(u64, u64)> {
        let rng = create_rng(seed);
        let distribution = Uniform::new(0, _MAX_HASH);
        rng.sample_iter(distribution)
            .take(num_perm * 2)
            .tuples()
            .collect_vec()
    }

    pub fn update<T: Hash>(&mut self, value_to_be_hashed: &T) {
        let mut hasher = DefaultHasher::new();
        value_to_be_hashed.hash(&mut hasher);
        let hash_value = hasher.finish() as u32 as u64;
        // TODO: Is there a better way to get u32 hashes?
        let hash_value_permutations = self
            .permutations
            .iter()
            .map(|(a, b)| (((a * hash_value) + b) % _MERSENNE_PRIME) & _MAX_HASH);
        // np.min
        self.hash_values
            .0
            .iter_mut()
            .zip_eq(hash_value_permutations)
            .for_each(|(old, new)| *old = min(*old, new));
    }

    pub fn jaccard(&mut self, other_minhash: &MinHash) -> Result<f32> {
        if other_minhash.seed != self.seed {
            return Err(MinHashingError::DifferentSeeds);
        }
        if other_minhash.num_perm != self.num_perm {
            return Err(MinHashingError::DifferentNumPermFuncs);
        }
        let matches = self
            .hash_values
            .0
            .iter_mut()
            .zip_eq(&other_minhash.hash_values.0)
            .filter(|(left, right)| left == right)
            .count();
        let result = matches as f32 / self.num_perm as f32;
        Ok(result)
    }

    pub fn update_batch<T: Hash>(&mut self, _value_to_be_hashed: &[T]) {
        unimplemented!("Can be added if we need it");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_init_() {
        let m1 = <MinHash>::new(4, Some(0));
        let m2 = <MinHash>::new(4, Some(0));
        assert_eq!(m1.hash_values.0, m2.hash_values.0);
        assert_eq!(m1.permutations, m2.permutations);
    }

    #[test]
    fn test_update() {
        let mut m1 = <MinHash>::new(4, Some(1));
        let m2 = <MinHash>::new(4, Some(1));
        m1.update(&12);
        for i in 0..4 {
            assert!(m1.hash_values.0[i] < m2.hash_values.0[i]);
        }
    }

    #[test]
    fn test_jaccard() -> Result<()> {
        let mut m1 = <MinHash>::new(4, Some(1));
        let mut m2 = <MinHash>::new(4, Some(1));
        assert_eq!(m1.jaccard(&m2)?, 1.0);
        m2.update(&12);
        assert_eq!(m1.jaccard(&m2)?, 0.0);
        m1.update(&13);
        assert!(m1.jaccard(&m2)? < 1.0);
        m1.update(&12);
        let distance = m1.jaccard(&m2)?;
        assert!(distance < 1.0 && distance > 0.0);
        Ok(())
    }

    #[test]
    fn test_data_sketch_minhash() {
        // A test similar to the one in lsh_rs_minhash
        let n_projections = 3;
        let mut m = <MinHash>::new(n_projections, Some(0));
        m.update(&0);
        m.update(&2);
        m.update(&4);
        assert_eq!(m.hash_values.0.len(), n_projections);
        println!("{:?}", &m.hash_values);
    }
}
