use ndarray::{Array2, Array, Array1, Axis, Zip};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use crate::create_rng;
use ndarray_rand::rand_distr::Uniform;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::ops::{Mul, Add};
use std::cmp::min;
use crate::error::MinHashingError;

const _MERSENNE_PRIME: u64 = (1 << 61) - 1;
const _MAX_HASH: u64 = (1 << 32) - 1;

type Result<T> = std::result::Result<T, MinHashingError>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HashValues(pub Array1<u64>);

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHash {
    seed: Option<u64>,
    num_perm: usize,
    pub hash_values: HashValues,
    permutations: Array2<u64>
}

impl DataSketchMinHash {
    pub fn new(num_perm: usize, seed: Option<u64>) -> DataSketchMinHash {
        let hash_values = Self::init_hash_values(num_perm);
        let permutations = Self::init_permutations(num_perm, seed);
        DataSketchMinHash {
            seed,
            num_perm,
            hash_values,
            permutations
        }
    }

    fn init_hash_values(num_perm: usize) -> HashValues {
        HashValues(Array1::from_elem(num_perm, _MAX_HASH))
    }

    fn init_permutations(num_perm: usize, seed: Option<u64>) -> Array2<u64> {
        let mut rng = create_rng(seed);
        let distribution = Uniform::new(0, _MAX_HASH);
        Array::random_using((num_perm, 2), distribution, &mut rng)
    }

    pub fn update<T: Hash>(&mut self, value_to_be_hashed: &T){
        let mut hasher = DefaultHasher::new();
        value_to_be_hashed.hash(&mut hasher);
        let hash_value = hasher.finish() as u32 as u64;
        // TODO: Is there a better way to get u32 hashes?
        let a = self.permutations.index_axis(Axis(1), 0);
        let b = self.permutations.index_axis(Axis(1), 1);
        let hash_value_permutations = (hash_value.mul(&a).add(&b) % _MERSENNE_PRIME) & _MAX_HASH;
        // np.min
        Zip::from(&mut self.hash_values.0).and(&hash_value_permutations)
            .apply(|left, &right| {
                *left = min(*left, right);
            });
    }

    pub fn jaccard(&mut self, other_minhash: &DataSketchMinHash) -> Result<f32>{
        if other_minhash.seed != self.seed{
            return Err(MinHashingError::DifferentSeeds);
        }
        if other_minhash.num_perm != self.num_perm{
            return Err(MinHashingError::DifferentNumPermFuncs);
        }
        let mut matches: usize = 0;
        Zip::from(&self.hash_values.0).and(&other_minhash.hash_values.0)
            .apply(|&left, &right| {
                matches += (left == right) as usize;
            });
        let result = matches as f32 / self.num_perm as f32;
        Ok(result)
    }

    pub fn update_batch<T: Hash>(&mut self, _value_to_be_hashed: &[T]){
        unimplemented!("Can be added if we need it");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_init_() {
        let m1 = <DataSketchMinHash>::new(4, Some(0));
        let m2 = <DataSketchMinHash>::new(4, Some(0));
        assert_eq!(m1.hash_values.0, m2.hash_values.0);
        assert_eq!(m1.permutations, m2.permutations);
    }

    #[test]
    fn test_update() {
        let mut m1 = <DataSketchMinHash>::new(4, Some(1));
        let m2 = <DataSketchMinHash>::new(4, Some(1));
        m1.update(&12);
        for i in 0..4 {
            assert!(m1.hash_values.0[i] < m2.hash_values.0[i]);
        }
    }

    #[test]
    fn test_jaccard() -> Result<()>{
        let mut m1 = <DataSketchMinHash>::new(4, Some(1));
        let mut m2 = <DataSketchMinHash>::new(4, Some(1));
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
        let mut m = <DataSketchMinHash>::new(n_projections, Some(0));
        m.update(&0);
        m.update(&2);
        m.update(&4);
        assert_eq!(m.hash_values.0.len(), n_projections);
        println!("{:?}", &m.hash_values);
    }
}
