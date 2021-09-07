use ndarray::{Array2, Array, Array1, Axis, Zip};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use crate::create_rng;
use ndarray_rand::rand_distr::Uniform;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::ops::{Mul, Add};
use std::cmp::min;

const _MERSENNE_PRIME: u64 = (1 << 61) - 1;
const _MAX_HASH: u64 = (1 << 32) - 1;
const _HASH_RANGE: u32 = u32::MAX;

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHash {
    num_perm: usize,
    hash_values: Array1<u64>,
    permutations: Array2<u64>
}

impl DataSketchMinHash {
    pub fn new(num_perm: usize, seed: Option<u64>) -> DataSketchMinHash {
        let _dim: usize = 0; // TODO
        let hash_values = Self::init_hash_values(num_perm);
        let permutations = Self::init_permutations(num_perm, seed);
        DataSketchMinHash {
            num_perm,
            hash_values,
            permutations
        }
    }

    fn init_hash_values(num_perm: usize) -> Array1<u64> {
        Array1::from_elem(num_perm, _MAX_HASH)
    }

    fn init_permutations(num_perm: usize, seed: Option<u64>) -> Array2<u64> {
        let mut rng = create_rng(seed);
        let distribution = Uniform::new(0, _MERSENNE_PRIME);
        Array::random_using((num_perm, 2), distribution, &mut rng)
    }

    fn _update<T: Hash>(&mut self, value_to_be_hashed: &T){
        let mut hasher = DefaultHasher::new();
        value_to_be_hashed.hash(&mut hasher);
        let hash_value = hasher.finish();
        let a = self.permutations.index_axis(Axis(0), 0);
        let b = self.permutations.index_axis(Axis(0), 1);
        let hash_value_permutations = hash_value.mul(&a).add(&b) % _MERSENNE_PRIME;
        // np.min
        Zip::from(&mut self.hash_values).and(&hash_value_permutations)
            .apply(|left, &right| {
                *left = min(*left, right);
            });
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_init_() {
        let m1 = <DataSketchMinHash>::new(4, Some(0));
        let m2 = <DataSketchMinHash>::new(4, Some(0));
        assert_eq!(m1.hash_values, m2.hash_values);
        assert_eq!(m1.permutations, m2.permutations);
    }

    #[test]
    fn test_data_sketch_minhash() {
        // TODO: This is the old test
        let n_projections = 3;
        let _h = <DataSketchMinHash>::new(n_projections, Some(0));
        // let hash = h.hash_vec(&[1, 0, 1, 0, 1]);
        // assert_eq!(hash.len(), n_projections);
        // println!("{:?}", hash);
    }
}
