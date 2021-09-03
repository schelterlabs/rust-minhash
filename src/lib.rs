use ndarray::prelude::*;
use num::{traits::NumCast, Zero, ToPrimitive};
use std::marker::PhantomData;
use serde::{Deserialize, Serialize};

use num::{FromPrimitive};
use std::cmp::{Ord};
use std::hash::Hash;

use rand::rngs::SmallRng;
use rand::{thread_rng, SeedableRng};
use ndarray::{LinalgScalar, ScalarOperand};
use std::ops::AddAssign;
use std::fmt::{Debug, Display};
use std::error;

pub trait Numeric:
LinalgScalar
+ ScalarOperand
+ NumCast
+ ToPrimitive
+ Send
+ Sync
+ PartialEq
+ PartialOrd
+ FromPrimitive
+ AddAssign
+ Serialize
+ Debug
+ Display
{
}

impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}

pub trait Integer: Numeric + Ord + Eq + Hash {}
impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}
impl Integer for u64 {}

impl Integer for i8 {}
impl Integer for i16 {}
impl Integer for i32 {}
impl Integer for i64 {}

const _MERSENNE_PRIME: u64 = (1 << 61) - 1;
const _MAX_HASH: u64 = (1 << 32) - 1;
const _HASH_RANGE: u64 = 1 << 32;

// Change the alias to `Box<error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

pub trait QueryDirectedProbe<N, K> {
    fn query_directed_probe(&self, q: &[N], budget: usize) -> Result<Vec<Vec<K>>>;
}

pub trait StepWiseProbe<N, K>: VecHash<N, K> {
    fn step_wise_probe(&self, q: &[N], budget: usize, hash_len: usize) -> Result<Vec<Vec<K>>>;
}

pub fn create_rng(seed: u64) -> SmallRng {
    // TODO: if seed == 0, use random seeded rng
    if seed == 0 {
        match SmallRng::from_rng(thread_rng()) {
            Ok(rng) => rng,
            Err(_) => SmallRng::from_entropy(),
        }
    } else {
        SmallRng::seed_from_u64(seed)
    }
}

/// Implement this trait to create your own custom hashers.
/// In case of a symmetrical hash function, only `hash_vec_query` needs to be implemented.
pub trait VecHash<N, K> {
    /// Create a hash for a query data point.
    fn hash_vec_query(&self, v: &[N]) -> Vec<K>;
    /// Create a hash for a data point that is being stored.
    fn hash_vec_put(&self, v: &[N]) -> Vec<K> {
        self.hash_vec_query(v)
    }

    /// If the hasher implements the QueryDirectedProbe trait it should return Some(self)
    fn as_query_directed_probe(&self) -> Option<&dyn QueryDirectedProbe<N, K>> {
        None
    }
    /// If the hasher implements the StepWiseProbe trait it should return Some(self)
    fn as_step_wise_probe(&self) -> Option<&dyn StepWiseProbe<N, K>> {
        None
    }
}

/// A hash family for the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
/// /// The generic integer N, needs to be able to hold the number of dimensions.
/// so a `u8` with a vector of > 255 dimensions will cause a `panic`.
#[derive(Serialize, Deserialize, Clone)]
pub struct MinHash<N = u8, K = i32> {
    pub pi: Array2<N>,
    n_projections: usize,
    phantom: PhantomData<K>,
}

impl<N, K> MinHash<N, K>
    where
        N: Integer,
        K: Integer,
{
    pub fn new(n_projections: usize, dim: usize, seed: u64) -> Self {
        let mut pi = Array::zeros((n_projections, dim));
        let mut rng = create_rng(seed);

        for row in 0..n_projections {
            // randomly permute the indexes of vector that should be hashed.
            // So a vector of length 4 could have the following random pi permutation:
            // [3, 2, 4, 1]
            // We start counting from 1, as we want to multiply with these pi vectors and take the
            // lowest non zero output
            let permutation_idx = rand::seq::index::sample(&mut rng, dim, dim)
                .into_iter()
                .map(|idx| N::from_usize(idx + 1).expect("could not cast idx to generic"))
                .collect::<Vec<_>>();
            let mut slice = pi.slice_mut(s![row, ..]);
            slice += &aview1(&permutation_idx);
        }
        println!("{:?}", pi);
        MinHash {
            pi,
            n_projections,
            phantom: PhantomData,
        }
    }
}

impl<N, K> VecHash<N, K> for MinHash<N, K>
    where
        N: Integer,
        K: Integer,
{
    fn hash_vec_query(&self, v: &[N]) -> Vec<K> {
        let a = &self.pi * &aview1(v);
        let init = K::from_usize(self.n_projections).expect("could not cast to K");
        let hash = a.map_axis(Axis(1), |view| {
            view.into_iter().fold(init, |acc, v| {
                if *v > Zero::zero() {
                    let v = K::from(*v).expect("could not cast N to K");
                    if v < acc {
                        v
                    } else {
                        acc
                    }
                } else {
                    acc
                }
            })
        });
        hash.to_vec()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_minhash() {
        let n_projections = 3;
        let h = <MinHash>::new(n_projections, 5, 0);
        let hash = h.hash_vec_query(&[1, 0, 1, 0, 1]);
        assert_eq!(hash.len(), n_projections);
        println!("{:?}", hash);
    }
}
