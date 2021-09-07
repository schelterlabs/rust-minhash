use ndarray::prelude::*;
use num::{traits::NumCast, Zero, ToPrimitive};
use std::marker::PhantomData;
use serde::{Deserialize, Serialize};

use num::{FromPrimitive};
use std::cmp::{Ord};
use std::hash::Hash;

use ndarray::{LinalgScalar, ScalarOperand};
use std::ops::AddAssign;
use std::fmt::{Debug, Display};
use crate::create_rng;

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

/// Implement this trait to create your own custom hashers.
/// In case of a symmetrical hash function, only `hash_vec_query` needs to be implemented.
pub trait VecHash<N, K> {
    /// Create a hash for a query data point.
    fn hash_vec(&self, v: &[N]) -> Vec<K>;
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
    pub fn new(n_projections: usize, dim: usize, seed: Option<u64>) -> Self {
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
    fn hash_vec(&self, v: &[N]) -> Vec<K> {
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
        let h = <MinHash>::new(n_projections, 5, Some(0));
        let hash = h.hash_vec(&[1, 0, 1, 0, 1]);
        assert_eq!(hash.len(), n_projections);
        println!("{:?}", hash);
    }
}
