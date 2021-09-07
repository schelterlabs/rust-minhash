use ndarray::Array2;
use std::marker::PhantomData;
use serde::{Deserialize, Serialize};

// A hash family for the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHash<N = u8, K = i32> {
    pub pi: Array2<N>,
    n_projections: usize,
    phantom: PhantomData<K>,
}
