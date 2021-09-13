use serde::{Deserialize, Serialize};
use crate::error::MinHashingError;
use float_cmp::ApproxEq;

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights(f32, f32);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinHashLshParams{
    pub b: f32,
    pub r: f32
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHashLsh {
    num_perm: usize,
    threshold: f32,
    weights: Weights,
    buffer_size: usize,
    params: MinHashLshParams
}

type Result<T> = std::result::Result<T, MinHashingError>;

impl DataSketchMinHashLsh {
    pub fn new(num_perm: usize, weights: Option<Weights>, threshold: Option<f32>) -> Result<DataSketchMinHashLsh> {
        if let Some(threshold) = threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(MinHashingError::WrongThresholdInterval);
            }
        }

        if num_perm < 2 {
            return Err(MinHashingError::NumPermFuncsTooLow);
        }
        if let Some(weights) = weights {
            let Weights(left, right) = weights;
            if !(0.0..=1.0).contains(&left) || !(0.0..=1.0).contains(&right) {
                return Err(MinHashingError::WrongWeightThreshold);
            }
            let sum_weights = left + right;
            if !sum_weights.approx_eq(1.0, (0.0, 2)) {
                return Err(MinHashingError::UnexpectedSumWeight);
            }
        }
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_init_() -> Result<()> {
        let _minhash_lsh = <DataSketchMinHashLsh>::new(4, None, None)?;
        Ok(())
    }
}