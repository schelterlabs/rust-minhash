use crate::error::MinHashingError;
use float_cmp::ApproxEq;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights(f32, f32);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinHashLshParams {
    pub b: usize,
    pub r: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHashLsh {
    num_perm: usize,
    threshold: f32,
    weights: Weights,
    buffer_size: usize,
    params: MinHashLshParams,
}

type Result<T> = std::result::Result<T, MinHashingError>;

impl DataSketchMinHashLsh {
    pub fn new(
        num_perm: usize,
        weights: Option<Weights>,
        threshold: Option<f32>,
    ) -> Result<DataSketchMinHashLsh> {
        let threshold = match threshold {
            Some(threshold) if !(0.0..=1.0).contains(&threshold) => {
                return Err(MinHashingError::WrongThresholdInterval);
            }
            Some(threshold) => threshold,
            _ => 0.9,
        };
        if num_perm < 2 {
            return Err(MinHashingError::NumPermFuncsTooLow);
        }

        let weights = match weights {
            Some(weights) => {
                let Weights(left, right) = weights;
                if !(0.0..=1.0).contains(&left) || !(0.0..=1.0).contains(&right) {
                    return Err(MinHashingError::WrongWeightThreshold);
                }
                let sum_weights = left + right;
                if !sum_weights.approx_eq(1.0, (0.0, 2)) {
                    return Err(MinHashingError::UnexpectedSumWeight);
                }
                weights
            }
            _ => Weights(0.5, 0.5),
        };
        let params = DataSketchMinHashLsh::find_optimal_params(threshold, num_perm, &weights);

        Ok(DataSketchMinHashLsh {
            num_perm,
            threshold,
            weights,
            buffer_size: 50_000,
            params,
        })
    }

    fn find_optimal_params(threshold: f32, num_perm: usize, weights: &Weights) -> MinHashLshParams {
        let Weights(false_positive_weight, false_negative_weight) = weights;
        let mut min_error = f32::infinity();
        let mut opt = MinHashLshParams { b: 0, r: 0 };
        for b in 1..num_perm + 1 {
            let max_r = num_perm / b;
            for r in 1..max_r + 1 {
                let false_pos: f32 =
                    DataSketchMinHashLsh::false_positive_probability(threshold, b, r);
                let false_neg: f32 =
                    DataSketchMinHashLsh::false_negative_probability(threshold, b, r);
                let error = false_pos * false_positive_weight + false_neg * false_negative_weight;
                if error < min_error {
                    min_error = error;
                    opt = MinHashLshParams { b, r };
                }
            }
        }
        opt
    }

    fn false_positive_probability(_threshold: f32, _b: usize, _r: usize) -> f32 {
        unimplemented!("TODO")
    }

    fn false_negative_probability(_threshold: f32, _b: usize, _r: usize) -> f32 {
        unimplemented!("TODO")
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
