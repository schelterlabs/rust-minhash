use crate::error::MinHashingError;
use float_cmp::ApproxEq;
use num::traits::Pow;
use num::Float;
use quadrature::integrate;
use serde::{Deserialize, Serialize};

const _ALLOWED_INTEGRATE_ERR: f64 = 0.001;

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights(f64, f64);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinHashLshParams {
    pub b: usize,
    pub r: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHashLsh {
    num_perm: usize,
    threshold: f64,
    weights: Weights,
    buffer_size: usize,
    params: MinHashLshParams,
}

type Result<T> = std::result::Result<T, MinHashingError>;

impl DataSketchMinHashLsh {
    pub fn new(
        num_perm: usize,
        weights: Option<Weights>,
        threshold: Option<f64>,
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

    fn find_optimal_params(threshold: f64, num_perm: usize, weights: &Weights) -> MinHashLshParams {
        let Weights(false_positive_weight, false_negative_weight) = weights;
        let mut min_error = f64::infinity();
        let mut opt = MinHashLshParams { b: 0, r: 0 };
        for b in 1..num_perm + 1 {
            let max_r = num_perm / b;
            for r in 1..max_r + 1 {
                let false_pos = DataSketchMinHashLsh::false_positive_probability(threshold, b, r);
                let false_neg = DataSketchMinHashLsh::false_negative_probability(threshold, b, r);
                let error = false_pos * false_positive_weight + false_neg * false_negative_weight;
                if error < min_error {
                    min_error = error;
                    opt = MinHashLshParams { b, r };
                }
            }
        }
        opt
    }

    fn false_positive_probability(threshold: f64, b: usize, r: usize) -> f64 {
        let b = b as f64;
        let r = r as f64;
        let _probability = |s| -> f64 { 1. - f64::pow(1. - f64::pow(s, r), b) };
        integrate(_probability, 0.0, threshold, _ALLOWED_INTEGRATE_ERR).integral
    }

    fn false_negative_probability(threshold: f64, b: usize, r: usize) -> f64 {
        let b = b as f64;
        let r = r as f64;
        let _probability = |s| -> f64 { 1. - (1. - f64::pow(1. - f64::pow(s, r), b)) };
        integrate(_probability, threshold, 1.0, _ALLOWED_INTEGRATE_ERR).integral
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
