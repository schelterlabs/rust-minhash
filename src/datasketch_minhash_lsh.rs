use crate::datasketch_minhash::DataSketchMinHash;
use crate::error::MinHashingError;
use float_cmp::ApproxEq;
use ndarray::{Array1, Axis, Slice};
use num::traits::Pow;
use num::Float;
use quadrature::integrate;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

const _ALLOWED_INTEGRATE_ERR: f64 = 0.001;

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights(f64, f64);

#[derive(Serialize, Deserialize, Clone, Debug, Hash, PartialEq, Eq)]
pub struct HashValuePart(pub Array1<u64>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinHashLshParams {
    pub b: usize,
    pub r: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHashLsh<KeyType: Eq + Hash + Clone> {
    num_perm: usize,
    threshold: f64,
    weights: Weights,
    buffer_size: usize,
    params: MinHashLshParams,
    hash_tables: Vec<HashMap<HashValuePart, HashSet<KeyType>>>,
    hash_ranges: Vec<(usize, usize)>,
    keys: HashMap<KeyType, Vec<HashValuePart>>,
}

type Result<T> = std::result::Result<T, MinHashingError>;

impl<KeyType: Eq + Hash + Clone> DataSketchMinHashLsh<KeyType> {
    pub fn new(
        num_perm: usize,
        weights: Option<Weights>,
        threshold: Option<f64>,
    ) -> Result<DataSketchMinHashLsh<KeyType>> {
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
        let params =
            DataSketchMinHashLsh::<KeyType>::find_optimal_params(threshold, num_perm, &weights);

        let hash_tables = (0..params.b).into_iter().map(|_| HashMap::new()).collect();
        let hash_ranges = (0..params.b)
            .into_iter()
            .map(|i| (i * params.r, (i + 1) * params.r))
            .collect();
        Ok(DataSketchMinHashLsh {
            num_perm,
            threshold,
            weights,
            buffer_size: 50_000,
            params,
            hash_tables,
            hash_ranges,
            keys: HashMap::new(),
        })
    }

    fn find_optimal_params(threshold: f64, num_perm: usize, weights: &Weights) -> MinHashLshParams {
        let Weights(false_positive_weight, false_negative_weight) = weights;
        let mut min_error = f64::infinity();
        let mut opt = MinHashLshParams { b: 0, r: 0 };
        for b in 1..num_perm + 1 {
            let max_r = num_perm / b;
            for r in 1..max_r + 1 {
                let false_pos =
                    DataSketchMinHashLsh::<KeyType>::false_positive_probability(threshold, b, r);
                let false_neg =
                    DataSketchMinHashLsh::<KeyType>::false_negative_probability(threshold, b, r);
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

    pub fn is_empty(&self) -> bool {
        self.hash_tables.iter().any(|table| table.len() == 0)
    }

    pub fn insert(&mut self, key: KeyType, min_hash: &DataSketchMinHash) -> Result<()> {
        // TODO: We could also add optional checks whether the key is already present in index
        // TODO: Why has the original implementation buffer params everywhere
        if min_hash.hash_values.0.len() != self.num_perm {
            return Err(MinHashingError::DifferentNumPermFuncs);
        }
        let mut hash_value_parts: Vec<HashValuePart> = self
            .hash_ranges
            .iter()
            .map(|(start, end)| {
                let hash_part = min_hash
                    .hash_values
                    .0
                    .slice_axis(Axis(0), Slice::from(*start..*end))
                    .to_owned();
                HashValuePart(hash_part)
            })
            .collect();
        self.keys.insert(key.clone(), hash_value_parts.clone());
        let hash_table_iter = &mut self.hash_tables.iter_mut();
        let zipped_drain_iter = hash_value_parts.drain(..).zip(hash_table_iter);
        for (hash_part, hash_table) in zipped_drain_iter {
            hash_table
                .entry(hash_part)
                .or_insert_with(HashSet::new)
                .insert(key.clone());
        }
        Ok(())
    }

    pub fn contains_key(&self, key: &KeyType) -> bool {
        self.keys.contains_key(key)
    }

    pub fn query(&mut self, _query_value: &DataSketchMinHash) {
        unimplemented!("TODO");
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::datasketch_minhash::DataSketchMinHash;

    #[test]
    fn test_init() -> Result<()> {
        let lsh = <DataSketchMinHashLsh<&str>>::new(128, None, Some(0.8))?;
        assert!(lsh.is_empty());
        let MinHashLshParams { b: b1, r: r1 } = lsh.params;
        let lsh = <DataSketchMinHashLsh<&str>>::new(128, Some(Weights(0.2, 0.8)), Some(0.8))?;
        let MinHashLshParams { b: b2, r: r2 } = lsh.params;
        assert!(b1 < b2);
        assert!(r1 > r2);
        Ok(())
    }

    #[test]
    /// Check _H output consistent bytes length given the same concatenated hash value size
    fn test_byteswap() -> Result<()> {
        for _ in (2..128 + 1).step_by(16) {
            // TODO: I don't understand yet why we need this
            let mut lsh = <DataSketchMinHashLsh<&str>>::new(128, None, None)?;
            let mut m = <DataSketchMinHash>::new(128, None);
            m.update(&"abcdefg");
            m.update(&"1234567");
            lsh.insert(&"m", &m)?;
            let sizes = lsh
                .hash_tables
                .iter()
                .flat_map(|table| table.values().map(|set| set.len()))
                .collect::<Vec<_>>();
            sizes.iter().for_each(|size| assert_eq!(*size, sizes[0]));
        }
        Ok(())
    }

    #[test]
    fn example_eg1() -> Result<()> {
        let set1: HashSet<&'static str> = [
            "minhash",
            "is",
            "a",
            "probabilistic",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "datasets",
        ]
        .iter()
        .cloned()
        .collect();
        let set2: HashSet<&'static str> = [
            "minhash",
            "is",
            "a",
            "probability",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "documents",
        ]
        .iter()
        .cloned()
        .collect();
        let set3: HashSet<&'static str> = [
            "minhash",
            "is",
            "probability",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "documents",
        ]
        .iter()
        .cloned()
        .collect();

        let n_projections = 128;
        let mut m1 = <DataSketchMinHash>::new(n_projections, Some(0));
        let mut m2 = <DataSketchMinHash>::new(n_projections, Some(0));
        let mut m3 = <DataSketchMinHash>::new(n_projections, Some(0));
        for d in set1 {
            m1.update(&d);
        }
        for d in set2 {
            m2.update(&d);
        }
        for d in set3 {
            m3.update(&d);
        }

        // Create LSHindex
        let mut lsh = <DataSketchMinHashLsh<&str>>::new(128, None, Some(0.5))?;
        lsh.insert(&"m2", &m2)?;
        lsh.insert(&"m3", &m3)?;
        let result = lsh.query(&m1);
        println!(
            "Approximate neighbours with Jaccard similarity > 0.5: {:?}",
            result
        );
        Ok(())
    }
}
