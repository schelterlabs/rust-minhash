use crate::error::MinHashingError;
use crate::minhash::MinHash;
use float_cmp::ApproxEq;
use quadrature::integrate;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

const _ALLOWED_INTEGRATE_ERR: f64 = 0.001;

#[derive(Clone)]
pub struct Weights(pub f64, pub f64);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct HashValuePart(pub Vec<u64>);

#[derive(Clone, Debug)]
pub struct LshParams {
    pub b: usize,
    pub r: usize,
}

impl LshParams {
    pub fn find_optimal_params(threshold: f64, num_perm: usize, weights: &Weights) -> LshParams {
        let Weights(false_positive_weight, false_negative_weight) = weights;
        let mut min_error = f64::INFINITY;
        let mut opt = LshParams { b: 0, r: 0 };
        for b in 1..num_perm + 1 {
            let max_r = num_perm / b;
            for r in 1..max_r + 1 {
                let false_pos = LshParams::false_positive_probability(threshold, b, r);
                let false_neg = LshParams::false_negative_probability(threshold, b, r);
                let error = false_pos * false_positive_weight + false_neg * false_negative_weight;
                if error < min_error {
                    min_error = error;
                    opt = LshParams { b, r };
                }
            }
        }
        opt
    }

    fn false_positive_probability(threshold: f64, b: usize, r: usize) -> f64 {
        let _probability =
            |s| -> f64 { 1. - f64::powf(1. - f64::powi(s, r as i32) as f64, b as f64) };
        integrate(_probability, 0.0, threshold, _ALLOWED_INTEGRATE_ERR).integral
    }

    fn false_negative_probability(threshold: f64, b: usize, r: usize) -> f64 {
        let _probability =
            |s| -> f64 { 1. - (1. - f64::powf(1. - f64::powi(s, r as i32), b as f64)) };
        integrate(_probability, threshold, 1.0, _ALLOWED_INTEGRATE_ERR).integral
    }
}

#[derive(Clone)]
pub struct MinHashLsh<KeyType: Eq + Hash + Clone> {
    num_perm: usize,
    threshold: f64,
    weights: Weights,
    buffer_size: usize,
    params: LshParams,
    hash_tables: Vec<HashMap<HashValuePart, HashSet<KeyType>>>,
    hash_ranges: Vec<(usize, usize)>,
    keys: HashMap<KeyType, Vec<HashValuePart>>,
}

type Result<T> = std::result::Result<T, MinHashingError>;

impl<KeyType: Eq + Hash + Clone> MinHashLsh<KeyType> {
    pub fn new(
        num_perm: usize,
        weights: Option<Weights>,
        threshold: Option<f64>,
    ) -> Result<MinHashLsh<KeyType>> {
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
        let params = LshParams::find_optimal_params(threshold, num_perm, &weights);

        let hash_tables = (0..params.b).into_iter().map(|_| HashMap::new()).collect();
        let hash_ranges = (0..params.b)
            .into_iter()
            .map(|i| (i * params.r, (i + 1) * params.r))
            .collect();
        Ok(MinHashLsh {
            num_perm,
            threshold,
            weights,
            buffer_size: 50_000,
            params,
            hash_tables,
            hash_ranges,
            keys: HashMap::<KeyType, Vec<HashValuePart>>::new(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.hash_tables.iter().any(|table| table.len() == 0)
    }

    pub fn insert(&mut self, key: KeyType, min_hash: &MinHash) -> Result<()> {
        // TODO: We could also add optional checks whether the key is already present in index
        // TODO: Why has the original implementation buffer params everywhere
        if min_hash.hash_values.0.len() != self.num_perm {
            return Err(MinHashingError::DifferentNumPermFuncs);
        }
        let mut hash_value_parts: Vec<HashValuePart> = self
            .hash_ranges
            .iter()
            .map(|(start, end)| {
                let hash_part = min_hash.hash_values.0[*start..*end].to_owned();
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

    pub fn remove(&mut self, key: &KeyType) -> Result<()> {
        if !self.keys.contains_key(key) {
            return Err(MinHashingError::KeyDoesNotExist);
        }
        for (hash_part, table) in self
            .keys
            .get_mut(key)
            .unwrap()
            .iter_mut()
            .zip(&mut self.hash_tables)
        {
            table.get_mut(hash_part).unwrap().remove(key);
            if let Some(set) = table.get(hash_part) {
                if set.is_empty() {
                    table.remove(hash_part);
                }
            }
        }
        self.keys.remove(key);
        Ok(())
    }

    pub fn get_counts(&self) -> Vec<HashMap<HashValuePart, usize>> {
        self.hash_tables
            .iter()
            .map(|table| {
                table
                    .iter()
                    .map(|(key, value)| (key.clone(), value.len()))
                    .collect()
            })
            .collect()
    }

    pub fn query(&mut self, min_hash: &MinHash) -> Result<HashSet<KeyType>> {
        if min_hash.hash_values.0.len() != self.num_perm {
            return Err(MinHashingError::DifferentNumPermFuncs);
        }
        let unique_candidates = self
            .hash_ranges
            .iter()
            .zip(&self.hash_tables)
            .flat_map(|(range, table)| {
                let (start, end) = range;
                let hash_part = min_hash.hash_values.0[*start..*end].to_owned();
                table.get(&HashValuePart(hash_part))
            })
            .flatten()
            .cloned()
            .collect();
        Ok(unique_candidates)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::minhash::MinHash;

    #[test]
    fn test_init() -> Result<()> {
        let lsh = <MinHashLsh<&str>>::new(128, None, Some(0.8))?;
        assert!(lsh.is_empty());
        let LshParams { b: b1, r: r1 } = lsh.params;
        let lsh = <MinHashLsh<&str>>::new(128, Some(Weights(0.2, 0.8)), Some(0.8))?;
        let LshParams { b: b2, r: r2 } = lsh.params;
        assert!(b1 < b2);
        assert!(r1 > r2);
        Ok(())
    }

    #[test]
    fn test_insert() -> Result<()> {
        let mut lsh = <MinHashLsh<&str>>::new(128, None, Some(0.5))?;
        let mut m1 = <MinHash>::new(128, Some(0));
        m1.update(&"a");
        let mut m2 = <MinHash>::new(128, Some(0));
        m2.update(&"b");
        lsh.insert("a", &m1)?;
        lsh.insert("b", &m2)?;
        for table in &lsh.hash_tables {
            assert!(table.len() >= 1);
            let table_values: HashSet<_> = table.values().flatten().collect();
            assert!(table_values.contains(&"a"));
            assert!(table_values.contains(&"b"));
        }
        assert!(lsh.contains_key(&"a"));
        assert!(lsh.contains_key(&"b"));
        let a_keys_content = lsh.keys.get(&"a").unwrap();
        for (index, hash_part) in a_keys_content.iter().enumerate() {
            assert!(lsh.hash_tables[index][hash_part].contains(&"a"));
        }
        Ok(())
    }

    #[test]
    fn test_query() -> Result<()> {
        let mut lsh = <MinHashLsh<&str>>::new(16, None, Some(0.5))?;
        let mut m1 = <MinHash>::new(16, Some(0));
        m1.update(&"a");
        let mut m2 = <MinHash>::new(16, Some(0));
        m2.update(&"b");
        lsh.insert("a", &m1)?;
        lsh.insert("b", &m2)?;
        let result = lsh.query(&m1)?;
        assert!(result.contains(&"a"));
        let result = lsh.query(&m2)?;
        assert!(result.contains(&"b"));
        assert!(result.len() <= 2);

        let m3 = <MinHash>::new(18, Some(0));
        let result = std::panic::catch_unwind(|| {
            lsh.clone().query(&m3).unwrap();
        });
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_remove() -> Result<()> {
        let mut lsh = <MinHashLsh<&str>>::new(16, None, Some(0.5))?;
        let mut m1 = <MinHash>::new(16, Some(0));
        m1.update(&"a");
        let mut m2 = <MinHash>::new(16, Some(0));
        m2.update(&"b");
        lsh.insert("a", &m1)?;
        lsh.insert("b", &m2)?;

        lsh.remove(&"a")?;
        assert!(!lsh.keys.contains_key("&a"));
        for table in lsh.hash_tables {
            for value in table.keys() {
                assert!(table[value].len() > 0);
                assert!(!table[value].contains(&"a"))
            }
        }
        Ok(())
    }

    #[test]
    fn test_get_counts() -> Result<()> {
        let mut lsh = <MinHashLsh<&str>>::new(16, None, Some(0.5))?;
        let mut m1 = <MinHash>::new(16, Some(0));
        m1.update(&"a");
        let mut m2 = <MinHash>::new(16, Some(0));
        m2.update(&"b");
        lsh.insert("a", &m1)?;
        lsh.insert("b", &m2)?;

        let counts = lsh.get_counts();
        assert_eq!(counts.len(), lsh.params.b);
        for table in &counts {
            assert_eq!(table.values().sum::<usize>(), 2);
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
        let mut m1 = <MinHash>::new(n_projections, Some(0));
        let mut m2 = <MinHash>::new(n_projections, Some(0));
        let mut m3 = <MinHash>::new(n_projections, Some(0));
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
        let mut lsh = <MinHashLsh<&str>>::new(128, None, Some(0.5))?;
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
