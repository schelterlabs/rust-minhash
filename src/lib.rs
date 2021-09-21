//! # minhash-lsh
//! [![Build Status](https://github.com/schelterlabs/rust-minhash/actions/workflows/build.yml/badge.svg)](https://github.com/schelterlabs/rust-minhash/actions/workflows/build.yml)
//!
//! This crate reimplements the `MinHash` and `MinHash LSH` approaches from the Python package [datasketch](https://github.com/ekzhu/datasketch) in Rust. It's only a partial reimplementation, use it at your own risk.
//!
//! ## Example MinHash
//!
//! ```rust
//!  use datasketch_minhash_lsh::MinHash;
//!
//!  let mut m1 = <MinHash>::new(4, Some(1));
//!  let mut m2 = <MinHash>::new(4, Some(1));
//!  assert_eq!(m1.jaccard(&m2)?, 1.0);
//!
//!  m2.update(&12);
//!  assert_eq!(m1.jaccard(&m2)?, 0.0);
//!
//!  m1.update(&13);
//!  assert!(m1.jaccard(&m2)? < 1.0);
//!
//!  m1.update(&12);
//!  let distance = m1.jaccard(&m2)?;
//!  assert!(distance < 1.0 && distance > 0.0);
//! ```
//! ## Example MinHashLsh
//!
//! ```rust
//!  use datasketch_minhash_lsh::{MinHashLsh, MinHash};
//!
//!  let mut lsh = <MinHashLsh<&str>>::new(16, None, Some(0.5))?;
//!  let mut m1 = <MinHash>::new(16, Some(0));
//!  m1.update(&"a");
//!
//!  let mut m2 = <MinHash>::new(16, Some(0));
//!  m2.update(&"b");
//!
//!  lsh.insert("a", &m1)?;
//!  lsh.insert("b", &m2)?;
//!
//!  let result = lsh.query(&m1)?;
//!  assert!(result.contains(&"a"));
//!
//!  let result = lsh.query(&m2)?;
//!  assert!(result.contains(&"b"));
//!  assert!(result.len() <= 2);
//! ```
//!

use rand::prelude::SmallRng;
use rand::{thread_rng, SeedableRng};

mod error;
mod minhash;
mod minhash_lsh;

pub use crate::minhash::*;
pub use crate::minhash_lsh::*;

pub fn create_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        _ => match SmallRng::from_rng(thread_rng()) {
            Ok(rng) => rng,
            Err(_) => SmallRng::from_entropy(),
        },
    }
}
