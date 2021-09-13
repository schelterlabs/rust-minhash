use std::{error, fmt};

#[derive(Debug)]
// https://doc.rust-lang.org/rust-by-example/error/multiple_error_types/wrap_error.html for from
//  example if we need it
pub enum MinHashingError {
    DifferentSeeds,
    DifferentNumPermFuncs,
    WrongThresholdInterval,
    NumPermFuncsTooLow,
    WrongWeightThreshold,
    UnexpectedSumWeight
}

impl fmt::Display for MinHashingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MinHashingError::DifferentSeeds =>
                write!(f, "computing jaccard similarity between minhashes only works if they \
                use the same seed"),
            MinHashingError::DifferentNumPermFuncs =>
                write!(f, "computing jaccard similarity between minhashes only works if they \
                use the same number of permutation functions"),
            MinHashingError::WrongThresholdInterval => write!(f, "threshold must be in [0.0, 1.0]"),
            MinHashingError::NumPermFuncsTooLow => write!(f, "Too few permutation functions"),
            MinHashingError::WrongWeightThreshold => write!(f, "Weight must be in [0.0, 1.0]"),
            MinHashingError::UnexpectedSumWeight => write!(f, "Weights must sum to 1.0"),
        }
    }
}

impl error::Error for MinHashingError {}
