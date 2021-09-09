use std::{error, fmt};

#[derive(Debug)]
// https://doc.rust-lang.org/rust-by-example/error/multiple_error_types/wrap_error.html for from
//  example if we need it
pub enum MinHashingError {
    DifferentSeeds,
    DifferentNumPermFuncs
}

impl fmt::Display for MinHashingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MinHashingError::DifferentSeeds =>
                write!(f, "computing jaccard similarity between minhashes only works if they \
                use the same seed"),
            MinHashingError::DifferentNumPermFuncs =>
                write!(f, "computing jaccard similarity between minhashes only works if they \
                use the same number of permutation functions")
        }
    }
}

impl error::Error for MinHashingError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            MinHashingError::DifferentSeeds => None,
            MinHashingError::DifferentNumPermFuncs => None
        }
    }
}
