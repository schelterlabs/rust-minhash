use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Weights(f32, f32);

#[derive(Serialize, Deserialize, Clone)]
pub struct DataSketchMinHashLsh {
    num_perm: usize,
    threshold: f32,
    weights: Weights,
    buffer_size: usize
}

impl DataSketchMinHashLsh {
    pub fn new(num_perm: usize, seed: Option<u64>) -> DataSketchMinHashLsh {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_init_() {
        let m1 = <DataSketchMinHashLsh>::new(4, Some(0));
        let m2 = <DataSketchMinHashLsh>::new(4, Some(0));
    }
}