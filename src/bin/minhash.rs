use minhash::datasketch_minhash::DataSketchMinHash;
use minhash::lsh_rs_minhash::{MinHash, VecHash};

pub fn main() {
    let n_projections = 3;

    // lsh-rs
    let h = <MinHash>::new(n_projections, 5, Some(0));
    let hash = h.hash_vec(&[1, 0, 1, 0, 1]);
    assert_eq!(hash.len(), n_projections);
    println!("Using lsh-rs approach: {:?}", hash);

    // datasketch
    let mut m = <DataSketchMinHash>::new(n_projections, Some(0));
    m.update(&0);
    m.update(&2);
    m.update(&4);
    assert_eq!(m.hash_values.0.len(), n_projections);
    println!("Using datasketch approach: {:?}", &m.hash_values);
}
