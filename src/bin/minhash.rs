use minhash::lsh_rs_minhash::{MinHash, VecHash};

pub fn main() {
    let n_projections = 3;
    let h = <MinHash>::new(n_projections, 5, Some(0));
    let hash = h.hash_vec(&[1, 0, 1, 0, 1]);
    assert_eq!(hash.len(), n_projections);
    println!("{:?}", hash);
}
