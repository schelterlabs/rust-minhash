use minhash::minhash::MinHash;

pub fn main() {
    let n_projections = 3;

    let mut m = <MinHash>::new(n_projections, Some(0));
    m.update(&0);
    m.update(&2);
    m.update(&4);
    assert_eq!(m.hash_values.0.len(), n_projections);
    println!("Using datasketch approach: {:?}", &m.hash_values);
}
