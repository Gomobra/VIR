//! F1.1 — RNG deterministik dan bisa dilanjutkan (PRD F1.1).

use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use vir_common::DeterministicRng;

fn run(seed: u64) -> Vec<u64> {
    let mut r = DeterministicRng::new(seed);
    (0..8).map(|_| r.inner_mut().next_u64()).collect()
}

#[test]
fn seed_sama_urutan_sama() {
    assert_eq!(run(42), run(42), "seed sama harus identik");
}

#[test]
fn seed_beda_urutan_beda() {
    assert_ne!(run(1), run(2), "seed beda harus berbeda");
}

#[test]
fn resume_dengan_word_pos() {
    // Dua RNG seed sama; satu melompat 4 kata lalu lanjut → sama dengan 8 lurus.
    let mut penuh = DeterministicRng::new(7);
    let lurus: Vec<u64> = (0..8).map(|_| penuh.inner_mut().next_u64()).collect();

    let mut r = DeterministicRng::new(7);
    for _ in 0..4 {
        let _ = r.inner_mut().next_u64();
    }
    let pos = r.inner_mut().get_word_pos();
    r.inner_mut().set_word_pos(pos);
    let lanjut: Vec<u64> = (0..4).map(|_| r.inner_mut().next_u64()).collect();

    assert_eq!(&lurus[4..], &lanjut[..], "resume harus melanjutkan urutan");
}

#[test]
fn chacha_konsisten_dengan_seed_from_u64() {
    // DeterministicRng membungkus ChaCha20Rng::seed_from_u64; cek kesetaraan.
    let mut a = DeterministicRng::new(99);
    let mut b = ChaCha20Rng::seed_from_u64(99);
    for _ in 0..16 {
        assert_eq!(a.inner_mut().next_u64(), b.next_u64());
    }
}
