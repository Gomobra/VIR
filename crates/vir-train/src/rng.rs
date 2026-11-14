//! RNG deterministik yang **bisa diserialisasi penuh** (xoshiro256++).
//!
//! Alasan ada RNG sendiri (bukan `rand_chacha`): checkpoint F1.7 wajib
//! menghasilkan langkah berikutnya yang identik, jadi state RNG harus ikut
//! tersimpan di `checkpoint.rs`. State xoshiro hanya 4×u64, cukup untuk satu
//! field JSON dan untuk fork per level (corruption F1.7 langkah 2).

use serde::{Deserialize, Serialize};

/// State RNG: empat kata 64-bit. Tersimpan apa adanya di checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
}

/// SplitMix64 dipakai hanya untuk menyiapkan state awal dari satu seed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// RNG baru dari satu seed 64-bit.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in s.iter_mut() {
            *slot = splitmix64(&mut sm);
        }
        if s == [0, 0, 0, 0] {
            s[0] = 1;
        }
        Self { s }
    }

    /// RNG dari state tersimpan (resume).
    pub fn from_state(s: [u64; 4]) -> Self {
        let mut rng = Self { s };
        if s == [0, 0, 0, 0] {
            rng.s[0] = 1;
        }
        rng
    }

    /// State saat ini, untuk checkpoint.
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Turunan RNG dengan label (mis. nama level); berguna agar tiap level
    /// punya aliran angka sendiri tanpa saling mengganggu.
    pub fn fork(&self, label: u64) -> Self {
        let mut acc = self.s[0] ^ self.s[3].rotate_left(17) ^ label;
        acc = splitmix64(&mut acc);
        Self::new(acc ^ self.s[1])
    }

    /// Ulang berikutnya (xoshiro256++).
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Float seragam di `[0, 1)` dengan 24 bit mantissa.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / 16_777_216.0)
    }

    /// Integer seragam di `[0, n)` (Lemire, tanpa bias berarti).
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let v = self.next_u64() as u128;
        ((v * n as u128) >> 64) as usize
    }

    /// Fisher–Yates in-place.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        if xs.len() < 2 {
            return;
        }
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// Ambil `k` posisi berbeda dari `0..n` (Fisher–Yates parsial).
    /// Dipakai corruption per level: jumlah terkorupsi dibuat tepat, bukan
    /// sekadar diharapkan.
    pub fn sample_positions(&mut self, n: usize, k: usize) -> Vec<usize> {
        let k = k.min(n);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            order.swap(i, j);
        }
        order.truncate(k);
        order
    }
}

/// Uji asap RNG: seed sama → urutan sama, state dapat dipulihkan, dan
/// `sample_positions` mengembalikan permutasi tanpa duplikat.
pub fn rng_smoke() -> Result<(), vir_common::VirError> {
    let bad = |m: &str| Err(vir_common::VirError::Internal(m.to_string()));
    let seq = |seed: u64| {
        let mut r = Rng::new(seed);
        (0..64).map(|_| r.next_u64()).collect::<Vec<u64>>()
    };
    if seq(7) != seq(7) {
        return bad("RNG tidak deterministik");
    }
    if seq(7) == seq(8) {
        return bad("seed berbeda menghasilkan urutan sama");
    }
    let mut a = Rng::new(7);
    for _ in 0..16 {
        let _ = a.next_u64();
    }
    let mut b = Rng::from_state(a.state());
    for _ in 0..16 {
        if a.next_u64() != b.next_u64() {
            return bad("state RNG tidak pulih");
        }
    }
    let pos = Rng::new(11).sample_positions(10, 10);
    if pos.len() != 10 || pos.iter().sum::<usize>() != 45 {
        return bad("sample_positions bukan permutasi penuh");
    }
    let few = Rng::new(11).sample_positions(1000, 5);
    if few.len() != 5 || few.iter().any(|p| *p >= 1000) {
        return bad("sample_positions di luar rentang");
    }
    Ok(())
}