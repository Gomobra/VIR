use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// RNG deterministik berbasis seed, untuk tes dan resume.
pub struct DeterministicRng {
    inner: ChaCha20Rng,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    pub fn inner_mut(&mut self) -> &mut ChaCha20Rng {
        &mut self.inner
    }
}
