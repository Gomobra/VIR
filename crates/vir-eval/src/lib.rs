//! Evaluasi: pass@k, rekonstruksi, RULER-style konteks panjang.

pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n - c < k {
        return 1.0;
    }
    let mut prod = 1.0f64;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - (k as f64) / (i as f64);
    }
    1.0 - prod
}

pub fn eval_pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    pass_at_k(n, c, k)
}

/// Pembangkit needle-in-haystack deterministik (sederhana).
pub fn gen_needle(seed: u64, haystack_tokens: usize) -> (Vec<u32>, u32) {
    // jarum = token khusus 42 di posisi acak
    let mut v = vec![1u32; haystack_tokens];
    let pos = (seed as usize * 37) % haystack_tokens.max(1);
    v[pos] = 42;
    (v, 42)
}

/// Rekonstruksi eksak: persentase posisi yang cocok.
pub fn recon_accuracy(original: &[u32], reconstructed: &[u32]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let n = original.len().min(reconstructed.len());
    let mut ok = 0usize;
    for i in 0..n {
        if original[i] == reconstructed[i] {
            ok += 1;
        }
    }
    ok as f64 / original.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_at_k_delegates() {
        assert!((eval_pass_at_k(4, 1, 2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn recon_accuracy_full() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        assert!((recon_accuracy(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recon_accuracy_partial() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 9, 3, 4];
        assert!((recon_accuracy(&a, &b) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn needle_contains_jarum() {
        let (hay, needle) = gen_needle(7, 100);
        assert!(hay.contains(&needle));
    }
}
