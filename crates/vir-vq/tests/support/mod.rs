//! Pembantu tes CPU untuk `vir-vq` (tanpa CUDA): RNG deterministik kecil,
//! pembaca tensor ke `Vec<f32>`/`Vec<i32>`, dan harness ablasi (F1.4.f).

#![allow(dead_code)]

pub mod ablation;

use burn::tensor::{Int, Tensor, TensorData};

/// RNG linear congruential sederhana (deterministik, tanpa dependensi).
pub struct Lcg(u64);

impl rand_core::RngCore for Lcg {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        Lcg::next_u64(self)
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            let n = chunk.len().min(8);
            chunk.copy_from_slice(&v[..n]);
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// Vektor f32 acak deterministik.
pub fn floats(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit()).collect()
}

/// Tensor 2D `[rows, cols]` dari data baris.
pub fn tensor2(rows: usize, cols: usize, data: &[f32]) -> Tensor<2> {
    Tensor::<2>::from_floats(
        TensorData::new(data.to_vec(), [rows, cols]),
        &Default::default(),
    )
}

/// Tensor 1D float.
pub fn vector1(data: &[f32]) -> Tensor<1> {
    Tensor::<1>::from_floats(
        TensorData::new(data.to_vec(), [data.len()]),
        &Default::default(),
    )
}

/// Tensor 1D Int dari data.
pub fn int1(data: &[i32]) -> Tensor<1, Int> {
    Tensor::<1, Int>::from_ints(
        TensorData::new(data.to_vec(), [data.len()]),
        &Default::default(),
    )
}

/// Baca tensor 2D float ke `Vec<f32>`.
pub fn read2(t: &Tensor<2>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().expect("f32")
}

/// Baca tensor 1D float ke `Vec<f32>`.
pub fn read1(t: &Tensor<1>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().expect("f32")
}

/// Baca tensor 1D Int ke `Vec<i32>`.
pub fn read_int1(t: &Tensor<1, Int>) -> Vec<i32> {
    t.to_data().try_to_vec::<i32>().expect("i32")
}
