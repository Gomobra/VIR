//! Benchmark throughput VQ forward (Burn).

use std::time::Instant;
use vir_vq::{Codebook, CodebookConfig};
use burn::tensor::Device;
use burn::prelude::*;

fn main() {
    let device = Device::default();
    let cfg = CodebookConfig { num_entries: 4096, dim: 384 };
    let cb = Codebook::new(&cfg, &device);
    let batch = 256usize;
    let seq = 128usize;
    let z: Tensor<2> = Tensor::ones([batch * seq, cfg.dim], &device);

    // warmup
    for _ in 0..5 {
        let _ = cb.quantize(z.clone());
    }
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = cb.quantize(z.clone());
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tokens = (batch * seq * iters) as f64;
    let toks_per_s = tokens / elapsed;
    println!("vq_forward toks/s = {:.2} (elapsed {:.3}s, iters {})", toks_per_s, elapsed, iters);
}