//! Benchmark throughput VQ forward (Burn, fp32).

use burn::prelude::*;
use burn::tensor::Device;
use std::hint::black_box;
use std::time::Instant;
use vir_vq::{Codebook, CodebookConfig};

fn main() {
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let device = Device::default();

    let cfg = CodebookConfig {
        num_entries: 4096,
        dim: 384,
        ..CodebookConfig::default()
    };
    let cb = Codebook::new(&cfg, &device);
    let batch = 256usize;
    let seq = 128usize;
    let z: Tensor<2> = Tensor::ones([batch * seq, cfg.dim], &device);
    println!("dtype = {:?}", z.dtype());

    for _ in 0..5 {
        let idx = cb.quantize_idx(z.clone());
        black_box(idx);
        let _ = device.sync();
    }

    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        let idx = cb.quantize_idx(z.clone());
        black_box(idx);
        let _ = device.sync();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tokens = (batch * seq * iters) as f64;
    println!(
        "vq_forward toks/s = {:.2} (elapsed {:.3}s, iters {}, dtype=fp32)",
        tokens / elapsed,
        elapsed,
        iters
    );
}