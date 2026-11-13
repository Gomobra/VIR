//! Benchmark throughput VQ forward (Burn, fp32).
//!
//! Jalur yang diukur:
//! - referensi: matmul + argmax (chunked) — `quantize_idx_reference`
//! - terfusi: satu kernel, tanpa buffer `[N, V]` — `quantize_idx` dengan feature
//!   `fused` (`--fused` atau `VIR_FUSED=1`; diabaikan bila build tanpa `fused`
//!   atau device bukan CUDA).

use burn::prelude::*;
use burn::tensor::Device;
use std::hint::black_box;
use std::time::Instant;
use vir_vq::{Codebook, CodebookConfig};

/// `true` bila pemanggil meminta jalur terfusi.
fn fused_requested() -> bool {
    let flag = std::env::args().any(|arg| arg == "--fused");
    let env = std::env::var("VIR_FUSED")
        .map(|v| v == "1")
        .unwrap_or(false);
    flag || env
}

/// Jalur terfusi benar-benar tersedia untuk device ini?
fn fused_available(_device: &Device) -> bool {
    #[cfg(feature = "fused")]
    {
        vir_vq::kernel::is_cube_device(_device)
    }
    #[cfg(not(feature = "fused"))]
    {
        false
    }
}

/// Ukur satu jalur, kembalikan detik total.
fn time_path(
    iters: usize,
    mut run: impl FnMut() -> burn::tensor::Tensor<1, Int>,
    device: &Device,
) -> f64 {
    for _ in 0..5 {
        black_box(run());
        let _ = device.sync();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run());
        let _ = device.sync();
    }
    t0.elapsed().as_secs_f64()
}

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
    #[cfg(feature = "fused")]
    println!("fused = true");
    #[cfg(not(feature = "fused"))]
    println!("fused = false");

    let iters = 50;
    let tokens = (batch * seq * iters) as f64;
    let report = |label: &str, elapsed: f64| {
        println!(
            "{label:<18} toks/s = {:>12.2}  (elapsed {elapsed:.3}s, iters {iters})",
            tokens / elapsed
        );
    };

    let elapsed = time_path(iters, || cb.quantize_idx_reference(z.clone()), &device);
    report("matmul+argmax", elapsed);

    if fused_requested() && fused_available(&device) {
        let elapsed = time_path(iters, || cb.quantize_idx(z.clone()), &device);
        report("fused", elapsed);
    } else if fused_requested() {
        println!("fused              dilewati (butuh build --features fused dan device CUDA)");
    }
}
