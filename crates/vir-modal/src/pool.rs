//! Operator turun/naik antar level (ADR-009 §2–§3): `Down_k` = average pooling
//! area, `R_k` = nearest neighbor. Dipakai bersama tokenizer gambar multiscale
//! (F6.3) dan codec audio residual (F6.4), sehingga kedua jalur memakai rumus
//! yang sama seperti `vir-vq::multiscale`.

use vir_common::VirError;

/// Turun `rows_in → rows_out` baris berdimensi `dim` dengan rata-rata blok
/// (blok `[⌊j·n/m⌋, ⌊(j+1)·n/m⌋)`, blok terakhir menutup `n`).
pub fn avg_pool_rows(
    x: &[f32],
    rows_in: usize,
    rows_out: usize,
    dim: usize,
) -> Result<Vec<f32>, VirError> {
    if dim == 0 || rows_in == 0 || rows_out == 0 || rows_out > rows_in {
        return Err(VirError::Config(format!(
            "avg pool {rows_in}→{rows_out} tidak valid"
        )));
    }
    check_len(x, rows_in, dim)?;
    let mut out = vec![0.0f32; rows_out * dim];
    for (j, row) in out.chunks_exact_mut(dim).enumerate() {
        let start = j * rows_in / rows_out;
        let mut end = ((j + 1) * rows_in / rows_out).max(start + 1).min(rows_in);
        if j + 1 == rows_out {
            end = rows_in;
        }
        let count = (end - start) as f32;
        for (slot, k) in row.iter_mut().zip(0..dim) {
            let mut acc = 0.0f32;
            for i in start..end {
                acc += x[i * dim + k];
            }
            *slot = acc / count;
        }
    }
    Ok(out)
}

/// Naik `rows_in → rows_out` baris dengan nearest neighbor (faktor naik bebas).
pub fn nearest_up_rows(
    x: &[f32],
    rows_in: usize,
    rows_out: usize,
    dim: usize,
) -> Result<Vec<f32>, VirError> {
    if dim == 0 || rows_in == 0 || rows_out < rows_in {
        return Err(VirError::Config(format!(
            "nearest up {rows_in}→{rows_out} tidak valid"
        )));
    }
    check_len(x, rows_in, dim)?;
    let mut out = vec![0.0f32; rows_out * dim];
    for i in 0..rows_out {
        let src = i * rows_in / rows_out;
        out[i * dim..(i + 1) * dim].copy_from_slice(&x[src * dim..(src + 1) * dim]);
    }
    Ok(out)
}

/// Jumlahkan `b` ke `a` di tempat (panjang harus sama).
pub fn add_assign(a: &mut [f32], b: &[f32]) -> Result<(), VirError> {
    if a.len() != b.len() {
        return Err(VirError::Config(format!(
            "panjang beda saat penjumlahan: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    for (slot, value) in a.iter_mut().zip(b.iter()) {
        *slot += value;
    }
    Ok(())
}

/// `a -= b` di tempat (update residual setelah rekonstruksi bertambah).
pub fn sub_assign(a: &mut [f32], b: &[f32]) -> Result<(), VirError> {
    if a.len() != b.len() {
        return Err(VirError::Config(format!(
            "panjang beda saat pengurangan: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    for (slot, value) in a.iter_mut().zip(b.iter()) {
        *slot -= value;
    }
    Ok(())
}

fn check_len(x: &[f32], rows: usize, dim: usize) -> Result<(), VirError> {
    if x.len() != rows * dim {
        return Err(VirError::Config(format!(
            "panjang x {} != {rows}×{dim}",
            x.len()
        )));
    }
    Ok(())
}

