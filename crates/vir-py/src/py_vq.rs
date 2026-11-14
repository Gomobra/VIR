//! `vir.vq`: matematika codebook FSQ (PRD F1.4.f).
//!
//! Model VQ penuh (codebook EMA, multiscale, decoder) hidup di crate `vir-vq`
//! yang membutuhkan backend Burn dan tidak ikut ke build inti ini. Yang terpapar
//! di sini adalah rumus FSQ yang murni aritmetika host:
//!
//! ```text
//! half_l = (L−1)(1+ε)/2 ,  offset = 0,5 bila L genap
//! shift  = atanh(offset / half_l)
//! bound  = round(tanh(z + shift)·half_l − offset)
//! idx    = Σ_i (q_i + ⌊L_i/2⌋)·b_i ,  b_i = ∏_{j<i} L_j
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fsq_num_codes, m)?)?;
    m.add_function(wrap_pyfunction!(fsq_bound, m)?)?;
    m.add_function(wrap_pyfunction!(fsq_index, m)?)?;
    m.add_function(wrap_pyfunction!(fsq_decode, m)?)?;
    Ok(())
}

/// Setengah-lebar dan offset bound FSQ untuk satu level.
fn bound_params(level: i64, eps: f64) -> PyResult<(f64, f64)> {
    if level < 2 {
        return Err(PyValueError::new_err(format!(
            "level FSQ minimal 2, dapat {level}"
        )));
    }
    let half = (level as f64 - 1.0) * (1.0 + eps) / 2.0;
    let offset: f64 = if level % 2 == 0 { 0.5 } else { 0.0 };
    if half <= 0.0 || offset.abs() >= half {
        return Err(PyValueError::new_err(format!(
            "level FSQ tidak valid: L={level}, eps={eps}"
        )));
    }
    Ok((half, offset))
}

/// Jumlah kode yang dapat diindeks oleh `levels` (produk level).
#[pyfunction]
fn fsq_num_codes(levels: Vec<i64>) -> PyResult<i64> {
    let mut total: i64 = 1;
    for level in &levels {
        bound_params(*level, 0.0)?;
        total = total
            .checked_mul(*level)
            .ok_or_else(|| PyValueError::new_err("jumlah kode FSQ melebihi i64"))?;
    }
    Ok(total)
}

/// Bulatkan laten ke kode integer FSQ per dimensi.
#[pyfunction]
#[pyo3(signature = (z, levels, eps = 1e-6))]
fn fsq_bound(z: Vec<f64>, levels: Vec<i64>, eps: f64) -> PyResult<Vec<i64>> {
    if z.len() != levels.len() {
        return Err(PyValueError::new_err(format!(
            "panjang z ({}) harus sama dengan levels ({})",
            z.len(),
            levels.len()
        )));
    }
    let mut out = Vec::with_capacity(z.len());
    for (&zi, &level) in z.iter().zip(levels.iter()) {
        let (half, offset) = bound_params(level, eps)?;
        let shift = (offset / half).atanh();
        let bounded = (zi + shift).tanh() * half - offset;
        out.push(bounded.round() as i64);
    }
    Ok(out)
}

/// Kode integer → indeks codebook datar.
#[pyfunction]
fn fsq_index(codes: Vec<i64>, levels: Vec<i64>) -> PyResult<i64> {
    if codes.len() != levels.len() {
        return Err(PyValueError::new_err(format!(
            "panjang codes ({}) harus sama dengan levels ({})",
            codes.len(),
            levels.len()
        )));
    }
    let mut index: i64 = 0;
    let mut base: i64 = 1;
    for (&code, &level) in codes.iter().zip(levels.iter()) {
        let half = level / 2;
        let q = code
            .checked_add(half)
            .ok_or_else(|| PyValueError::new_err("kode FSQ di luar i64"))?;
        if q < 0 || q >= level {
            return Err(PyValueError::new_err(format!(
                "kode {code} di luar rentang level {level}"
            )));
        }
        index += q * base;
        base = base
            .checked_mul(level)
            .ok_or_else(|| PyValueError::new_err("indeks FSQ melebihi i64"))?;
    }
    Ok(index)
}

/// Indeks codebook datar → kode integer per dimensi.
#[pyfunction]
fn fsq_decode(index: i64, levels: Vec<i64>) -> PyResult<Vec<i64>> {
    if index < 0 {
        return Err(PyValueError::new_err(format!(
            "index FSQ harus >= 0, dapat {index}"
        )));
    }
    let mut rest = index;
    let mut out = Vec::with_capacity(levels.len());
    for &level in levels.iter().rev() {
        let q = rest % level;
        out.push(q - level / 2);
        rest /= level;
    }
    if rest != 0 {
        return Err(PyValueError::new_err(format!(
            "index {index} di luar rentang levels"
        )));
    }
    out.reverse();
    Ok(out)
}
