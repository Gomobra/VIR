use crate::config::LevelScheduleKind;

/// Pembulatan ke genap terdekat (banker's rounding), sesuai numpy/`np.round`
/// yang dipakai kode acuan NCM; `f64::round` (half away from zero) berbeda pada
/// nilai `.5` dan membuat jadwal menyimpang dari golden PRD (mis. WMT14 K=11).
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if (diff - 0.5).abs() < f64::EPSILON {
        // tepat di tengah: pilih yang genap
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// Jadwal panjang level: ℓ_k = round(1 + (L-1)*g((k-1)/(K-1))), k=1..K.
/// g(t) per jenis; K=1 mengembalikan [L].
pub fn level_schedule(
    kind: LevelScheduleKind,
    target_length: usize,
    num_levels: usize,
) -> Vec<usize> {
    if num_levels == 1 {
        return vec![target_length];
    }
    let l = target_length as f64;
    let mut out = Vec::with_capacity(num_levels);
    for k in 0..num_levels {
        let t = k as f64 / (num_levels - 1) as f64;
        let g = match kind {
            LevelScheduleKind::Linear => t,
            LevelScheduleKind::Quadratic => t * t,
            LevelScheduleKind::Sqrt => t.sqrt(),
            LevelScheduleKind::Log => {
                if t <= 0.0 {
                    0.0
                } else {
                    (1.0 + t).ln() / 2.0_f64.ln()
                }
            }
            LevelScheduleKind::Exp => (t.exp() - 1.0) / (std::f64::consts::E - 1.0),
            LevelScheduleKind::Cosine => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),
        };
        let val = round_half_even(1.0 + (l - 1.0) * g) as usize;
        let val = val.clamp(1, target_length);
        out.push(val);
    }
    let n = out.len();
    if n > target_length {
        // tidak mungkin naik ketat; kembalikan jadwal minimal 1..=L
        return (1..=target_length)
            .take(n)
            .chain(std::iter::repeat(target_length))
            .take(n)
            .collect();
    }
    // pastikan naik ketat dan akhir = L
    out[0] = out[0].max(1).min(target_length);
    for i in 1..n {
        out[i] = out[i].max(out[i - 1] + 1).min(target_length);
    }
    out[n - 1] = target_length;
    for i in (0..n - 1).rev() {
        out[i] = out[i].min(out[i + 1].saturating_sub(1)).max(1);
    }
    for i in 1..n {
        out[i] = out[i].max(out[i - 1] + 1);
    }
    out[n - 1] = target_length;
    out
}
