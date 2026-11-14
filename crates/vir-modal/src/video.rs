//! Video understanding: frame sampler + penanda waktu (F6.5).

use vir_common::VirError;

/// Sampling frame: laju tetap (1 frame per `stride` detik).
pub fn sample_fixed(duration_s: u32, fps: u32, stride: u32) -> Result<Vec<u32>, VirError> {
    if stride == 0 {
        return Err(VirError::Config("stride 0".into()));
    }
    let total = (duration_s * fps) as usize;
    let step = (stride * fps) as usize;
    let mut times = Vec::new();
    let mut t = 0usize;
    while t < total {
        times.push((t / fps as usize) as u32);
        t += step;
    }
    Ok(times)
}

/// Sampling frame berbasis perubahan adegan (threshold selisih rata-rata).
pub fn sample_scene_change(frame_diffs: &[f32], threshold: f32) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, &d) in frame_diffs.iter().enumerate() {
        if d >= threshold {
            out.push(i as u32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_sampling() {
        let t = sample_fixed(10, 30, 1).unwrap();
        assert_eq!(t.len(), 10);
        assert_eq!(t[0], 0);
        assert_eq!(t[9], 9);
    }

    #[test]
    fn scene_change() {
        let diffs = [0.1, 0.5, 0.2, 0.9, 0.05];
        let s = sample_scene_change(&diffs, 0.4);
        assert_eq!(s, vec![1, 3]);
    }
}
