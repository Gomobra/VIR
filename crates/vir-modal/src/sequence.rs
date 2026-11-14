//! Perakitan urutan token multimodal (ADR-009 §4–§5).
//!
//! Urutan kode **level-major**: penanda `open`, seluruh kode level 0, level 1,
//! ... , penanda `close`. Video menyisipkan penanda waktu `<t=…>` di antara
//! frame (ADR-009 §4).

use vir_common::VirError;

use crate::kind::ModalKind;
use crate::token::VocabLayout;

/// Kode aset satu modalitas: satu `Vec<u32>` per level, level 0 = paling kasar.
pub fn encode_codes(
    layout: &VocabLayout,
    kind: ModalKind,
    levels: &[Vec<u32>],
) -> Result<Vec<u32>, VirError> {
    if kind == ModalKind::Text {
        return Err(VirError::Config("teks tidak punya kode".into()));
    }
    let markers = layout
        .marker(kind)
        .ok_or(VirError::Config(format!("tidak ada penanda {}", kind)))?;
    if levels.is_empty() {
        return Err(VirError::Config("daftar level kosong".into()));
    }
    let mut out = Vec::new();
    out.push(markers.open);
    for (level, codes) in levels.iter().enumerate() {
        if codes.is_empty() {
            return Err(VirError::Config(format!("level {level} kosong")));
        }
        let key = crate::token::CodebookKey::new(kind, level as u8);
        for code in codes {
            let id = layout.encode(key, *code).ok_or(VirError::Config(format!(
                "kode {code} di luar codebook {key}"
            )))?;
            out.push(id);
        }
    }
    out.push(markers.close);
    Ok(out)
}

/// Video: frame (kode gambar level 0) diselingi penanda waktu dalam detik.
pub fn encode_video(
    layout: &VocabLayout,
    frames: &[Vec<u32>],
    times_s: &[u32],
) -> Result<Vec<u32>, VirError> {
    if frames.is_empty() {
        return Err(VirError::Config("video tanpa frame".into()));
    }
    if frames.len() != times_s.len() {
        return Err(VirError::Config(format!(
            "frame {} != waktu {}",
            frames.len(),
            times_s.len()
        )));
    }
    let markers = layout
        .marker(ModalKind::Video)
        .ok_or(VirError::Config("tidak ada penanda video".into()))?;
    let image_key = crate::token::CodebookKey::new(ModalKind::Image, 0);
    let mut out = vec![markers.open];
    for (frame, t) in frames.iter().zip(times_s) {
        let time_id = markers.time(*t).ok_or(VirError::Config(format!(
            "penanda waktu {t} di luar jangkauan"
        )))?;
        out.push(time_id);
        for code in frame {
            let id = layout
                .encode(image_key, *code)
                .ok_or(VirError::Config(format!(
                    "kode frame {code} di luar codebook {image_key}"
                )))?;
            out.push(id);
        }
    }
    out.push(markers.close);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{build_layout, ModalToken};

    #[test]
    fn level_major_dengan_penanda() -> Result<(), VirError> {
        let layout = build_layout(256, 1, 2, 128)?;
        let audio = layout
            .marker(ModalKind::Audio)
            .ok_or(VirError::Internal("marker audio".into()))?;
        let ids = encode_codes(&layout, ModalKind::Audio, &[vec![0, 1], vec![2, 3]])?;
        assert_eq!(ids[0], audio.open);
        assert_eq!(*ids.last().unwrap_or(&0), audio.close);
        assert_eq!(ids.len(), 6);
        assert_eq!(
            layout.decode(ids[1]),
            Some(ModalToken::new(ModalKind::Audio, 0, 0))
        );
        assert_eq!(
            layout.decode(ids[3]),
            Some(ModalToken::new(ModalKind::Audio, 1, 2))
        );
        Ok(())
    }

    #[test]
    fn tolak_kode_dan_level_tidak_valid() -> Result<(), VirError> {
        let layout = build_layout(256, 1, 1, 128)?;
        assert!(encode_codes(&layout, ModalKind::Image, &[]).is_err());
        assert!(encode_codes(&layout, ModalKind::Image, &[vec![]]).is_err());
        assert!(encode_codes(&layout, ModalKind::Image, &[vec![128]]).is_err());
        assert!(encode_codes(&layout, ModalKind::Image, &[vec![0], vec![0]]).is_err());
        assert!(encode_codes(&layout, ModalKind::Text, &[vec![0]]).is_err());
        Ok(())
    }

    #[test]
    fn video_dengan_penanda_waktu() -> Result<(), VirError> {
        let layout = build_layout(256, 1, 1, 128)?;
        let ids = encode_video(&layout, &[vec![0, 1], vec![2]], &[0, 1])?;
        let video = layout
            .marker(ModalKind::Video)
            .ok_or(VirError::Internal("marker video".into()))?;
        assert_eq!(ids[0], video.open);
        assert_eq!(*ids.last().unwrap_or(&0), video.close);
        assert!(ids.contains(&video.time(0).unwrap_or(0)));
        assert!(ids.contains(&video.time(1).unwrap_or(0)));
        assert_eq!(ids.len(), 1 + 1 + 2 + 1 + 1 + 1);
        assert!(encode_video(&layout, &[], &[]).is_err());
        assert!(encode_video(&layout, &[vec![0]], &[]).is_err());
        Ok(())
    }
}
