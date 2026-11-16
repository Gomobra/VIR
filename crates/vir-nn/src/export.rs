//! Ekspor/muat model terkuantisasi (F7.2 langkah 2): format `vir-quant-1`.
//!
//! Direktori `{path}/` berisi:
//!
//! | Berkas | Isi |
//! |---|---|
//! | `meta.json` | skema, granularitas, `block_size`, `numel`, `num_blocks`, bentuk |
//! | `data.bin` | nilai terkuantisasi (int8/int4-packed/E4M3) + skala f32 LE |
//!
//! Muat ulang mendekodekan bit yang sama, sehingga hasil dekuantisasi
//! identik bit demi bit (round-trip deterministik).

use crate::quant::{QuantGranularity, QuantScheme, QuantizedTensor};
use vir_common::VirError;

/// Nama berkas metadata di dalam direktori ekspor.
const META_FILE: &str = "meta.json";
/// Nama berkas payload di dalam direktori ekspor.
const DATA_FILE: &str = "data.bin";

/// Tulis model terkuantisasi ke `{path}/` (meta.json + data.bin).
///
/// `shape` opsional (metadata); dekode ulang tidak membutuhkannya.
pub fn export_quantized(
    qt: &QuantizedTensor,
    path: &str,
    shape: Option<&[usize]>,
) -> Result<(), VirError> {
    let meta = encode_meta(qt, shape)?;
    let payload = encode_payload(qt);
    std::fs::create_dir_all(path)?;
    std::fs::write(format!("{path}/{META_FILE}"), meta)?;
    std::fs::write(format!("{path}/{DATA_FILE}"), payload)?;
    Ok(())
}

/// Muat ulang hasil ekspor sebagai [`QuantizedTensor`] mentah.
pub fn read_export(path: &str) -> Result<QuantizedTensor, VirError> {
    let meta = std::fs::read_to_string(format!("{path}/{META_FILE}"))?;
    let payload = std::fs::read(format!("{path}/{DATA_FILE}"))?;
    decode_export(&meta, &payload)
}

/// Muat ulang bobot terkuantisasi (dekuantisasi) dari berkas ekspor.
pub fn load_quantized(path: &str) -> Result<Vec<f32>, VirError> {
    let qt = read_export(path)?;
    crate::quant::dequantize_tensor(&qt)
}

/// Metadata JSON ekspor (`vir-quant-1`).
fn encode_meta(qt: &QuantizedTensor, shape: Option<&[usize]>) -> Result<String, VirError> {
    let (kind, block) = match qt.granularity {
        QuantGranularity::PerTensor => ("per_tensor", 0usize),
        QuantGranularity::PerBlock { block_size } => ("per_block", block_size),
    };
    let scheme = match qt.scheme {
        QuantScheme::Int8 => "int8",
        QuantScheme::Int4 => "int4",
        QuantScheme::Fp8 => "fp8",
    };
    let meta = serde_json::json!({
        "version": crate::quant::EXPORT_VERSION,
        "scheme": scheme,
        "granularity": kind,
        "block_size": block,
        "numel": qt.numel(),
        "num_blocks": qt.scales.len(),
        "shape": shape,
    });
    serde_json::to_string(&meta).map_err(|e| VirError::Internal(e.to_string()))
}

/// Payload biner: nilai terkuantisasi lalu skala f32 little-endian.
fn encode_payload(qt: &QuantizedTensor) -> Vec<u8> {
    let mut out = qt.packed.clone();
    for s in &qt.scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Rekonstruksi dari metadata + payload (kebalikan [`encode_payload`]).
fn decode_export(meta: &str, payload: &[u8]) -> Result<QuantizedTensor, VirError> {
    #[derive(serde::Deserialize)]
    struct Meta {
        scheme: String,
        granularity: String,
        block_size: usize,
        numel: usize,
        num_blocks: usize,
        shape: Vec<usize>,
    }
    let m: Meta = serde_json::from_str(meta).map_err(|e| VirError::Internal(e.to_string()))?;
    let scheme = match m.scheme.as_str() {
        "int8" => QuantScheme::Int8,
        "int4" => QuantScheme::Int4,
        "fp8" => QuantScheme::Fp8,
        other => return Err(VirError::Config(format!("skema tak dikenal: {other}"))),
    };
    let granularity = match m.granularity.as_str() {
        "per_tensor" => QuantGranularity::PerTensor,
        "per_block" => QuantGranularity::PerBlock {
            block_size: m.block_size,
        },
        other => {
            return Err(VirError::Config(format!(
                "granularitas tak dikenal: {other}"
            )));
        }
    };
    let packed_len = m.numel.div_ceil(8 / scheme.bits());
    if payload.len() < packed_len + m.num_blocks * 4 {
        return Err(VirError::Internal("payload ekspor terpotong".into()));
    }
    let mut scales = Vec::with_capacity(m.num_blocks);
    for i in 0..m.num_blocks {
        let off = packed_len + i * 4;
        let b = [
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ];
        scales.push(f32::from_le_bytes(b));
    }
    Ok(QuantizedTensor {
        scheme,
        granularity,
        shape: m.shape,
        packed: payload[..packed_len].to_vec(),
        scales,
    })
}
