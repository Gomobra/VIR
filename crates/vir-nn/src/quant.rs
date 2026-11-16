//! PTQ nyata (F7.2): kuantisasi pasca-latih bobot ke int8/int4/fp8 (E4M3),
//! per-tensor atau per-blok, plus ekspor/muat model terkuantisasi dan
//! pengukuran kualitas (MSE) vs ukuran (byte).
//!
//! Arsitektur mengikuti Burn 0.22 (`burn::tensor::quantization`): skala
//! simetris (nol = tepat 0, tanpa zero-point), `q = round(clamp(x/scale))`,
//! rekonstruksi `x̂ = q·scale`. Nilai bulat disimpan sebagai `i8`; int4
//! dikemas 2 nilai/byte (low nibble = elemen genap). fp8 E4M3: 1 tanda,
//! 4 eksponen (bias 7), 3 mantissa, maksimum 448; NaN (0x7F/0xFF)
//! dihindari dengan saturasi ke 448.
//!
//! Skala MinMax mengikuti Burn Flex: `scale = peak/max(|a|,|b|)` dengan
//! `[a, b]` = rentang skema (`[-127,127]`, `[-7,7]`, `[-448,448]`), sama
//! dengan `2·peak/(b−a)`; `0` selalu tepat dan puncak selalu terwakili.
//!
//! Catatan fp8: backend non-Flex Burn menyimpan E4M3 sebagai *nilai* bulat
//! (bukan bit IEEE), sedangkan Flex menyimpan `i8` beralih-float. Modul ini
//! memakai representasi bit IEEE (seperti spesifikasi FP8), yang untuk data
//! terskala (skala ekspor ikut disimpan) ekuivalen: perbedaan hanya pada
//! nilai mentah, bukan hasil dekuantisasi.
//!
//! Format ekspor (`vir-quant-1`): direktori `{path}/` berisi `meta.json`
//! (skema, granularitas, ukuran blok, jumlah blok, bentuk opsional) dan
//! `data.bin` (nilai terkuantisasi + semua skala f32, little-endian).
//! Muat ulang mendekodekan bit yang sama, sehingga hasil dekuantisasi
//! identik bit demi bit (round-trip deterministik).

use burn::tensor::quantization::QuantValue;
use vir_common::VirError;

/// Versi format ekspor.
pub const EXPORT_VERSION: &str = "vir-quant-1";

/// Skema kuantisasi bobot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Integer simetris 8-bit, rentang [-127, 127].
    Int8,
    /// Integer simetris 4-bit, rentang [-7, 7], dikemas 2 nilai/byte.
    Int4,
    /// Float 8-bit E4M3 (rentang [-448, 448]).
    Fp8,
}

impl QuantScheme {
    /// Rentang kuantisasi Burn-setara `[a, b]` untuk skala MinMax:
    /// int simetris = `[-q, q]`, fp8 = `[-448, 448]`.
    fn range(self) -> (f32, f32) {
        match self {
            QuantScheme::Int8 => (-127.0, 127.0),
            QuantScheme::Int4 => (-7.0, 7.0),
            QuantScheme::Fp8 => (-448.0, 448.0),
        }
    }

    /// Skala dari puncak `peak = max|x|` (rumus MinMax Burn:
    /// `scale = 2·peak/(b−a) = peak/max(|a|,|b|)`, sehingga `0` tepat).
    fn scale_from_peak(self, peak: f32) -> f32 {
        let (a, b) = self.range();
        peak / a.abs().max(b)
    }

    /// Bit per nilai terkuantisasi.
    pub fn bits(self) -> usize {
        match self {
            QuantScheme::Int8 | QuantScheme::Fp8 => 8,
            QuantScheme::Int4 => 4,
        }
    }

    /// Versi Burn yang setara (untuk pembanding opsional di tes).
    pub fn burn_value(self) -> QuantValue {
        match self {
            QuantScheme::Int8 => QuantValue::Q8S,
            QuantScheme::Int4 => QuantValue::Q4S,
            QuantScheme::Fp8 => QuantValue::E4M3,
        }
    }
}

/// Granularitas skala.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    /// Satu skala untuk seluruh tensor.
    PerTensor,
    /// Satu skala per blok `block_size` nilai berurutan.
    PerBlock { block_size: usize },
}

/// Hasil kuantisasi satu tensor: nilai bulat (i8; int4 = 2 nilai/byte
/// pada `packed`), skala per blok, dan granularitas.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Skema.
    pub scheme: QuantScheme,
    /// Granularitas.
    pub granularity: QuantGranularity,
    /// Bentuk tensor asli (`[]` = 1-D tanpa metadata bentuk).
    pub shape: Vec<usize>,
    /// Nilai terkuantisasi; untuk int4, 2 nilai per byte (low = genap).
    pub packed: Vec<u8>,
    /// Skala per blok (`len == 1` untuk per-tensor).
    pub scales: Vec<f32>,
}

impl QuantizedTensor {
    /// Jumlah elemen tensor asli (dipulihkan dari panjang payload).
    pub fn numel(&self) -> usize {
        match self.scheme {
            QuantScheme::Int8 | QuantScheme::Fp8 => self.packed.len(),
            QuantScheme::Int4 => self.packed.len() * 2,
        }
    }

    /// Skala sebuah elemen (indeks rata).
    fn scale_of(&self, idx: usize) -> f32 {
        match self.granularity {
            QuantGranularity::PerTensor => self.scales[0],
            QuantGranularity::PerBlock { block_size } => self.scales[idx / block_size],
        }
    }

    /// Dekode nilai bulat teranggal ke `Vec<i8>` (int4 di-unpack).
    fn unpack(&self) -> Result<Vec<i8>, VirError> {
        match self.scheme {
            QuantScheme::Int8 | QuantScheme::Fp8 => {
                Ok(self.packed.iter().map(|b| *b as i8).collect())
            }
            QuantScheme::Int4 => {
                let mut out = Vec::with_capacity(self.packed.len() * 2);
                for b in &self.packed {
                    out.push(nibble_to_i8(b & 0x0F));
                    out.push(nibble_to_i8(b >> 4));
                }
                Ok(out)
            }
        }
    }
}

/// Ubah nibble 4-bit unsigned ke i8 bertanda (two's complement).
fn nibble_to_i8(v: u8) -> i8 {
    if v >= 8 { v as i8 - 16 } else { v as i8 }
}

/// rata-rata kuadrat galat antara dua bobot sama panjang.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f32::NAN;
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc / n as f32
}

/// Kuantisasi bobot (PTQ). `data` panjang `n`; skala = `max|x|/max_value`
/// per blok (atau per tensor), `q = round(clamp(x/scale))`.
///
/// Mengembalikan nilai terkuantisasi + skala siap rekonstruksi.
pub fn quantize_tensor(
    data: &[f32],
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> Result<QuantizedTensor, VirError> {
    if data.is_empty() {
        return Err(VirError::Config("data kosong".into()));
    }
    let block_size = match granularity {
        QuantGranularity::PerTensor => data.len(),
        QuantGranularity::PerBlock { block_size } => {
            if block_size == 0 {
                return Err(VirError::Config("block_size 0 tidak valid".into()));
            }
            block_size
        }
    };
    let num_blocks = data.len().div_ceil(block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut codes: Vec<i8> = Vec::with_capacity(data.len());
    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(data.len());
        let slice = &data[start..end];
        let peak = slice.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = scheme.scale_from_peak(peak);
        scales.push(scale);
        for &x in slice {
            codes.push(quantize_one(x, scale, scheme));
        }
    }
    Ok(QuantizedTensor {
        scheme,
        granularity,
        shape: vec![data.len()],
        packed: pack_codes(&codes, scheme),
        scales,
    })
}

/// Kode bulat satu nilai (untuk fp8, konversi E4M3 penuh).
fn quantize_one(x: f32, scale: f32, scheme: QuantScheme) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    match scheme {
        QuantScheme::Fp8 => {
            let scaled = x / scale;
            let clamped = scaled.clamp(-448.0, 448.0);
            f32_to_e4m3(clamped) as i8
        }
        QuantScheme::Int8 | QuantScheme::Int4 => {
            let (lo, hi) = scheme.range();
            let q = (x / scale).round().clamp(lo, hi);
            q as i8
        }
    }
}

/// Rekonstruksi bobot float dari hasil kuantisasi (dekuantisasi).
pub fn dequantize_tensor(qt: &QuantizedTensor) -> Result<Vec<f32>, VirError> {
    let codes = qt.unpack()?;
    let n = qt.numel();
    if codes.len() != n {
        return Err(VirError::Config(format!(
            "panjang kode {} != numel {}",
            codes.len(),
            n
        )));
    }
    let mut out = Vec::with_capacity(n);
    for (i, &c) in codes.iter().enumerate() {
        let v = match qt.scheme {
            QuantScheme::Fp8 => e4m3_to_f32(c as u8),
            QuantScheme::Int8 | QuantScheme::Int4 => c as f32,
        };
        out.push(v * qt.scale_of(i));
    }
    Ok(out)
}

/// Muat ulang bobot terkuantisasi (dekuantisasi) dari berkas ekspor.
pub use crate::export::{export_quantized, load_quantized, read_export};

/// Kemas kode bulat ke byte penyimpanan (int4: 2 nilai/byte).
fn pack_codes(codes: &[i8], scheme: QuantScheme) -> Vec<u8> {
    match scheme {
        QuantScheme::Int8 | QuantScheme::Fp8 => codes.iter().map(|&c| c as u8).collect(),
        QuantScheme::Int4 => {
            let mut out = Vec::with_capacity(codes.len().div_ceil(2));
            for pair in codes.chunks(2) {
                let lo = (pair[0] as u8) & 0x0F;
                let hi = if pair.len() > 1 {
                    (pair[1] as u8) & 0x0F
                } else {
                    0
                };
                out.push(lo | (hi << 4));
            }
            out
        }
    }
}

/// Konversi f32 → bit E4M3 (round-to-nearest-even, saturasi 448).
///
/// E4M3: 1 bit tanda, 4 bit eksponen (bias 7), 3 bit mantissa.
/// - Normal: `e ∈ [1, 15]`, nilai `(1 + m/8)·2^(e−7)`.
/// - Subnormal: `e = 0`, nilai `(m/8)·2^(1−7)`.
/// - `0x7F` (NaN) tidak pernah dihasilkan: nilai > 448 di-saturasi ke
///   `0x7E` (448); NaN → 0.
fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let ax = x.abs();
    if ax == 0.0 {
        return sign;
    }
    let bits = ax.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    if f32_exp == 0xFF {
        return sign | 0x7E; // inf → saturasi
    }
    let hidden = (bits & 0x7F_FFFF) | 0x80_0000; // 1.mantissa (24 bit)
    let e = f32_exp - 120; // eksponen E4M3 (bias 7); 0 = subnormal
    if e >= 16 || (e == 15 && hidden > 0xE0_0000) {
        return sign | 0x7E; // > 448 → saturasi
    }
    if e >= 1 {
        // Normal: bulatkan 24-bit → 4-bit (1+3), nearest-even.
        let mant = round_shift(hidden >> 20, hidden & 0xF_FFFF, 1 << 19);
        return encode_e4m3(sign, e, mant);
    }
    // Subnormal: kuantum 2^(1−7−3) = 2^-9; nilai = hidden·2^(e−23),
    // sehingga jumlah kuantum = hidden >> (21 − e).
    let s = 21 - e;
    if s > 24 {
        return sign; // nilia < setengah kuantum terkecil → 0
    }
    let s = s as u32;
    let mant = round_shift(hidden >> s, hidden & ((1 << s) - 1), 1 << (s - 1));
    if mant >= 8 {
        sign | (1 << 3) // bawa ke normal terkecil (e=1, m=0)
    } else {
        sign | (mant as u8)
    }
}

/// Rakit bit E4M3 dari `mant ∈ [8, 16]` (8 = bit tersembunyi, m = mant−8).
fn encode_e4m3(sign: u8, e: i32, mant: u32) -> u8 {
    let (e, m) = if mant >= 16 {
        (e + 1, 0)
    } else {
        (e, mant - 8)
    };
    sign | ((e as u8) << 3) | (m as u8)
}

/// Pembulatan nearest-even: tie ke genap.
fn round_shift(mant: u32, rem: u32, half: u32) -> u32 {
    let up = rem > half || (rem == half && (mant & 1) == 1);
    if up { mant + 1 } else { mant }
}

/// Dekode bit E4M3 ke f32 (kebalikan [`f32_to_e4m3`]).
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as i32;
    let val = if e == 0 {
        // Subnormal: m / 8 · 2^(1-7)
        (m as f32) / 8.0 * 2f32.powi(1 - 7)
    } else if e == 15 && m == 7 {
        // 0x7F/0xFF: NaN pada spesifikasi; kita perlakukan sebagai 448.
        448.0
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    sign * val
}

/// Tabel kualitas vs ukuran untuk satu set bobot (dipakai tes + laporan).
///
/// Mengembalikan baris `(nama, bytes, mse)` untuk float32 dan tiap skema.
pub fn quality_table(data: &[f32]) -> Vec<(&'static str, usize, f32)> {
    let mut rows = vec![("float32", data.len() * 4, 0.0)];
    for scheme in [QuantScheme::Int8, QuantScheme::Int4, QuantScheme::Fp8] {
        let Some(qt) = quantize_tensor(data, scheme, QuantGranularity::PerTensor).ok() else {
            continue;
        };
        let bytes = qt.packed.len() + qt.scales.len() * 4;
        let Ok(deq) = dequantize_tensor(&qt) else {
            continue;
        };
        let name = match scheme {
            QuantScheme::Int8 => "int8",
            QuantScheme::Int4 => "int4",
            QuantScheme::Fp8 => "fp8(e4m3)",
        };
        rows.push((name, bytes, mse(data, &deq)));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_known_values() {
        // Nilai yang tepat representabel di E4M3.
        let cases: &[(f32, u8)] = &[
            (0.0, 0x00),
            (1.0, 0x38),
            (2.0, 0x40),
            (0.5, 0x30),
            (-1.0, 0xB8),
            (448.0, 0x7E),
            (-448.0, 0xFE),
            (0.001953125, 0x01), // 2^-9, subnormal terkecil
        ];
        for &(x, want) in cases {
            let got = f32_to_e4m3(x);
            assert_eq!(got, want, "f32_to_e4m3({x})");
            assert!((e4m3_to_f32(got) - x).abs() < 1e-6, "decode {x}");
        }
    }

    #[test]
    fn e4m3_saturates_large_and_nan() {
        assert_eq!(f32_to_e4m3(1000.0), 0x7E); // > 448 → 448
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3(f32::NAN), 0);
        assert_eq!(f32_to_e4m3(-1000.0), 0xFE);
    }
}
