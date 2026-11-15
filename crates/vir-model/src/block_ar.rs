//! Loop blok (F2.3): `generate` blok → `decode` → tambah prefix → blok berikut.
//!
//! PRD F2.3 langkah 3: "generate blok (semua level) → decode → tambahkan ke
//! prefix → blok berikutnya; berhenti di EOS atau batas blok." Biaya posisi per
//! blok tanpa cache adalah `K·P + Σℓ_k` (F2.3 langkah 2); karena prompt tidak
//! melihat target dan embedding prompt tetap, KV prompt identik antar level
//! sehingga cukup dihitung sekali (P + Σℓ_k).
//!
//! Pembungkus ini bebas dari model: pemanggil memberi `generate` (satu blok +
//! metrik posisi) dan `decode` (blok → token). Prefix dipegang pemanggil dan
//! dilaporkan lewat [`BlockMeta::next_prefix_len`], sehingga blok berikutnya
//! sudah melihat blok sebelumnya.

use vir_common::VirError;

/// Konfigurasi loop blok.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockArConfig {
    /// Batas jumlah blok.
    pub max_blocks: usize,
    /// Batas jumlah token keluaran.
    pub max_tokens: usize,
    /// Token penanda akhir blok; keluaran berhenti setelah token ini.
    pub eos: u32,
}

impl BlockArConfig {
    /// Batas blok 8, token 4096, EOS = `u32::MAX` (tidak pernah kena).
    pub fn new() -> Self {
        Self {
            max_blocks: 8,
            max_tokens: 4096,
            eos: u32::MAX,
        }
    }

    fn validate(&self) -> Result<(), VirError> {
        if self.max_blocks == 0 {
            return Err(VirError::Config("max_blocks harus > 0".into()));
        }
        if self.max_tokens == 0 {
            return Err(VirError::Config("max_tokens harus > 0".into()));
        }
        Ok(())
    }
}

impl Default for BlockArConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrik satu blok.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockMeta {
    /// Posisi yang diproses generator pada blok ini (`Σ ℓ_k`).
    pub positions: usize,
    /// Panjang konteks blok ini (`P + L`), yaitu posisi yang dilihat decoder.
    pub decode_positions: usize,
    /// Panjang prefix untuk blok berikutnya (prefix + blok ini).
    pub next_prefix_len: usize,
}

impl BlockMeta {
    /// Metrik blok kosong.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Biaya loop blok yang diakumulasi.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockArCost {
    /// Blok yang dijalankan.
    pub blocks: usize,
    /// Token keluaran (tanpa EOS).
    pub tokens: usize,
    /// Posisi generator terakumulasi (`Σ Σ ℓ_k`).
    pub positions: usize,
    /// Posisi decoder terakumulasi (`Σ (P + L)`).
    pub decode_positions: usize,
    /// `true` bila keluaran berhenti karena EOS.
    pub hit_eos: bool,
}

/// Loop blok (F2.3 langkah 3).
///
/// `generate(prefix_len)` menghasilkan satu blok beserta metriknya; `decode`
/// memetakan blok ke token. Prefix bertambah tiap blok lewat
/// [`BlockMeta::next_prefix_len`].
pub fn block_ar<B, G, D>(
    cfg: &BlockArConfig,
    mut generate: G,
    mut decode: D,
) -> Result<(Vec<u32>, BlockArCost), VirError>
where
    G: FnMut(usize) -> Result<(B, BlockMeta), VirError>,
    D: FnMut(&B) -> Result<Vec<u32>, VirError>,
{
    cfg.validate()?;
    let mut prefix_len = 0usize;
    let mut tokens: Vec<u32> = Vec::new();
    let mut cost = BlockArCost::default();

    while cost.blocks < cfg.max_blocks && tokens.len() < cfg.max_tokens {
        let (block, meta) = generate(prefix_len)?;
        cost.blocks += 1;
        cost.positions += meta.positions;
        cost.decode_positions += meta.decode_positions;
        prefix_len = meta.next_prefix_len;

        let decoded = decode(&block)?;
        let mut hit_eos = false;
        for token in decoded {
            if token == cfg.eos {
                hit_eos = true;
                break;
            }
            if tokens.len() >= cfg.max_tokens {
                break;
            }
            tokens.push(token);
        }
        cost.hit_eos = hit_eos;
        if hit_eos {
            break;
        }
    }
    Ok((tokens, cost))
}
