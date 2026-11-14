//! Pipeline data: shard, stream, pack, FIM, synth.

use std::fs::File;
use std::io::{Read, Write};
use vir_common::VirError;

pub const SHARD_MAGIC: u32 = 0x56495253; // "VIRS"
pub const FIM_PRE: u32 = 999_001;
pub const FIM_SUF: u32 = 999_002;
pub const FIM_MID: u32 = 999_003;

pub fn write_shard(path: &str, tokens: &[u32]) -> Result<(), VirError> {
    let mut f = File::create(path)?;
    f.write_all(&SHARD_MAGIC.to_le_bytes())?;
    f.write_all(&(tokens.len() as u32).to_le_bytes())?;
    for t in tokens {
        f.write_all(&t.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_shard(path: &str) -> Result<Vec<u32>, VirError> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() < 8 {
        return Err(VirError::Internal("shard terlalu pendek".into()));
    }
    let magic = u32::from_le_bytes(
        buf[0..4]
            .try_into()
            .map_err(|_| VirError::Internal("magic shard".into()))?,
    );
    if magic != SHARD_MAGIC {
        return Err(VirError::Internal("magic shard salah".into()));
    }
    let count = u32::from_le_bytes(
        buf[4..8]
            .try_into()
            .map_err(|_| VirError::Internal("count shard".into()))?,
    ) as usize;
    if buf.len() < 8 + count * 4 {
        return Err(VirError::Internal("shard terpotong".into()));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = 8 + i * 4;
        out.push(u32::from_le_bytes(
            buf[off..off + 4]
                .try_into()
                .map_err(|_| VirError::Internal("token shard".into()))?,
        ));
    }
    Ok(out)
}

/// Kemas token menjadi blok panjang tetap, pisah dengan EOS (0) bila perlu.
pub fn pack_tokens(tokens: &[u32], block_size: usize) -> Vec<Vec<u32>> {
    if block_size == 0 {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(block_size);
    for &t in tokens {
        if cur.len() >= block_size {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(t);
    }
    if !cur.is_empty() {
        while cur.len() < block_size {
            cur.push(0);
        }
        out.push(cur);
    }
    out
}

/// FIM PSM: prefix, suffix, middle -> [pre] prefix [suf] suffix [mid] middle
pub fn fim_psm(prefix: &[u32], suffix: &[u32], middle: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(prefix.len() + suffix.len() + middle.len() + 3);
    out.push(FIM_PRE);
    out.extend_from_slice(prefix);
    out.push(FIM_SUF);
    out.extend_from_slice(suffix);
    out.push(FIM_MID);
    out.extend_from_slice(middle);
    out
}

/// FIM SPM: suffix, prefix, middle
pub fn fim_spm(prefix: &[u32], suffix: &[u32], middle: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(prefix.len() + suffix.len() + middle.len() + 3);
    out.push(FIM_SUF);
    out.extend_from_slice(suffix);
    out.push(FIM_PRE);
    out.extend_from_slice(prefix);
    out.push(FIM_MID);
    out.extend_from_slice(middle);
    out
}

/// Stream deterministik sederhana: urutan indeks berdasarkan seed.
pub struct TokenStream {
    data: Vec<u32>,
    pos: usize,
    stride: usize,
}

impl TokenStream {
    pub fn new(data: Vec<u32>, seed: u64, block_size: usize) -> Self {
        // shuffle deterministik sederhana (Fisher-Yates dengan LCG)
        let mut v = data;
        let n = v.len();
        let mut state = seed;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            v.swap(i, j);
        }
        Self {
            data: v,
            pos: 0,
            stride: block_size.max(1),
        }
    }

    pub fn next_block(&mut self) -> Option<Vec<u32>> {
        if self.pos >= self.data.len() {
            return None;
        }
        let end = (self.pos + self.stride).min(self.data.len());
        let block = self.data[self.pos..end].to_vec();
        self.pos = end;
        Some(block)
    }
}

/// Format SFT: instruksi -> jawaban; kemas sebagai (prefix, target).
pub fn sft_pack(instruction: &str, answer: &str) -> (String, String) {
    (instruction.to_string(), answer.to_string())
}

/// Generator sintetis: aritmetika sederhana "a + b = c".
pub fn synth_arithmetic(n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i * 7 + 3) % 20;
        let b = (i * 11 + 5) % 20;
        out.push(format!("{a} + {b} = {}", a + b));
    }
    out
}
