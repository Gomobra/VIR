//! Format shard biner + penulis/pembaca streaming (PRD F1.3 langkah 1).
//!
//! Header 24 byte little endian: magic `VIRS`, versi, dtype, token per blok,
//! jumlah token, jumlah blok. Tiap blok berisi `[clen u32][raw_len u32]` lalu satu
//! frame zstd. Pembaca membaca indeks blok sekali (tanpa dekompresi) lalu tiap blok
//! didekompresi sendiri, sehingga memori terpakai sebanding satu blok.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use vir_common::VirError;

/// Magic berkas shard: `VIRS`.
pub const SHARD_MAGIC: u32 = 0x5649_5253;
/// Versi format shard.
pub const SHARD_VERSION: u16 = 1;
/// Ukuran header dalam byte.
pub const HEADER_BYTES: usize = 24;
/// Token per blok bila penulis tidak menentukan.
pub const DEFAULT_BLOCK_TOKENS: u32 = 1024;
/// Level kompresi zstd.
pub const ZSTD_LEVEL: i32 = 3;

/// Panjang prefiks blok: `clen` + `raw_len`.
const BLOCK_PREFIX_BYTES: u64 = 8;

/// Jenis data token di dalam shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// Token `u32` little endian.
    U32,
}

impl DType {
    /// Kode satu byte di header.
    pub const fn code(self) -> u8 {
        match self {
            Self::U32 => 0,
        }
    }

    /// Ukuran satu token dalam byte.
    pub const fn token_size(self) -> u64 {
        match self {
            Self::U32 => 4,
        }
    }

    /// Uraikan kode header.
    pub fn parse(code: u8) -> Result<Self, VirError> {
        match code {
            0 => Ok(Self::U32),
            other => Err(VirError::Internal(format!(
                "dtype shard 0x{other:02x} tidak dikenal"
            ))),
        }
    }
}

/// Header shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardHeader {
    pub version: u16,
    pub dtype: DType,
    pub block_tokens: u32,
    pub token_count: u64,
    pub block_count: u32,
}

impl ShardHeader {
    /// `true` bila versi dan ukuran blok valid (shard kosong tetap sah).
    pub fn is_consistent(&self) -> bool {
        self.version == SHARD_VERSION && self.block_tokens > 0
    }
}

/// Penulis shard: tulis token, lalu `finish` untuk menutup header.
pub struct ShardWriter {
    file: File,
    header: ShardHeader,
    pending: Vec<u32>,
    written: u64,
}

impl ShardWriter {
    /// Buat shard baru dengan `block_tokens` token per blok.
    pub fn create(path: impl AsRef<Path>, block_tokens: u32) -> Result<Self, VirError> {
        let block_tokens = if block_tokens == 0 {
            DEFAULT_BLOCK_TOKENS
        } else {
            block_tokens
        };
        let header = ShardHeader {
            version: SHARD_VERSION,
            dtype: DType::U32,
            block_tokens,
            token_count: 0,
            block_count: 0,
        };
        let mut file = File::create(path)?;
        write_header(&mut file, &header)?;
        Ok(Self {
            file,
            header,
            pending: Vec::with_capacity(block_tokens as usize),
            written: 0,
        })
    }

    /// Tambahkan token; blok ditulis penuh setiap `block_tokens` token.
    pub fn write_tokens(&mut self, tokens: &[u32]) -> Result<(), VirError> {
        for token in tokens {
            self.pending.push(*token);
            self.written += 1;
            if self.pending.len() >= self.header.block_tokens as usize {
                self.flush_block()?;
            }
        }
        Ok(())
    }

    /// Tulis blok tertunda sebagai satu frame zstd.
    fn flush_block(&mut self) -> Result<(), VirError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let raw = tokens_to_bytes(&self.pending);
        let compressed = compress(&raw)?;
        self.file
            .write_all(&(compressed.len() as u32).to_le_bytes())?;
        self.file.write_all(&(raw.len() as u32).to_le_bytes())?;
        self.file.write_all(&compressed)?;
        self.header.block_count += 1;
        self.pending.clear();
        Ok(())
    }

    /// Tutup shard: sisa token ditulis, header diperbarui di awal berkas.
    pub fn finish(mut self) -> Result<ShardHeader, VirError> {
        self.flush_block()?;
        self.header.token_count = self.written;
        self.file.seek(SeekFrom::Start(0))?;
        write_header(&mut self.file, &self.header)?;
        self.file.flush()?;
        if !self.header.is_consistent() {
            return Err(VirError::Internal("header shard tidak konsisten".into()));
        }
        Ok(self.header)
    }

    /// Header sementara (token_count belum final sebelum `finish`).
    pub fn header(&self) -> &ShardHeader {
        &self.header
    }
}

/// Pembaca shard: indeks blok di memori, dekompresi satu blok setiap kali.
pub struct ShardReader {
    file: File,
    header: ShardHeader,
    index: Vec<(u64, u32, u32)>,
}

impl ShardReader {
    /// Buka shard dan baca indeks blok.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VirError> {
        let mut file = File::open(path)?;
        let header = read_header(&mut file)?;
        let file_len = file.seek(SeekFrom::End(0))?;
        // Batasi jumlah blok agar header rusak tidak memicu alokasi raksasa.
        let max_blocks = (file_len.saturating_sub(HEADER_BYTES as u64)) / BLOCK_PREFIX_BYTES;
        if u64::from(header.block_count) > max_blocks {
            return Err(VirError::Internal(format!(
                "jumlah blok {} mustahil untuk berkas {file_len} byte",
                header.block_count
            )));
        }
        let mut index = Vec::with_capacity(header.block_count as usize);
        let mut pos = HEADER_BYTES as u64;
        for _ in 0..header.block_count {
            let (compressed_len, raw_len) = read_block_prefix(&mut file, pos)?;
            pos += BLOCK_PREFIX_BYTES;
            if pos + u64::from(compressed_len) > file_len {
                return Err(VirError::Internal("blok shard terpotong".into()));
            }
            index.push((pos, compressed_len, raw_len));
            pos += u64::from(compressed_len);
        }
        if pos != file_len {
            return Err(VirError::Internal(format!(
                "panjang shard {file_len} tidak cocok dengan indeks {pos}"
            )));
        }
        Ok(Self {
            file,
            header,
            index,
        })
    }

    /// Header shard.
    pub fn header(&self) -> &ShardHeader {
        &self.header
    }

    /// Jumlah blok.
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Jumlah token menurut header.
    pub fn token_count(&self) -> u64 {
        self.header.token_count
    }

    /// Baca satu blok berdasarkan indeks.
    pub fn block(&mut self, index: usize) -> Result<Vec<u32>, VirError> {
        let Some((offset, compressed_len, raw_len)) = self.index.get(index).copied() else {
            return Err(VirError::Internal(format!("blok {index} di luar indeks")));
        };
        self.file.seek(SeekFrom::Start(offset))?;
        let mut compressed = vec![0u8; compressed_len as usize];
        self.file.read_exact(&mut compressed)?;
        let raw = decompress(&compressed, raw_len as usize)?;
        bytes_to_tokens(&raw)
    }

    /// Seluruh token (semua blok).
    pub fn tokens(&mut self) -> Result<Vec<u32>, VirError> {
        let mut out = Vec::with_capacity(self.header.token_count as usize);
        for index in 0..self.index.len() {
            out.extend_from_slice(&self.block(index)?);
        }
        Ok(out)
    }

    /// Jalankan `f` untuk tiap blok berurutan (tanpa menampung semuanya).
    pub fn for_each_block<F>(&mut self, mut f: F) -> Result<(), VirError>
    where
        F: FnMut(&[u32]) -> Result<(), VirError>,
    {
        for index in 0..self.index.len() {
            let block = self.block(index)?;
            f(&block)?;
        }
        Ok(())
    }

    /// Periksa jumlah token hasil dekompresi sama dengan header.
    pub fn verify(&mut self) -> Result<(), VirError> {
        let mut total = 0u64;
        self.for_each_block(|block| {
            total += block.len() as u64;
            Ok(())
        })?;
        if total != self.header.token_count {
            let expected = self.header.token_count;
            return Err(VirError::Internal(format!(
                "token shard {total} != header {expected}"
            )));
        }
        Ok(())
    }
}

/// Tulis shard sederhana: satu blok berisi seluruh token.
pub fn write_shard(path: impl AsRef<Path>, tokens: &[u32]) -> Result<ShardHeader, VirError> {
    let block_tokens = tokens.len().max(1) as u32;
    let mut writer = ShardWriter::create(path, block_tokens)?;
    writer.write_tokens(tokens)?;
    writer.finish()
}

/// Baca seluruh token dari shard.
pub fn read_shard(path: impl AsRef<Path>) -> Result<Vec<u32>, VirError> {
    let mut reader = ShardReader::open(path)?;
    reader.tokens()
}

/// Token `u32` → byte little endian.
pub fn tokens_to_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        out.extend_from_slice(&token.to_le_bytes());
    }
    out
}

/// Byte little endian → token `u32`.
pub fn bytes_to_tokens(bytes: &[u8]) -> Result<Vec<u32>, VirError> {
    let len = bytes.len();
    if len % 4 != 0 {
        return Err(VirError::Internal(format!(
            "panjang blok {len} bukan kelipatan 4"
        )));
    }
    let mut out = Vec::with_capacity(len / 4);
    for chunk in bytes.chunks_exact(4) {
        let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(u32::from_le_bytes(raw));
    }
    Ok(out)
}

fn compress(raw: &[u8]) -> Result<Vec<u8>, VirError> {
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    encoder.write_all(raw)?;
    Ok(encoder.finish()?)
}

fn decompress(compressed: &[u8], raw_len: usize) -> Result<Vec<u8>, VirError> {
    let mut decoder = zstd::stream::read::Decoder::new(compressed)?;
    let mut out = Vec::with_capacity(raw_len);
    decoder.read_to_end(&mut out)?;
    if out.len() != raw_len {
        let got = out.len();
        return Err(VirError::Internal(format!(
            "blok dekompresi {got} != raw_len {raw_len}"
        )));
    }
    Ok(out)
}

fn write_header(file: &mut File, header: &ShardHeader) -> Result<(), VirError> {
    let mut buf = [0u8; HEADER_BYTES];
    buf[0..4].copy_from_slice(&SHARD_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&header.version.to_le_bytes());
    buf[6] = header.dtype.code();
    buf[7] = 0;
    buf[8..12].copy_from_slice(&header.block_tokens.to_le_bytes());
    buf[12..20].copy_from_slice(&header.token_count.to_le_bytes());
    buf[20..24].copy_from_slice(&header.block_count.to_le_bytes());
    file.write_all(&buf)?;
    Ok(())
}

fn read_header(file: &mut File) -> Result<ShardHeader, VirError> {
    let mut buf = [0u8; HEADER_BYTES];
    file.read_exact(&mut buf)?;
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != SHARD_MAGIC {
        return Err(VirError::Internal(format!(
            "magic shard 0x{magic:08x} bukan VIRS"
        )));
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    let dtype = DType::parse(buf[6])?;
    let block_tokens = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let token_count = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
    ]);
    let block_count = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let header = ShardHeader {
        version,
        dtype,
        block_tokens,
        token_count,
        block_count,
    };
    if !header.is_consistent() {
        let version = header.version;
        return Err(VirError::Internal(format!(
            "header shard tidak sah: versi {version}, blok {block_tokens}, token {token_count}"
        )));
    }
    Ok(header)
}

fn read_block_prefix(file: &mut File, pos: u64) -> Result<(u32, u32), VirError> {
    let mut buf = [0u8; BLOCK_PREFIX_BYTES as usize];
    file.seek(SeekFrom::Start(pos))?;
    file.read_exact(&mut buf)?;
    let compressed_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let raw_len = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if raw_len == 0 || raw_len % 4 != 0 {
        return Err(VirError::Internal(format!(
            "raw_len blok {raw_len} tidak sah"
        )));
    }
    Ok((compressed_len, raw_len))
}
