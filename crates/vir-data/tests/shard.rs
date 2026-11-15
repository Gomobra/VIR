//! Tes format shard: penulis, pembaca, kompresi, dan konversi byte (PRD F1.3).

use vir_data::{
    DType, HEADER_BYTES, SHARD_MAGIC, SHARD_VERSION, ShardHeader, ShardReader, ShardWriter,
    bytes_to_tokens, read_shard, tokens_to_bytes, write_shard,
};

fn temp_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vir-data-shard-{name}-{pid}.shard"))
}

#[test]
fn tulis_baca_multi_blok() {
    let path = temp_path("multi");
    let tokens: Vec<u32> = (0..2500).collect();
    let mut writer = ShardWriter::create(&path, 1000).expect("buat penulis");
    assert_eq!(writer.header().block_tokens, 1000);
    writer.write_tokens(&tokens).expect("tulis token");
    let header = writer.finish().expect("tutup shard");
    assert_eq!(header.block_count, 3);
    assert_eq!(header.token_count, 2500);
    assert_eq!(header.version, SHARD_VERSION);
    assert_eq!(header.dtype, DType::U32);
    assert!(header.is_consistent());

    let mut reader = ShardReader::open(&path).expect("buka pembaca");
    assert_eq!(reader.header().token_count, 2500);
    assert_eq!(reader.token_count(), 2500);
    assert_eq!(reader.block_count(), 3);
    assert_eq!(reader.tokens().expect("baca token"), tokens);
    reader.verify().expect("verifikasi");
    assert_eq!(reader.block(0).expect("blok 0").len(), 1000);
    assert!(reader.block(3).is_err());
    assert!(
        reader
            .for_each_block(|block| {
                assert!(!block.is_empty());
                Ok(())
            })
            .is_ok()
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn shard_kosong_dan_kecil() {
    let path = temp_path("kecil");
    let empty = write_shard(&path, &[]).expect("shard kosong");
    assert_eq!(empty.token_count, 0);
    assert_eq!(empty.block_count, 0);
    assert_eq!(read_shard(&path).expect("baca kosong"), Vec::<u32>::new());
    let header = write_shard(&path, &[7, 8, 9]).expect("shard kecil");
    assert_eq!(header.token_count, 3);
    assert_eq!(read_shard(&path).expect("baca kecil"), vec![7, 8, 9]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn blok_tidak_penuh_tetap_dibaca() {
    let path = temp_path("sisa");
    let tokens: Vec<u32> = (0..1500).collect();
    let mut writer = ShardWriter::create(&path, 0).expect("blok bawaan");
    writer.write_tokens(&tokens).expect("tulis");
    let header = writer.finish().expect("tutup");
    assert_eq!(header.block_tokens, 1024);
    assert_eq!(header.block_count, 2);
    assert_eq!(read_shard(&path).expect("baca"), tokens);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn berkas_korup_dan_terpotong_ditolak() {
    let path = temp_path("korup");
    std::fs::write(&path, b"ini bukan shard").expect("tulis");
    assert!(ShardReader::open(&path).is_err());
    let magic = SHARD_MAGIC.to_le_bytes();
    std::fs::write(&path, magic).expect("tulis magic saja");
    assert!(ShardReader::open(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn konversi_byte() {
    let tokens = vec![0u32, 1, 258, u32::MAX];
    let bytes = tokens_to_bytes(&tokens);
    assert_eq!(bytes.len(), 16);
    assert_eq!(bytes_to_tokens(&bytes).expect("balik"), tokens);
    assert!(bytes_to_tokens(&[1, 2, 3]).is_err());
    assert_eq!(HEADER_BYTES, 24);
}

#[test]
fn dtype_dan_header_konsisten() {
    assert_eq!(DType::parse(0).ok(), Some(DType::U32));
    assert!(DType::parse(9).is_err());
    let header = ShardHeader {
        version: SHARD_VERSION,
        dtype: DType::U32,
        block_tokens: 4,
        token_count: 8,
        block_count: 2,
    };
    assert!(header.is_consistent());
    let rusak = ShardHeader {
        block_tokens: 0,
        ..header
    };
    assert!(!rusak.is_consistent());
}
