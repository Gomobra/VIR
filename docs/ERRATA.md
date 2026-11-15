# ERRATA — VIR Vbeta

Format entri:
## E-XXX — judul singkat
- Gejala:
- Akar masalah:
- Perbaikan:
- Tes regresi:
- Commit:

## E-001 — (pantau) Rekonstruksi eksak
- Gejala: akurasi level penuh < 99.8% pada korpus tes.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e001_*
- Commit: -

## E-002 — (pantau) Jumlah level tetap
- Gejala: halting tidak adaptif, biaya boros.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e002_*
- Commit: -

## E-003 — (pantau) Ketidakcocokan latihan dan inferensi
- Gejala: mode latih (corruption) vs inferensi (bersih) tidak sinkron.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e003_*
- Commit: -

## E-004 — (pantau) Utilisasi codebook rendah
- Gejala: utilisasi < 60% pada level tertentu.
- Akar masalah: (belum)
- Perbaikan: (belum)
- Tes regresi: regression_e004_*
- Commit: -

## E-005 — Koreksi PRD: SNAC untuk teks
- Gejala: PRD F1.4.b menyebut SNAC sebagai kandidat Down/R_k untuk teks.
- Akar masalah: salah tafsir; kode NCM memakai `F.interpolate(area/linear)` untuk teks, SNAC hanya audio.
- Perbaikan: PRD F1.4.b diperbarui; ADR-003 dicatat.
- Tes regresi: N/A (dokumentasi)
- Commit: a50dcbf

## E-006 — kda_step kehilangan suku rank-1
- Gejala: `kda_step` lama hanya menerapkan diagonal `I − βkkᵀ`, kehilangan `−βk(α⊙k)ᵀS`.
- Akar masalah: implementasi tidak sesuai rumus PRD F5.2.
- Perbaikan: `kda_step` diperbaiki + `kda_step_chunkwise` (UT) + tes kesetaraan.
- Tes regresi: `kda_chunkwise_matches_naive`
- Commit: (nn-eng)

## E-007 — fallback fused_argmin_bias mengabaikan bias
- Gejala: fallback `argmax(z·eᵀ)` bukan `argmax(z·eᵀ − ½‖e‖²)`.
- Akar masalah: bias tidak ditambahkan di jalur fallback.
- Perbaikan: `+ bias.reshape([1, codes])` + validasi bentar/device.
- Tes regresi: `fallback_applies_bias`
- Commit: (nn-eng)
## E-008 — nextest menggantung >25 menit: iterator epoch tanpa batas
- Gejala: job `nextest` (run 43685595689) di-cancel setelah 30 menit; log terakhir
  `SLOW [>1380s] vir-data::pipeline pipeline_tokenizer_ke_shard_stream_fim`.
  Job lain (clippy, vir-model, vir-nn, vir-train, vir-vq, vir-eval, vir-modal, e2e) hijau.
- Akar masalah (dua, terpisah):
  1. `TokenStream` sengaja tak berujung (epoch berulang), tetapi tes
     `pipeline_tokenizer_ke_shard_stream_fim` memakainya sebagai iterator hingga
     `None` (`while let Some(block) = stream.next_block()`) → loop tak berhingga.
  2. `vir-text export::tests::ekspor_impor_identik` gagal: baris token kosong
     (slot cadangan modalitas, base64 `""`) diekspor sebagai `" <id>"`, lalu
     `load_tiktoken` memanggil `line.trim()` sehingga spasi pembuka hilang,
     `rsplit_once(' ')` = `None`, dan impor mengembalikan `Err`. Di baliknya,
     `MergeTable::from_vocab` juga mustahil memulihkan merge dari byte token
     (split ambigu; encode dengan tabel parsial tak pernah menghasilkan merge
     peringkat berikutnya).
- Perbaikan:
  - `crates/vir-data/tests/pipeline.rs`: batasi konsumsi stream ke satu epoch
    (`epoch_len()`), plus assert epoch berikutnya masih mengalir.
  - `crates/vir-text/src/export.rs`: impor tak lagi `trim` (hanya lepas `\r`),
    token kosong utuh; ekspor menulis blok merge `#merge <kiri> <kanan>` dan
    impor memulihkannya lewat `MergeTable::from_vocab_merges` (lossless).
  - `crates/vir-text/src/bpe.rs`: `from_vocab` (tanpa merge) + `from_vocab_merges`
    (validasi tata letak + penerapan merge berperingkat); `recover_merges` yang
    keliru dibuang. Catatan: vocab impor sudah memuat token hasil merge, jadi
    `apply_merges` memisahkannya (`split_off`) lalu membandingkan byte hasil
    `push_merge` dengan token impor pada id yang sama; sisa token tanpa merge
    (berkas tiktoken pihak lain) dipertahankan.
  - `.config/nextest.toml`: `slow-timeout = { period = "60s", terminate-after = 2 }`
    agar tes yang menggantung gagal cepat dengan namanya, bukan memakan 30 menit.
- Tes regresi: `token_kosong_tidak_terpotong_saat_impor`,
  `merge_tak_berurutan_ditolak`, `ekspor_impor_tiktoken_setara` (baris token vs
  baris merge), `pipeline_tokenizer_ke_shard_stream_fim` (batas satu epoch).
- Commit: b0a97c9
