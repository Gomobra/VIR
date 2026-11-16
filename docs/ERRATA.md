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

## E-009 — train_bpe: `positions` kehilangan indeks kata (frekuensi & hasil latih salah)
- Gejala: `vir-text::roundtrip latih_deterministik_dan_tidak_bergantung_urutan` gagal
  setelah E-008 diperbaiki (job nextest run 43688617560). Latih korpus KORPUS = 85
  merge; korpus yang sama dengan urutan kalimat dibalik = 40 merge.
- Akar masalah: `PairIndex::positions` menyimpan `(node_kiri, node_kanan)` dari
  `LocalIndex` (yang per kata) tanpa indeks kata, tetapi `apply` membacanya sebagai
  `(indeks_kata, node_kiri)` dan `merge_at` mencatat posisi baru sebagai `(p, i)`
  (dua indeks node). Akibatnya merge diterapkan ke kata yang salah (atau ditolak),
  frekuensi pasangan tidak pernah turun, pasangan yang sudah di-merge bisa muncul
  lagi (`push_merge` gagal → latih berhenti dini), dan hasilnya bergantung urutan
  korpus. Reproduksi: `repro_train.rs` (salinan struktur train.rs + pretok asli,
  `rustc` satu berkas) → 85 vs 40 merge; setelah `positions` menyimpan
  `(kata, kiri)` → 60 vs 60 merge, identik.
- Perbaikan: `crates/vir-text/src/train.rs` (text-eng) — `positions` menyimpan
  `(indeks kata, node kiri)`; alternatif yang dipakai text-eng: buang indeks
  inkremental, hitung ulang frekuensi tiap iterasi (`count_pairs` + `apply_merge`).
  Kedua jalur terverifikasi memberi angka sama: KORPUS 60 merge (maju = mundur),
  `synth_corpus(8)` 49, `synth_corpus(6)` 29, `synth_sentences(4)` 15,
  `"abab abab"` 2, `"ab ab ab"` 2, KORPUS banana 6.
  Catatan: `best_pair` versi in-progress memakai `max_by(|a, b| b.1.cmp(a.1)...)` —
  komparatornya terbalik (memilih frekuensi terkecil) sehingga latih berhenti di
  iterasi pertama; arah yang benar `a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))`.
- Tes regresi: `latih_deterministik_dan_tidak_bergantung_urutan` (sudah ada, kini
  benar-benar menguji), `vocab_susut_saat_korpus_kecil` (ambang `>= 10` → `>= 2`:
  korpus 8 byte maksimal 2 merge nyata; ambang lama hanya tercapai bila frekuensi
  salah hitung), `indeks_pasangan_sesuai_hitung_ulang`.
- Commit: (debugger, tes) + (text-eng, train.rs)

## E-010 — vir-tui tanpa tes nyata; `NO_COLOR` dan Ctrl+P tidak ter-wiring
- Gejala: `docs/PROGRESS.md` mengklaim "TUI multi-screen ... snapshot 3 ukuran ...
  CI hijau" padahal `crates/vir-tui` tidak punya satu pun tes, sehingga F3.8.h
  (snapshot 3 ukuran, fuzz input, PTY) tidak pernah dijalankan. `NO_COLOR` hanya
  ada sebagai `detect_mode(bool)` yang tidak pernah dipanggil, dan `Ctrl+P`
  kedua saat palet terbuka justru mengetik `p` ke query sehingga palet tidak
  pernah tertutup (F3.8: palet lewat `Ctrl+P` atau `:`).
- Akar masalah: crate TUI ditulis tanpa berkas `tests/` (0 tes → job `nextest`
  hijau tanpa mengeksekusi apa pun) dan tanpa menyambung mode warna efektif ke
  `App::palette()`; cabang `Ctrl+P` berada di bawah cabang palet pada `on_key`.
- Perbaikan: `crates/vir-tui/tests/` (render 3 ukuran, fuzz proptest, layout
  responsif, tema/`NO_COLOR`, state `update`, anggaran render, PTY);
  `theme.rs` (`no_color`, `mode_for`, `resolve_mode`); `app.rs` (`palette()` dan
  `on_key` memakai mode warna efektif serta menangani `Ctrl+P` lebih dulu).
  `scripts/forbid_stubs.sh`: `vir-tui` tidak lagi di-allowlist "tanpa tes".
- Tes regresi: `snapshot_tiap_layar_tiga_ukuran`,
  `input_acak_tidak_panik_dan_invariant_terjaga`,
  `pty_no_color_mematikan_warna_truecolor`, `ctrl_p_tidak_mengetik_p_ke_query`,
  `no_color_selalu_menang_atas_setelan`.
- Commit: (ui-eng)

## E-011 — vir-vq tidak punya tes CPU yang berjalan (klaim "50+ tes" palsu)
- Gejala: `docs/PROGRESS.md` mengklaim "F1.4 vir-vq lengkap ... 50+ tes lulus di
  CI", padahal satu-satunya berkas tes (`src/kernel.rs`) memakai
  `#[cfg(all(test, feature = "cuda"))]`; job CI `vir-vq` menjalankan
  `cargo test` (default, tanpa cuda) sehingga **0 tes** berjalan. Gerbang R7
  (crate punya tes yang dijalankan) tidak terpenuhi.
- Akar masalah: logika host kernel terfusi (`plan_launch`, `naive_argmin`,
  `output_shape`, `output_dtype`, `args_compatible`) bercampur dengan modul
  `#[cfg(feature = "fused")]` yang tak bisa dikompilasi di CPU tanpa cuda.
- Perbaikan: `src/fused_plan.rs` memisahkan logika sisi-host (selalu dikompilasi)
  dari `src/kernel.rs` (feature `fused`); `crates/vir-vq/tests/`
  (codebook, multiscale, fsq, metrics, fused_plan) — puluhan tes CPU nyata.
  `scripts/forbid_stubs.sh`: `vir-vq` tidak lagi di-allowlist "tanpa tes".
- Tes regresi: `naive_argmin_sama_dengan_quantize_reference`,
  `argmin_memilih_entri_terdekat`, `fsq_indices_dalam_rentang`,
  `ukuran_codebook_adalah_produk_level`, `akurasi_token_numerik`,
  `broadcast_dari_satu_posisi`.
- Commit: (vq-eng)

## E-012 — `Codebook::set_entries` tidak menyetel ulang state EMA
- Gejala: inisialisasi codebook dari k-means / nilai tetap (`set_entries`) langsung
  rusak pada pembaruan EMA pertama, karena `ema_sum`/`ema_count` masih menyimpan
  init acak `N(0,1)` sehingga `e = m/N` mencampur bobot lama dan baru.
- Akar masalah: `set_entries` hanya mengganti `weight`, tidak mereset buffer EMA.
- Perbaikan: `set_entries` kini menyetel `ema_sum = e` dan `ema_count = 1`.
- Tes regresi: `set_entries_menepati_ema_state`, `ema_update_av_trigger_count_hitung_developer`.
- Commit: (vq-eng)

## E-013 — `multiscale::interp_matrix` panik saat `in_len == 1`
- Gejala: `ScaleOps::new(1, ℓ)` (broadcast satu posisi ke ℓ) panik di
  `TensorData` karena `interp_matrix(1, ℓ)` mengembalikan `[[1.0]]` (panjang 1),
  bukan `[1, ℓ]`.
- Akar masalah: kasus `in_len == 1` tidak ditangani khusus di `interp_matrix`.
- Perbaikan: `interp_matrix` mengembalikan `[1, ℓ]` berisi satu saat `in_len == 1`.
- Tes regresi: `broadcast_dari_satu_posisi`.
- Commit: (vq-eng)

## E-014 — `Fsq::dim()` mengembalikan jumlah level, bukan dimensi laten
- Gejala: `Quantizer::dim()` pada `Fsq` mengembalikan `n = levels.len()` (mis. 5),
  sehingga `MultiscaleVq<Fsq>` membentuk `Phi`/kode pada dimensi salah dan
  `Fsq::new(cfg, d)` mengabaikan `d`.
- Akar masalah: `dim()` salah kembalikan panjang `levels`.
- Perbaikan: `Fsq` menyimpan `dim: usize` dan `Quantizer::dim()` mengembalikan `d`.
- Tes regresi: `fsq_dim_adalah_dimensi_laten`, `fsq_roundtrip_depan_belakang_konsisten`.
- Commit: (vq-eng)

## E-015 — Landlock tanpa `workdir` kehilangan konfinemen diam-diam
- Gejala: `Policy::default()` (landlock = true) tanpa `workdir` tetap berjalan
  tanpa pembatasan sistem berkas apa pun; pemanggil menyangka konfinemen kernel
  aktif padahal `wire_landlock` hanya menambal aturan bila `workdir` ada. Pada
  jalur yang sama, `confinemen_error()` di tes isolasi memakai `.ok()?`, sehingga
  galat pemasangan konfinemen (spawn anak gagal) terbaca sebagai "tidak ada
  galat" — kegagalan sandbox bisa lewat sebagai sukses.
- Akar masalah: aturan tulis Landlock berbasis `workdir` + `allow_write`, jadi
  tanpa `workdir` tidak ada sasaran aturan tulis. Kode melewati konfinemen tanpa
  memberi tanda, dan deteksi kemampuan kernel menelan galat spawn.
- Perbaikan: `build_command` memeriksa kebijakan lebih dulu (E-015) — dengan
  fitur `sandbox`, `landlock = true` tanpa `workdir` ditolak dengan
  `VirError::Config` (fail closed); `confinemen_error()` melaporkan galat
  pemasangan sebagai teks, bukan `None`.
- Tes regresi: `landlock_tanpa_workdir_ditolak` (vir-sandbox, hanya fitur
  `sandbox`), `tanpa_landlock_tanpa_workdir_tetap_jalan`, dan
  `konfinemen_anak_terpasang` yang kini benar-benar gagal bila pemasangan
  konfinemen error.
- Commit: (agent-eng)
