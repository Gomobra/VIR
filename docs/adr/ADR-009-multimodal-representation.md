# ADR-009: Representasi multimodal (gambar, audio, video)

> Melengkapi dan mengunci ADR-008 (yang hanya memuat garis besar). ADR-008 tetap
> berlaku sebagai ringkasan; ADR ini adalah keputusan rinci F6.0–F6.5.

## Konteks

Fase 6 meminta model yang sama memahami **dan** menghasilkan gambar, audio, dan
video, bukan hanya teks. Kendala yang mengikat:

1. **NCM sudah diskret dan berlevel.** Teks = token BPE, gambar/audio = kode
   diskret berlevel (kuantisasi residual + next-scale). Karena itu modalitas baru
   harus bisa masuk ke `vir-model` sebagai *urutan kode berlevel*, bukan sebagai
   tensor fitur asing. Generator sudah punya `embedding per level` + `head per
   level` (`crates/vir-model/src/lib.rs`), jadwal `ℓ_k` (F1.1), dan mask
   block-diagonal/prefix-LM (`vir-nn`, F1.5). Semua itu dapat dipakai ulang.
2. **R6: tanpa HF Tokenizers.** Tidak ada tokenizer pretrained yang diunduh;
   semua tokenizer harus kode sendiri di repo (FSQ/codec), bobot hanya lewat
   `safetensors`.
3. **Anggaran konteks nyata.** Prefix gambar/audio bersaing dengan teks di
   `P = 96–512` (F2.3) dan `L = 128–1.024` (F1.1). Jumlah token per aset harus
   bisa dihitung **sebelum** encoder dijalankan, supaya bisa dijadwalkan.
4. **Termux tidak bisa build Burn** (gemm-f16). Kode yang menyentuh tensor
   tidak boleh memaksa crate data jadi berat.
5. **Lisensi.** Data berpasangan (caption/QA) punya lisensi campur; tanpa
   metadata lisensi, data tidak boleh dipakai latih.

Pertanyaan yang harus dijawab ADR ini: *bagaimana satu aset modalitas
diwakili di dalam urutan token, bagaimana vocab diperluas, dan bagaimana data
berpasangan disimpan.*

## Keputusan

### 1. Dua jalur representasi, satu ruang ID

Setiap modalitas punya **dua** jalur, dan keduanya berakhir sebagai ID token di
satu ruang ID global:

| Jalur | Untuk | Isi |
|---|---|---|
| **Jalur kode** (diskret, berlevel) | generasi gambar/audio | kode dari tokenizer/codec sendiri (FSQ multiscale / SNAC) |
| **Jalur fitur** (encoder beku + konektor) | pemahaman | embedding kontinu dari encoder pretrained beku → konektor terlatih → diperlakukan sebagai prefix |

Keduanya tidak dibangun sekaligus. **Pemahaman (F6.1, F6.2, F6.5) memakai jalur
fitur; generasi (F6.3, F6.4) memakai jalur kode.** Alasannya: jalur fitur
mendapat kualitas tinggi tanpa melatih tokenizer dari nol, sedangkan jalur kode
adalah satu-satunya cara menghasilkan piksel/gelombang dengan arsitektur yang
sudah ada. ADR ini mengunci *bentuk* kedua jalur; bobot encoder dipilih di
F6.1/F6.2 lewat ADR terpisah (kandidat SigLIP dan Whisper belum diverifikasi).

### 2. Gambar

- **Jalur kode (generasi, F6.3):** tokenizer multiscale **FSQ/iFSQ** dengan
  decoder, dilatih rekonstruksi lebih dulu. FSQ dipilih karena bebas *codebook
  collapse* dan sudah dipakai sebagai tokenizer terpadu AR+diffusion; keputusan
  VQ-EMA vs FSQ tetap diukur di G1 (ADR-002), dan tokenizer gambar mewarisi
  pemenangnya. Level `k` memakai `Down_k`/`R_k` yang sudah ada di
  `vir-vq::multiscale`.
- **Jalur fitur (pemahaman, F6.1):** encoder visi pretrained **beku** (keluarga
  SigLIP sebagai kandidat, diverifikasi di F6.1) + **konektor terlatih**
  (proyeksi atau MLP 2 lapis). Output konektor = prefix token kontinu.
- **Anggaran token:** `N = (H/p)·(W/p)` dengan `p` ukuran patch.
  224/16 → 14×14 = **196**; 384/16 → 24×24 = **576**. `N` masuk ke `P`.

### 3. Audio

- **Jalur kode (generasi, F6.4):** **SNAC** (RVQ multiskala, stride `[4,2,1]`
  pada 24 kHz → laju 12/23/47 Hz, codebook 4096 = 12 bit). SNAC dipilih karena
  strukturnya sejajar dengan hierarki kasar→halus NCM, sehingga jadwal `ℓ_k`
  untuk audio **geometrik** (rasio 1:2:4, 3 level) — bukan linear seperti teks.
  Kandidat lain (Mimi, DAC, codec berbasis FSQ) ditunda; dibandingkan lewat ADR
  bila SNAC tidak memadai.
- **Jalur fitur (pemahaman, F6.2):** encoder ucapan pretrained **beku**
  (keluarga Whisper sebagai kandidat, diverifikasi di F6.2) + konektor terlatih.
- **Anggaran token (pemahaman):** `frame = durasi × sample_rate / hop`, dibagi
  faktor stride encoder; laju keluaran ≈ 50 posisi/detik → 10 s = **500** token.
- **Anggaran token (generasi):** 82 kode/detik → 10 s = **820** kode.
  Dengan `L = 128–1.024`, satu urutan ≈ **1,6–12,5 s** audio.
- **Pemeriksaan bitrate (wajib di CI F6.4):** `bps = Σ_k laju_k × log2|V|`.
  24 kHz: (12+23+47)×12 = **984 bps**; 32 kHz: 156×12 = **1.872**; 44 kHz:
  215×12 = **2.580**. Angka ini harus cocok dengan deklarasi codec sebelum
  dekoder dipakai.

### 4. Video (F6.5) — pemahaman saja

Video **tidak** punya jalur kode di F6.5. Frame diambil sampler (laju tetap atau
berdasarkan perubahan adegan), tiap frame masuk encoder gambar F6.1, lalu token
frame disusun berurutan dengan **penanda waktu** di antara frame. Tidak ada
encoder 3D/temporal; keputusan ini disengaja (biaya, dan F6.5 hanya butuh QA
video pendek).

### 5. Perluasan vocab dan penanda modalitas

Satu **ruang ID global** dengan empat blok berurutan:

```
[ 0 .. T )              teks: BPE byte-level (F1.2) + EOS + FIM (999_001..999_003)
[ T .. T+M )            penanda modalitas (marker)
[ T+M .. )              codebook per (kind, level): (Image,l), (Audio,l), ...
```

- **Penanda (M):** `<img>`/`</img>`, `<aud>`/`</aud>`, `<vid>`/`</vid>`, dan
  `<t=…>` untuk cap waktu video. Penanda adalah token teks biasa, sehingga LM
  memperlakukannya tanpa cabang khusus.
- **Codebook per (kind, level):** tiap pasangan `(kind, level)` mendapat rentang
  ID sepanjang `|V|` (default 4096), **diduplikasi** antar level. Ini penting:
  kode level 0 dan level 1 tidak boleh bertabrakan di tabel embedding, karena
  `vir-model` memakai embedding terpisah per level. Rentang disejajarkan ke
  kelipatan 64 agar hemat memori kernel.
- **Jalur fitur tidak memakai ID.** Vektor konektor disisipkan sebagai baris
  embedding hasil injeksi (bukan ID di vocab); jadi jalur fitur **tidak**
  memperbesar vocab, hanya memperbesar panjang urutan.
- **Migrasi checkpoint:** saat BPE dilatih ulang (`T` berubah), embedding
  di-*resize* (tambah/kurangi baris) dan rentang marker/codebook digeser
  secara deterministik oleh fungsi `VocabLayout::remap_after_text_resize`.
  Tanpa migrasi eksplisit, ID lama akan menunjuk baris yang salah.

**Urutan di dalam urutan:** *level-major* — semua kode level 0, lalu level 1,
dst. (bukan time-major/delay-pattern). Alasan: `vir-model` menghasilkan kode
level demi level, dan NCM menjadwalkan `ℓ_k` kasar→halus; level-major membuat
posisi token cocok dengan level tanpa mask tambahan.

### 6. Format data berpasangan

Satu baris JSONL per sampel (UTF-8, `serde`), minimal:

```json
{"id":"coco-000123","text":"Seekor kucing di atas kertas.","segments":[
  {"role":"user","text":"Jelaskan gambar ini."},
  {"role":"image","asset":{"kind":"image","path":"img/000123.webp","width":384,"height":384}},
  {"role":"assistant","text":"Seekor kucing ..."}],
 "license":"CC-BY-4.0","source":"coco"}
```

Aturan:
- `license` **wajib** (SPDX id, atau `unknown`). `unknown` ditolak oleh loader
  latih secara bawaan (`--allow-unknown-license` untuk kasus riset).
- Aset disimpan sebagai **referensi path** + dimensi/durasi + `sha256`; byte
  gambar tidak masuk JSONL (agar shard tetap kecil dan bisa di-dedup).
- `segments` adalah urutan asli (interleaved), bukan `text` + `assets`
  terpisah, supaya posisi modalitas di dalam percakapan tidak hilang.
- Derivasi urutan token: teks → BPE; aset jalur fitur → `N` baris embedding;
  aset jalur kode → `Σ_level len_k` ID; sisipkan penanda awal/akhir.

### 7. Crate `vir-modal`

`crates/vir-modal/` berisi **tipe data dan trait saja**, tanpa Burn dan tanpa
dependency tensor:

| Berkas | Isi |
|---|---|
| `kind.rs` | `ModalKind` (+ urutan, apakah mendukung generasi) |
| `token.rs` | `ModalToken`, `Span`, `VocabLayout` (perluasan vocab + migrasi) |
| `asset.rs` | `ModalAsset` (gambar/audio/video + metadata + validasi) |
| `sample.rs` | `Segment`, `PairedSample`, JSONL + validasi lisensi |
| `encoder.rs` | trait `ModalEncoder` (output generik, anggaran token, beku) |
| `budget.rs` | rumus anggaran token F6.1–F6.4 |

Implementasi encoder/decoder (tensor) tinggal di `vir-nn`/`vir-vq`/`vir-model`
dan mengimplementasikan trait ini. Pemisahan ini menjaga `vir-modal` bisa
`cargo check` di Termux, sementara kerja tensor tetap di CI/Kaggle (aturan R1).

## Konsekuensi

**Positif**
- Satu jalur kode untuk semua modalitas → tidak ada cabang arsitektur baru di
  `vir-model`; jadwal `ℓ_k`, mask, dan loss R6 dipakai apa adanya.
- Anggaran token bisa dihitung sebelum encoder jalan → prefix bisa dijadwalkan
  terhadap `P = 96–512` tanpa kejutan memori.
- Vocab satu ruang ID + `VocabLayout` membuat resize embedding dapat diuji
  (unit test) sebelum checkpoint menyentuh disk.
- `vir-modal` ringan: bisa dipakai alat data/CLI tanpa menarik Burn.

**Negatif / biaya**
- Jalur fitur menambah **panjang urutan** tanpa menambah koherensi jangka panjang
  pada prefix; `P` harus dijaga (mis. maksimal 1 aset per sampel di F6.1).
- Vocab membengkak: `Σ_(kind,level) |V|` baris embedding. Dengan 2 modalitas ×
  3 level × 4096 = 24.576 baris tambahan; harus masuk anggaran memori F5.5.
- Level-major memaksa seluruh level 0 selesai sebelum level 1 dimulai → tidak
  bisa streaming per-frame untuk audio; F6.5 menerima ini (video hanya
  pemahaman).
- Dua jalur (kode dan fitur) berarti dua evaluator dan dua set bobot; risiko
  divergensi kualitas antara "lihat" dan "buat".

**Risiko & mitigasi**
- *Kandidat encoder belum diverifikasi* (SigLIP, Whisper) → ADR F6.1/F6.2
  mengunci setelah verifikasi; `ModalEncoder::is_frozen()` dan `EncoderSpec`
  menyimpan `family`/`revision` agar eksperimen dapat direproduksi.
- *Codec audio tidak cocok* → bitrate check + ADR pembanding (Mimi/DAC).
- *Lisensi data* → penolakan bawaan pada `unknown`, dicatat di manifest.

**Gerbang (Definition of Done F6.0)**
1. ADR ini disetujui.
2. `vir-modal` punya tipe + validasi + tes (tanpa `todo!`/`unwrap`).
3. `VocabLayout` lulus tes: encode/decode roundtrip tiap `(kind, level)`, dan
   migrasi setelah resize teks menjaga kode lama tetap valid.
4. Anggaran token lulus tes terhadap angka PRD (196/576, 500, 820).

## Alternatif yang dipertimbangkan

1. **Hanya jalur fitur (gaya LLaVA) untuk semua modalitas.**
   *Tolak:* tidak bisa menghasilkan gambar/audio — F6.3/F6.4 mustahil.
2. **Hanya jalur kode (tanpa encoder beku) untuk pemahaman.**
   *Tolak:* tokenizer sendiri belum tentu semantik; F6.1/F6.2 butuh kualitas
   tinggi cepat, dan encoder beku memberi itu tanpa latihan tokenizer.
3. **Codebook bersama (satu rentang ID untuk semua level).**
   *Tolak:* bertabrakan dengan embedding per-level `vir-model`; memaksa
   pembagian bobot yang tidak diinginkan dan menyulitkan ablasi per level.
4. **Vocab terpisah per modalitas (tabel embedding berbeda).**
   *Tolak:* melanggar satu ruang ID; LM kehilangan kemampuan menautkan penanda
   teks dengan kode, dan checkpoint tidak bisa di-resize secara seragam.
5. **Time-major / delay-pattern (audio) dan frame-major (video).**
   *Tolak sebagai default:* tidak cocok dengan jadwal `ℓ_k` level-major; delay
   pattern menyulitkan implementasi mask. Bisa jadi opsi ablasi bila kualitas
   audio time-major terbukti lebih baik (dicatat, bukan diimplementasikan).
6. **Encoder 3D/temporal untuk video.**
   *Tolak:* F6.5 hanya QA video pendek; biaya bobot dan data tidak sebanding.
7. **Byte aset langsung di JSONL.**
   *Tolak:* JSONL membengkak, dedup mustahil, dan lisensi sulit ditelusuri.
8. **Menunda `vir-modal` sampai encoder pertama jadi.**
   *Tolak:* tipe data adalah prasyarat format data (F6.0 langkah 3); menunda
   berarti format didefinisikan dua kali. Crate tetap tanpa tensor agar aman.

## Status

Sebagian diterima: ADR disetujui untuk F6.0; `vir-modal` (tipe + trait) ada dan
lulus `cargo check`. Encoder beku (F6.1/F6.2), tokenizer gambar (F6.3), codec
audio (F6.4), dan sampler video (F6.5) **belum** diimplementasikan — masing-masing
butuh ADR turunan setelah verifikasi kandidat.