# ADR-001 — `vir-mfs`: memori sebagai berkas berpohon

Status: diterima (Vbeta). Ruang lingkup: PRD F3.3.

## Konteks

Memori agen perlu bisa dibuka, dicari, ditautkan, dan ditelusuri riwayatnya tanpa
pernah kehilangan isi mentah. Vbeta: satu pengguna, lokal, persisten antar sesi.

## Keputusan

### Unit dan ruang nama

- Unit memori adalah **berkas** dengan path mutlak berpohon, mis. `/semantik/a`.
  Path dinormalkan (tanpa segmen kosong/`.`; `..` memangkas).
- **Ruang nama** adalah segmen pertama path: `/episodik`, `/semantik`,
  `/prosedural`, `/kerja`. Default bila path tidak diberi ruang nama diturunkan
  dari segmen pertama; root memakai `akar`.
- Setiap path memiliki banyak **versi**. Versi adalah baris `versions` yang tidak
  pernah diperbarui isinya; `write` selalu `INSERT`.

### Aturan isi tak berubah

- Kolom `versions.content` hanya pernah ditulis sekali saat `INSERT`. Tidak ada
  `UPDATE`/`DELETE` terhadapnya di crate ini.
- `versions.hash` adalah SHA-256 isi. `read` menghitung ulang hash dan menolak
  bila berbeda (deteksi korupsi).
- `forget` memasang `deleted = 1`; `restore` mengembalikan `0`. Keduanya tidak
  menyentuh isi. Pencarian mengecualikan versi terhapus kecuali
  `include_forgotten`.
- Membaca hanya menambah baris `access_log` (tabel terpisah). Isi dan versi
  tidak berubah.

### Relasi antar versi

- `supersedes`: versi baru menunjuk versi sebelumnya (kolom + tautan).
- `duplicate_of`: dedup kosinus ≥ 0,98 menaut, tidak menghapus.
- `derived_from` / `consolidated_from`: jejak konsolidasi.
- `source`: versi asal pada versi hasil konsolidasi.

### Pencarian

- **Kata kunci**: FTS5 + BM25, ekspresi token alfanumerik berprefiks dan
  digabung `OR` (aman terhadap sintaks FTS).
- **Vektor**: `Embedder` di belakang trait; bawaan `HashNgramEmbedder` (FNV-1a
  atas kata dan n-gram karakter, 256 dims, deterministik). Kosinus brute-force.
- **Fusi**: RRF `Σ w_i / (k + rank_i(d))`, `k = 60` (Cormack dkk., SIGIR 2009),
  lalu dikali faktor usia `0,5^(umur/30 hari)` dan akses
  `1 + 0,1·ln(1 + jumlah_akses)`.
- **Filter**: path (per segmen), ruang nama, tag, dan rentang waktu.
- **Graf**: `links_from`, `links_to`, `neighbors`.

### Konsolidasi

Pekerja merangkum versi episodik menjadi versi baru di path semantik. Versi baru
menyimpan `source` dan tautan `derived_from`; isi sumber tetap utuh (diverifikasi
tes hash sebelum/sesudah). `pending_episodes` memilih episodik yang belum
diturunkan.

## Batas dan catatan

- Brute-force vektor berlaku sampai ~100 ribu vektor (O(N·d) per kueri). Jalur
  skala yang disiapkan: LanceDB (HNSW) + Tantivy. Skema `vectors` cukup untuk
  memuat indeks tersebut tanpa mengubah tabel versi.
- Embedder bawaan adalah heuristik leksikal, bukan model semantik; trait
  memungkinkan penggantian tanpa perubahan skema.
- FTS menyimpan salinan isi untuk indeks (lapisan turunan), bukan mengubah isi
  versi.

## Konsekuensi

- Semua operasi baca aman diulang dan dapat diverifikasi.
- Riwayat tak terbatas; retensi diatur oleh `forget`/`restore`, bukan penghapusan.
- Kualitas semantik Vbeta bergantung pada embedder bawaan; siap diganti model.