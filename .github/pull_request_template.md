## Ringkasan
<!-- Apa yang diubah dan mengapa -->

## Bukti
<!-- Wajib: tautan run CI, benchmark, metrik, atau tes -->
- [ ] CI hijau: <!-- link -->
- [ ] Tes: <!-- perintah + hasil -->
- [ ] Benchmark (jika jalur panas): <!-- sebelum/sesudah -->

## Checklist
- [ ] `cargo fmt --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `bash scripts/forbid_stubs.sh`
- [ ] Tidak ada stub/`unwrap` di kode pustaka
- [ ] Spec diperbarui (`specs/<ID>.md`)
- [ ] ADR bila keputusan arsitektur