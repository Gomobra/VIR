# Security Policy

## Dukungan
Vbeta adalah eksperimen. Laporkan kerentanan lewat GitHub Security Advisories atau email maintainer.

## Sandbox
`vir-sandbox` menjalankan kode di subprocess dengan timeout + batas keluaran. Landlock/seccomp (F3.1) belum aktif; jangan jalankan kode tak tepercaya di host produksi.

## Dependensi
`cargo deny check licenses bans sources` di CI. Advisories RUSTSEC dipantau; `libsqlite3-sys` dan `pyo3` punya advisory known (diizinkan sementara).