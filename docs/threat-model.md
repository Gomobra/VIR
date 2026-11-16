//! Model ancaman sandbox VIR (F3.1).
//!
//! Rujukan: `PRD.md` F3.1, `docs/adr/ADR-006-sandbox.md`.
//! Dokumen ini adalah batas keamanan yang **dinyatakan**, bukan klaim isolasi
//! total. Setiap butir "tidak dilindungi" adalah batas yang harus diingat
//! pemakai `vir agent`/`vir rlm` saat menjalankan kode yang tidak dipercaya.

## 1. Aset yang dilindungi

| Aset | Mengapa penting |
|---|---|
| Berkas pengguna di luar workdir | Kode agen tidak boleh membaca/menulis data pribadi, kunci SSH, token |
| Identitas & memori proses lain | `ptrace`, `process_vm_readv`, fd proses lain |
| Jaringan | Eksfiltrasi data, unduh muatan, akses layanan internal |
| Sumber daya host | CPU, memori, proses, deskriptor berkas runner |
| Integritas kernel | Tidak ada jalur bagi kode agen untuk memuat modul atau memasang filter |

## 2. Aktor

- **Agen (tidak dipercaya)**: kode yang dihasilkan model atau ditulis pengguna
  (REPL F3.2, RLM F3.4, `llm_query`). Subjek utama sandbox.
- **Model**: dapat menghasilkan kode yang salah, berbahaya, atau sengaja
  berikhtiar keluar. Diperlakukan tidak dipercaya.
- **Pemanggil (host/CLI)**: tepercaya; memilih `Policy` dan membaca `RunResult`.

## 3. Kontrol yang ditegakkan (in-scope)

Diterapkan pada setiap proses anak oleh `limits::confine_command` +
`landlock::confine` + `seccomp::confine` (urutan Sandlock: `setpgid` → `chdir` →
`NO_NEW_PRIVS` → Landlock → seccomp → `exec`).

| Kontrol | Mekanisme | Syarat kernel |
|---|---|---|
| Baca/exec sistem berkas | Landlock: baca+exec pada `/`, `/usr`, `/lib`, `/etc`, `/proc`, `/sys`, `/dev`, `/tmp`, `/run` | ABI 1 (5.13+) |
| Tulis hanya di workdir | Landlock: tulis pada workdir + `Policy.allow_write` | ABI 1 |
| Tulis di luar workdir | **Ditolak** (tanpa aturan tulis di path itu) | ABI 1 |
| Jaringan TCP | Landlock: tanpa aturan port, `bind`/`connect` TCP → `EPERM` | ABI 4 (6.7+) |
| Syscall | seccomp-bpf allowlist; sisanya `EPERM` | 3.5+ (`CONFIG_SECCOMP_FILTER`) |
| Memori | `RLIMIT_AS` (`Policy.limits.mem_bytes`) | — |
| CPU | `RLIMIT_CPU` + timeout dinding `Policy.timeout` | — |
| Proses | `RLIMIT_NPROC` + `kill(-pgid, SIGKILL)` | — |
| Deskriptor berkas | `RLIMIT_NOFILE` | — |
| Keluaran besar | `Policy.max_output` (dipotong, `truncated`) | — |
| Lingkungan | `env_clear` + env eksplisit (`Policy.env`) | — |

Konfinemen bersifat **fail-closed**: `landlock::detect_abi` dan
`seccomp::detect_support` mengembalikan galat jelas bila kernel tak mendukung, dan
pemasangan yang gagal membuat anak keluar dengan galat — bukan "jalan tanpa
sandbox".

## 4. Yang TIDAK dilindungi (out-of-scope, dinyatakan)

1. **Eksploitasi kernel / LPE.** Landlock+seccomp tidak menambal bug kernel.
   Syscall yang diizinkan tetap dapat dipakai menyerang permukaan serangan kernel.
   Sandbox ini **bukan** pengganti microVM (Firecracker/gVisor) untuk kode yang
   benar-benar tidak dipercaya.
2. **Side-channel.** Timing, cache, speculative execution: tidak dimitigasi.
   `RLIMIT` membatasi jumlah, bukan kebocoran lewat waktu.
3. **Keadilan tingkat sistem (noisy neighbor).** `RLIMIT_NPROC` per-UID dan
   `RLIMIT_AS` per-proses tidak membatasi IO, inode, atau memori kernel bersama.
   Butuh cgroup v2 (`memory.max`, `pids.max`, `io.max`) untuk jaminan itu; PRD
   F3.1 menyebutnya dan menundanya (belum ada di F3.1).
4. **Anggaran waktu sesi.** Timeout berlaku per-`run`, bukan kuota total sesi.
   REPL berumur panjang (F3.2) wajib menambahkan anggaran sendiri.
5. **Isi keluaran.** stdout/stderr diperlakukan sebagai data, bukan perintah.
   Lapisan agen (RLM, prompt) harus menganggap keluaran agen sebagai tak
   tepercaya (prompt-injection).
6. **Rahasia di env.** `Policy.env` dibersihkan, tetapi agen dapat membaca
   `/proc/self/environ` dirinya sendiri; jangan menaruh token di env sandbox.
7. **Kerahasiaan baca sistem.** `/`, `/usr`, `/etc`, `/tmp` **dibaca** (bukan
   hanya workdir) agar runtime dapat mulai. Kode agen dapat membaca berkas
   dunia-baca mana pun (mis. `/etc/passwd`). Untuk lebih ketat, kurangi
   `COMMON_READ` di `landlock.rs` — saat ini mengorbankan kerahasiaan baca demi
   "selalu bisa jalan".
8. **`/proc` dan `/sys`.** Dibaca; metadata proses lain (cmdline) dapat terlihat.
   `ptrace`/`process_vm_readv` ditolak, tetapi isi `/proc` tidak.
9. **IPC non-TCP.** Landlock ABI < 6 tidak membatasi `AF_UNIX`; `seccomp`
   mengizinkan `socket` (agar kegagalan terlihat sebagai EPERM dari Landlock),
   sehingga jalur non-TCP harus dianggap terbuka pada ABI < 6.
10. **Integritas data host.** Bobot model, cache, berkas memori MFS tidak
    dilindungi sandbox dari penulisan proses lain di host; itu tugas izin
    filesystem host.
11. **Ketersediaan.** Host yang menolak Landlock/seccomp membuat `run` gagal
    (fail-closed), tetapi tidak mencegah DoS oleh proses di luar sandbox.

## 5. Asumsi lingkungan

- Kernel Linux ≥ 5.13 dengan `CONFIG_SECURITY_LANDLOCK=y`.
- ABI 4 (6.7+) diperlukan untuk penolakan TCP; pada ABI lebih rendah tes jaringan
  `SKIP` (bukan lulus) dan **jaringan tidak terkunci**.
- `CONFIG_SECCOMP` + `CONFIG_SECCOMP_FILTER=y`, boot dengan `seccomp=on`.
- `RLIMIT_NPROC` hanya berlaku bila UID nyata bukan root.
- Runner CI (`ubuntu-latest`) dan Kaggle memenuhi syarat; host Termux/Android
  **tidak** menjalankan sandbox (PRD F3.1).

## 6. Tes yang membuktikan kontrol (F3.1 butir 7)

`crates/vir-sandbox/tests/isolation.rs`, dijalankan job CI `vir-sandbox`
(`cargo test -p vir-sandbox --features sandbox,iso-helper`):

| Tes | Membuktikan |
|---|---|
| `loop_tak_hingga_dibunuh_timeout_dinding` | timeout dinding membunuh proses |
| `timeout_membunuh_seluruh_grup_proses` | `kill(-pgid)` mematikan cucu |
| `ledakan_memori_dibatasi_rlimit` | `RLIMIT_AS` menghentikan alokasi besar |
| `ledakan_proses_dibatasi_rlimit_nproc` | `RLIMIT_NPROC` membatasi fork |
| `tulis_di_luar_workdir_ditolak` | Landlock menolak tulis; pembanding dalam workdir berhasil |
| `jaringan_tcp_ditolak` | `connect` TCP → `EPERM` (ABI ≥ 4) |
| `syscall_terlarang_ditolak` | `ptrace` → `EPERM`; pembanding tanpa seccomp bukan EPERM |
| `helper_jalan_tanpa_konfinemen` | biner uji sehat (pembanding) |
| `konfinemen_anak_terpasang` | konfinemen benar-benar dipasang di runner |

Tes yang butuh kemampuan kernel yang tidak ada mencetak `SKIP: <alasan> [ABI,
seccomp, alasan pemasangan]` — tidak pernah lulus tanpa menguji.

## 7. Rekomendasi

1. Tambahkan cgroup v2 per sesi (`memory.max`, `pids.max`) di Fase berikutnya.
2. Turunkan `COMMON_READ` bila kerahasiaan baca lebih penting daripada
   "selalu bisa jalan".
3. Untuk isolasi kuat: microVM (Firecracker) atau gVisor.
4. Bila ABI jaringan < 4: tolak permintaan jaringan-mati secara eksplisit,
   jangan berasumsi aman.
5. Jangan menaruh rahasia di env atau workdir proses sandbox.
6. Perlakukan stdout/stderr agen sebagai data tak tepercaya di lapisan RLM/prompt.