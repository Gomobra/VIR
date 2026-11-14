# ADR-006: Sandbox (Landlock + seccomp)

## Konteks
Industri: Landlock + seccomp-bpf (Codex/Cursor), bubblewrap (Claude Code), Sandlock (ASPLOS 2026).

## Keputusan
- Landlock (filesystem) + seccomp-bpf allowlist + rlimit + timeout + kill process group.
- Deteksi ABI runtime; ABI 6 (kernel 6.12+) untuk IPC/signal; fallback eksplisit.
- Jaringan mati default.

### Urutan konfinemen (F3.1)
Sesudah `fork`, mengikuti Sandlock (arXiv 2605.26298):

1. `setpgid(0, 0)` — anak menjadi pemimpin grup proses sendiri (`limits.rs`);
2. `chdir(workdir)` — `Command::current_dir`;
3. `PR_SET_NO_NEW_PRIVS` — wajib sebelum seccomp agar filter berlaku tanpa root;
4. aturan Landlock (`landlock.rs`), yang juga memasang `NO_NEW_PRIVS`;
5. filter seccomp-bpf (`seccomp.rs`);
6. `exec`.

Langkah 3 diulang secara eksplisit di `vir-sandbox/src/lib.rs::wire_seccomp`
walaupun Landlock sudah memasangnya, supaya modul seccomp tetap benar bila
Landlock dimatikan lewat `Policy::with_landlock(false)`.

### seccomp: allowlist, bukan deny-list
Sandlock memakai deny-list (tolak syscall berbahaya tanpa syarat) plus
notifikasi ke supervisor. VIR memakai **allowlist**: hanya syscall yang ada di
`vir-sandbox/src/syscalls.rs` yang boleh jalan, sisanya ditolak.

Alasan:
- Permukaan serangan lebih kecil dan dapat diaudit: menambah kemampuan berarti
  menambah baris pada satu daftar, bukan mencari syscall baru yang berbahaya.
- Kegagalan aman secara bawaan. Syscall baru di kernel (mis. `io_uring_*`)
  tidak otomatis lolos; ia ditolak sampai sengaja diizinkan.
- Biaya: setiap runtime baru (CPython, Node, dan sebagainya) dapat menemukan
  syscall yang belum tercantum. Karena itu allowlist disusun untuk menampung
  CPython dan inti libc, dan kegagalan dibuat terlihat (lihat di bawah).

Aksi tolak memakai `SECCOMP_RET_ERRNO(EPERM)`, bukan `KILL`/`TRAP`. Program
menerima `EPERM` pada syscall yang ditolak sehingga galat muncul di keluaran
sendiri dan dapat dilacak; Sandlock memilih SIGSYS agar pelanggaran langsung
tampak. VIR menilai keterlacakan lebih berguna pada tahap ini; bila nanti perlu
deteksi pelanggaran yang lebih keras, aksi dapat diubah menjadi `TRAP` tanpa
mengubah allowlist.

### Deteksi kemampuan kernel
- Landlock: `landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)`
  mengembalikan versi ABI; kernel < 5.13 atau Landlock nonaktif ditolak dengan
  galat jelas (`landlock::detect_abi`).
- seccomp-bpf: `prctl(PR_GET_SECCOMP)`; galat berarti kernel < 3.5,
  `CONFIG_SECCOMP`/`CONFIG_SECCOMP_FILTER` mati, atau boot dengan
  `seccomp=off` (`seccomp::detect_support`).
- Penolakan dinyatakan sebagai galat, bukan "fitur dimatikan diam-diam", dan
  dapat dilewati secara sadar lewat `Policy::with_landlock(false)` /
  `Policy::with_seccomp(false)`.

### Arsitektur
`seccompiler` 0.4 hanya mendukung x86_64 dan aarch64. Nama konstanta syscall
berbeda antar libc, jadi allowlist disusun berlapis: daftar `COMMON` untuk semua
arsitektur/libc, ditambah daftar arsitektur-spesifik (mis. `newfstatat` tidak
ada pada Bionic; `open`/`stat`/`poll` hanya ada pada x86_64).

### Yang sengaja tidak diizinkan
`ptrace`, `process_vm_readv`/`writev`, `io_uring_*`, `mount*`, `pivot_root`,
`init_module`/`finit_module`, `bpf`, `kexec_load`, `perf_event_open`,
`userfaultfd`, `keyctl`/`add_key`, `mq_*`, `shm*`, `sem*`, `fanotify_*`,
`unshare`, dan `seccomp` itu sendiri (proses tidak boleh memasang filter baru).
Jaringan tetap mati: Landlock menolak `bind`/`connect` TCP; `socket` diizinkan
agar kegagalan terlihat sebagai `EPERM` dari Landlock, bukan dari seccomp.

## Status
Diimplementasikan pada F3.1 (`crates/vir-sandbox`): `limits.rs` (rlimit +
process group), `landlock.rs` (filesystem/jaringan + deteksi ABI),
`seccomp.rs` (filter + deteksi dukungan), `syscalls.rs` (allowlist),
`lib.rs` (urutan konfinemen, timeout dinding, pemotongan keluaran).

Belum ada:
- Tes isolasi `tests/isolation.rs` (loop tak hingga, ledakan memori, ledakan
  proses, tulis di luar workdir, akses jaringan, syscall terlarang) — F3.1
  butir 7, ditulis oleh `eval-eng`.
- Model ancaman dan batas keamanan di `docs/` — F3.1 butir 8.
- Landlock ABI 6 (IPC/signal): saat ini dipakai `ABI::V9` dengan penurunan hak
  eksplisit, bukan penolakan; keputusan ini ditunda sampai kernel sesi CI/Kaggle
  diperiksa.