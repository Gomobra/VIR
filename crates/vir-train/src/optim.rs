//! Muon (Newton–Schulz) + AdamW hibrid, clipping, dan penyimpanan parameter
//! datar (PRD F1.7 langkah 4–6).
//!
//! **Resep Muon** (PRD F1.7 langkah 4): untuk matriks 2D di blok (q, k, v, o,
//! FFN), `M_t = μM_{t−1} + ∇L`; bentuk Nesterov `M̃ = ∇L + μM_t`;
//! `X₀ = M̃/‖M̃‖_F`; lima iterasi `X ← aX + b(XXᵀ)X + c(XXᵀ)²X` dengan
//! `(a,b,c) = (3,4445; −4,7750; 2,0315)` (ditranspos dulu bila A > B); update
//! `W ← W − η(0,2·√max(A,B)·O + λW)` dengan λ=0,1, μ=0,95. Embedding, norm, dan
//! head memakai AdamW (β₁=0,9, β₂=0,95, wd=0,1, clip norma 1).
//!
//! **Kenapa parameter datar.** Optimizer di sini bekerja pada `ParamSlot`
//! (buffer `f32` + gradien) sehingga bisa diuji numerik tanpa membangun model
//! Burn, bisa di-checkpoint byte demi byte, dan bisa dipakai ulang baik oleh
//! jalur host maupun oleh jembatan Burn ([`crate::burn_bridge`]).
//!
//! **Skala.** RMS update teoretis Muon ≈ `√(1/max(A,B))` (Lemma 1), jadi faktor
//! `0,2·√max(A,B)` membuat RMS ≈ 0,2 seperti AdamW sehingga LR dan weight decay
//! AdamW dapat dipakai ulang; weight decay wajib agar norma bobot tidak
//! membengkak keluar jangkauan BF16.

use crate::schedule::LrSchedule;
use serde::{Deserialize, Serialize};

/// Koefisien Newton–Schulz lima iterasi (Keller Jordan; PRD langkah 4).
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
/// Faktor skala Muon agar RMS update ≈ RMS AdamW (0,2).
pub const MUON_SCALE: f32 = 0.2;
/// Iterasi Newton–Schulz default.
pub const NS_STEPS: usize = 5;

/// Peran parameter dalam latihan dua tahap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamRole {
    /// Bagian VQ-VAE (tahap 1); dibekukan saat masuk tahap 2.
    Vq,
    /// Bagian generator (tahap 2).
    Generator,
}

/// Grup optimizer: Muon untuk matriks, AdamW untuk sisanya.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptGroup {
    /// Matriks 2D di blok (q, k, v, o, FFN).
    Muon,
    /// Embedding, norm, head, bias.
    AdamW,
}

/// Aturan klasifikasi parameter (PRD langkah 4).
///
/// Nama yang mengandung `head`, `embed`, `codebook`, `norm`, `bias`, atau
/// parameter 1D → AdamW; selain itu (matriks 2D blok) → Muon.
pub fn classify(name: &str, rows: usize, cols: usize) -> OptGroup {
    let lower = name.to_lowercase();
    let special = [
        "head", "embed", "codebook", "norm", "bias", "scale", "shift",
    ];
    if rows <= 1 || cols <= 1 || special.iter().any(|s| lower.contains(*s)) {
        return OptGroup::AdamW;
    }
    OptGroup::Muon
}

/// Satu parameter datar beserta gradiennya.
#[derive(Debug, Clone)]
pub struct ParamSlot {
    /// Nama parameter (dipakai klasifikasi dan checkpoint).
    pub name: String,
    /// Baris matriks (1 untuk vektor).
    pub rows: usize,
    /// Kolom matriks (1 untuk vektor).
    pub cols: usize,
    /// Peran tahap.
    pub role: ParamRole,
    /// Grup optimizer.
    pub group: OptGroup,
    /// `true` bila dibekukan (tidak diupdate).
    pub frozen: bool,
    /// Nilai parameter, row-major.
    pub data: Vec<f32>,
    /// Gradien parameter, row-major.
    pub grad: Vec<f32>,
}

impl ParamSlot {
    /// Slot baru; `data` dipendekkan/dilengkapi agar sepanjang `rows*cols`.
    pub fn new(name: &str, rows: usize, cols: usize, role: ParamRole, data: Vec<f32>) -> Self {
        let n = rows.saturating_mul(cols).max(1);
        let mut data = data;
        data.resize(n, 0.0);
        Self {
            name: name.to_string(),
            rows,
            cols,
            role,
            group: classify(name, rows, cols),
            frozen: false,
            grad: vec![0.0; n],
            data,
        }
    }

    /// Slot vektor (bias/norm/head 1D).
    pub fn vector(name: &str, role: ParamRole, data: Vec<f32>) -> Self {
        let n = data.len().max(1);
        Self {
            name: name.to_string(),
            rows: n,
            cols: 1,
            role,
            group: OptGroup::AdamW,
            frozen: false,
            grad: vec![0.0; n],
            data,
        }
    }

    /// Jumlah elemen.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` bila tidak ada elemen.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Kumpulan parameter datar (state model yang bisa di-checkpoint).
#[derive(Debug, Clone, Default)]
pub struct ParamStore {
    /// Parameter berurutan sesuai rekaman checkpoint.
    pub slots: Vec<ParamSlot>,
}

impl ParamStore {
    /// Store kosong.
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Tambah slot, kembalikan indeksnya.
    pub fn add(&mut self, slot: ParamSlot) -> usize {
        self.slots.push(slot);
        self.slots.len() - 1
    }

    /// Cari indeks slot berdasarkan nama.
    pub fn index(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|s| s.name == name)
    }

    /// Slot pada indeks (None bila di luar rentang).
    pub fn get(&self, i: usize) -> Option<&ParamSlot> {
        self.slots.get(i)
    }

    /// Total elemen seluruh parameter.
    pub fn num_params(&self) -> usize {
        self.slots.iter().map(|s| s.len()).sum()
    }

    /// Nol-kan semua gradien.
    pub fn clear_grads(&mut self) {
        for s in self.slots.iter_mut() {
            for g in s.grad.iter_mut() {
                *g = 0.0;
            }
        }
    }

    /// `true` bila ada gradien non-nol (dipakai untuk melewati update).
    pub fn has_grads(&self) -> bool {
        self.slots
            .iter()
            .any(|s| !s.frozen && s.grad.iter().any(|g| *g != 0.0))
    }

    /// Norma L2 global seluruh gradien (untuk clipping dan log).
    pub fn grad_norm(&self) -> f32 {
        let mut acc = 0.0f64;
        for s in &self.slots {
            if s.frozen {
                continue;
            }
            for g in &s.grad {
                acc += (*g as f64) * (*g as f64);
            }
        }
        acc.sqrt() as f32
    }

    /// Bekukan semua slot dengan peran tertentu (transisi tahap).
    pub fn freeze_role(&mut self, role: ParamRole) -> usize {
        let mut n = 0;
        for s in self.slots.iter_mut() {
            if s.role == role && !s.frozen {
                s.frozen = true;
                n += 1;
            }
        }
        n
    }

    /// Set gradien satu slot (panjang harus sama).
    pub fn set_grad(&mut self, i: usize, g: &[f32]) -> bool {
        match self.slots.get_mut(i) {
            Some(s) if s.grad.len() == g.len() => {
                s.grad.copy_from_slice(g);
                true
            }
            _ => false,
        }
    }

    /// Tambah gradien satu slot (untuk akumulasi gradien).
    pub fn add_grad(&mut self, i: usize, g: &[f32]) -> bool {
        match self.slots.get_mut(i) {
            Some(s) if s.grad.len() == g.len() => {
                for (acc, v) in s.grad.iter_mut().zip(g.iter()) {
                    *acc += *v;
                }
                true
            }
            _ => false,
        }
    }
}

/// Matriks `a` (`m×k`) kali `b` (`k×n`), row-major.
fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Transpos matriks `m×n` → `n×m`.
fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Satu langkah iterasi `X ← aX + b(XXᵀ)X + c(XXᵀ)²X`.
fn ns_step(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (a, b, c) = NS_COEFFS;
    if rows == 0 || cols == 0 {
        return x.to_vec();
    }
    if rows <= cols {
        let xt = transpose(x, rows, cols);
        let g = matmul(x, rows, cols, &xt, rows);
        let gx = matmul(&g, rows, rows, x, cols);
        let g2x = matmul(&g, rows, rows, &gx, cols);
        combine(x, &gx, &g2x, a, b, c)
    } else {
        let xt = transpose(x, rows, cols);
        let g = matmul(&xt, cols, rows, x, cols);
        let gx = matmul(&g, cols, cols, &xt, rows);
        let g2x = matmul(&g, cols, cols, &gx, rows);
        let yt = combine(&xt, &gx, &g2x, a, b, c);
        transpose(&yt, cols, rows)
    }
}

/// `a·x + b·gx + c·g2x` elemen demi elemen.
fn combine(x: &[f32], gx: &[f32], g2x: &[f32], a: f32, b: f32, c: f32) -> Vec<f32> {
    x.iter()
        .zip(gx.iter())
        .zip(g2x.iter())
        .map(|((x0, g1), g2)| a * x0 + b * g1 + c * g2)
        .collect()
}

/// Ortogonalisasi Newton–Schulz 5 iterasi (in-place, row-major `[rows, cols]`).
///
/// `X₀ = M/‖M‖_F`. Bila `rows > cols` iterasi dijalankan pada `Mᵀ` lalu
/// ditranspos kembali (PRD langkah 4: "transpos dulu bila A > B").
pub fn newton_schulz(m: &mut [f32], rows: usize, cols: usize) {
    newton_schulz_steps(m, rows, cols, NS_STEPS);
}

/// Newton–Schulz dengan jumlah iterasi eksplisit.
pub fn newton_schulz_steps(m: &mut [f32], rows: usize, cols: usize, steps: usize) {
    if rows == 0 || cols == 0 || steps == 0 || m.len() != rows * cols {
        return;
    }
    let norm = m.iter().map(|v| *v * *v).sum::<f32>().sqrt().max(1e-7);
    for v in m.iter_mut() {
        *v /= norm;
    }
    for _ in 0..steps {
        let next = ns_step(m, rows, cols);
        m.copy_from_slice(&next);
    }
}

/// Baca `f32` little-endian pada `pos` (byte di luar rentang dianggap nol).
fn read_f32(buf: &[u8], pos: usize) -> f32 {
    let b = [
        buf.get(pos).copied().unwrap_or(0),
        buf.get(pos + 1).copied().unwrap_or(0),
        buf.get(pos + 2).copied().unwrap_or(0),
        buf.get(pos + 3).copied().unwrap_or(0),
    ];
    f32::from_le_bytes(b)
}

/// State momentum Muon satu slot: `f32` penuh atau 8-bit linear (F1.7 task 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MomState {
    /// Momentum `f32` (default).
    F32(Vec<f32>),
    /// Momentum terkuantisasi: `q = round(v/scale)`, `scale = max|v|/127`.
    I8 { scale: f32, q: Vec<i8> },
}

impl MomState {
    /// State nol untuk `len` elemen.
    pub fn zeros(len: usize, bits: u8) -> Self {
        if bits == 8 {
            Self::I8 {
                scale: 1.0,
                q: vec![0i8; len],
            }
        } else {
            Self::F32(vec![0.0; len])
        }
    }

    /// Jumlah elemen.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::I8 { q, .. } => q.len(),
        }
    }

    /// `true` bila kosong.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Nilai momentum pada indeks.
    pub fn value(&self, i: usize) -> f32 {
        match self {
            Self::F32(v) => v.get(i).copied().unwrap_or(0.0),
            Self::I8 { scale, q } => q.get(i).copied().unwrap_or(0) as f32 * *scale,
        }
    }

    /// Simpan momentum baru (mengkuantisasi bila 8-bit).
    pub fn store(&mut self, v: &[f32]) {
        match self {
            Self::F32(buf) => {
                if buf.len() == v.len() {
                    buf.copy_from_slice(v);
                } else {
                    *buf = v.to_vec();
                }
            }
            Self::I8 { scale, q } => {
                let peak = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let s = (peak / 127.0).max(1e-8);
                *scale = s;
                if q.len() != v.len() {
                    *q = vec![0i8; v.len()];
                }
                for (slot, val) in q.iter_mut().zip(v.iter()) {
                    *slot = (val / s).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }

    /// Byte mentah untuk checkpoint.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::F32(v) => {
                out.push(0);
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Self::I8 { scale, q } => {
                out.push(1);
                out.extend_from_slice(&scale.to_le_bytes());
                for x in q {
                    out.push(*x as u8);
                }
            }
        }
    }

    /// Pulihkan dari byte checkpoint.
    pub fn decode(
        tag: u8,
        len: usize,
        buf: &[u8],
        pos: &mut usize,
    ) -> Result<Self, vir_common::VirError> {
        use vir_common::VirError;
        let need = |n: usize, pos: usize| -> Result<(), VirError> {
            if pos + n <= buf.len() {
                Ok(())
            } else {
                Err(VirError::Internal("checkpoint momentum terpotong".into()))
            }
        };
        match tag {
            0 => {
                need(len * 4, *pos)?;
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(read_f32(buf, *pos));
                    *pos += 4;
                }
                Ok(Self::F32(v))
            }
            1 => {
                need(4 + len, *pos)?;
                let scale = read_f32(buf, *pos);
                *pos += 4;
                let mut q = Vec::with_capacity(len);
                for _ in 0..len {
                    q.push(buf[*pos] as i8);
                    *pos += 1;
                }
                Ok(Self::I8 { scale, q })
            }
            other => Err(VirError::Internal(format!(
                "tag momentum tak dikenal: {other}"
            ))),
        }
    }
}

/// Parameter Muon (β, Nesterov, iterasi NS, lebar state).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MuonParams {
    /// Momentum μ (default 0,95).
    pub beta: f32,
    /// Bentuk Nesterov `M̃ = ∇L + μM_t` (default `true`).
    pub nesterov: bool,
    /// Iterasi Newton–Schulz (default 5).
    pub ns_steps: usize,
    /// Lebar state momentum: 32 atau 8 bit (default 32).
    pub state_bits: u8,
}

impl Default for MuonParams {
    fn default() -> Self {
        Self {
            beta: 0.95,
            nesterov: true,
            ns_steps: NS_STEPS,
            state_bits: 32,
        }
    }
}

impl MuonParams {
    /// Validasi parameter.
    pub fn validate(&self) -> Result<(), vir_common::VirError> {
        use vir_common::VirError;
        if !(0.0..1.0).contains(&self.beta) {
            return Err(VirError::Config("Muon beta harus di [0,1)".into()));
        }
        if self.ns_steps == 0 {
            return Err(VirError::Config("ns_steps harus > 0".into()));
        }
        if self.state_bits != 8 && self.state_bits != 32 {
            return Err(VirError::Config("state_bits harus 8 atau 32".into()));
        }
        Ok(())
    }
}

/// Optimizer Muon untuk matriks 2D (update manual, tanpa Burn).
#[derive(Debug, Clone)]
pub struct Muon {
    /// Momentum per slot (sejajar dengan `ParamStore::slots`).
    pub states: Vec<MomState>,
    /// Parameter Muon.
    pub params: MuonParams,
}

impl Muon {
    /// Muon dengan state nol untuk semua slot.
    pub fn new(num_slots: usize, params: &MuonParams) -> Self {
        let mut states = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            states.push(MomState::zeros(0, params.state_bits));
        }
        Self {
            states,
            params: *params,
        }
    }

    /// Muon dengan state nol sepanjang `ParamStore`.
    pub fn from_store(store: &ParamStore, params: &MuonParams) -> Self {
        let mut states = Vec::with_capacity(store.slots.len());
        for s in &store.slots {
            states.push(MomState::zeros(s.len(), params.state_bits));
        }
        Self {
            states,
            params: *params,
        }
    }

    /// Muon dari state tersimpan (resume).
    pub fn from_states(states: Vec<MomState>, params: &MuonParams) -> Self {
        Self {
            states,
            params: *params,
        }
    }

    /// Update satu slot: momentum, Nesterov, Newton–Schulz, lalu langkah.
    pub fn step(&mut self, i: usize, slot: &mut ParamSlot, lr: f32, wd: f32) -> bool {
        let n = slot.len();
        if slot.frozen || slot.grad.len() != n {
            return false;
        }
        if self.states.len() <= i || self.states[i].len() != n {
            let bits = self.params.state_bits;
            if self.states.len() <= i {
                self.states.resize(i + 1, MomState::zeros(n, bits));
            }
            self.states[i] = MomState::zeros(n, bits);
        }
        let beta = self.params.beta;
        let prev: Vec<f32> = (0..n).map(|k| self.states[i].value(k)).collect();
        let mut mom = vec![0.0f32; n];
        for ((m, g), p) in mom.iter_mut().zip(slot.grad.iter()).zip(prev.iter()) {
            *m = beta * p + g;
        }
        let mut view: Vec<f32> = if self.params.nesterov {
            slot.grad
                .iter()
                .zip(mom.iter())
                .map(|(g, m)| g + beta * m)
                .collect()
        } else {
            mom.clone()
        };
        newton_schulz_steps(
            &mut view,
            slot.rows.max(1),
            slot.cols.max(1),
            self.params.ns_steps,
        );
        self.states[i].store(&mom);
        let scale = MUON_SCALE * (slot.rows.max(slot.cols).max(1) as f32).sqrt();
        for (w, v) in slot.data.iter_mut().zip(view.iter()) {
            *w -= lr * (scale * v + wd * *w);
        }
        true
    }
}

/// State AdamW satu slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamState {
    /// Momen pertama.
    pub m: Vec<f32>,
    /// Momen kedua.
    pub v: Vec<f32>,
    /// Hitungan langkah.
    pub t: u64,
}

impl AdamState {
    /// State nol untuk `len` elemen.
    pub fn zeros(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
        }
    }
}

/// Parameter AdamW.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdamParams {
    /// β₁ (default 0,9).
    pub beta1: f32,
    /// β₂ (default 0,95).
    pub beta2: f32,
    /// Epsilon (default 1e-8).
    pub eps: f32,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
        }
    }
}

/// AdamW dengan weight decay terpisah (decoupled) untuk embedding/norm/head.
#[derive(Debug, Clone)]
pub struct AdamW {
    /// State per slot.
    pub states: Vec<AdamState>,
    /// Parameter AdamW.
    pub params: AdamParams,
}

impl AdamW {
    /// AdamW dengan state nol untuk semua slot.
    pub fn from_store(store: &ParamStore, params: &AdamParams) -> Self {
        let states = store
            .slots
            .iter()
            .map(|s| AdamState::zeros(s.len()))
            .collect();
        Self {
            states,
            params: *params,
        }
    }

    /// AdamW dari state tersimpan (resume).
    pub fn from_states(states: Vec<AdamState>, params: &AdamParams) -> Self {
        Self {
            states,
            params: *params,
        }
    }

    /// Update satu slot.
    pub fn step(&mut self, i: usize, slot: &mut ParamSlot, lr: f32, wd: f32) -> bool {
        let n = slot.len();
        if slot.frozen || slot.grad.len() != n {
            return false;
        }
        if self.states.len() <= i || self.states[i].m.len() != n {
            if self.states.len() <= i {
                self.states.resize(i + 1, AdamState::zeros(n));
            }
            self.states[i] = AdamState::zeros(n);
        }
        let b1 = self.params.beta1;
        let b2 = self.params.beta2;
        let eps = self.params.eps;
        let st = &mut self.states[i];
        st.t += 1;
        let bc1 = 1.0 - b1.powi(st.t as i32);
        let bc2 = 1.0 - b2.powi(st.t as i32);
        for ((w, g), (m, v)) in slot
            .data
            .iter_mut()
            .zip(slot.grad.iter())
            .zip(st.m.iter_mut().zip(st.v.iter_mut()))
        {
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            let mh = *m / bc1;
            let vh = *v / bc2;
            let upd = mh / (vh.sqrt() + eps);
            *w -= lr * (upd + wd * *w);
        }
        true
    }
}

/// Pengaturan optimizer hibrid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OptimSettings {
    /// Parameter Muon.
    pub muon: MuonParams,
    /// Parameter AdamW.
    pub adamw: AdamParams,
    /// Weight decay (0,1 pada NCM Tabel 6).
    pub weight_decay: f32,
    /// Skala LR Muon relatif terhadap AdamW (grid terpisah).
    pub muon_lr_scale: f32,
}

impl Default for OptimSettings {
    fn default() -> Self {
        Self {
            muon: MuonParams::default(),
            adamw: AdamParams::default(),
            weight_decay: 0.1,
            muon_lr_scale: 0.5,
        }
    }
}

/// Optimizer hibrid Muon + AdamW atas `ParamStore`.
#[derive(Debug, Clone)]
pub struct Hybrid {
    /// Muon (slot grup Muon).
    pub muon: Muon,
    /// AdamW (slot grup AdamW).
    pub adamw: AdamW,
    /// Pengaturan aktif.
    pub settings: OptimSettings,
}

impl Hybrid {
    /// Bangun optimizer untuk `store` (state nol).
    pub fn new(store: &ParamStore, settings: &OptimSettings) -> Self {
        Self {
            muon: Muon::from_store(store, &settings.muon),
            adamw: AdamW::from_store(store, &settings.adamw),
            settings: *settings,
        }
    }

    /// Bangun dari state tersimpan (resume).
    pub fn from_states(
        _store: &ParamStore,
        settings: &OptimSettings,
        muon_states: Vec<MomState>,
        adam_states: Vec<AdamState>,
    ) -> Self {
        Self {
            muon: Muon::from_states(muon_states, &settings.muon),
            adamw: AdamW::from_states(adam_states, &settings.adamw),
            settings: *settings,
        }
    }

    /// Satu langkah atas semua slot yang tidak dibekukan.
    /// `lr_muon` dan `lr_adamw` sengaja terpisah (grid LR berbeda).
    pub fn step(&mut self, store: &mut ParamStore, lr_muon: f32, lr_adamw: f32) -> usize {
        let wd = self.settings.weight_decay;
        let mut updated = 0;
        for (i, slot) in store.slots.iter_mut().enumerate() {
            if slot.frozen {
                continue;
            }
            let done = match slot.group {
                OptGroup::Muon => self.muon.step(i, slot, lr_muon, wd),
                OptGroup::AdamW => self.adamw.step(i, slot, lr_adamw, wd),
            };
            if done {
                updated += 1;
            }
        }
        updated
    }

    /// Langkah dengan satu laju (dipakai bila grid tidak dipisah).
    pub fn step_single(&mut self, store: &mut ParamStore, lr: f32) -> usize {
        let scale = self.settings.muon_lr_scale;
        self.step(store, lr * scale, lr)
    }

    /// Langkah mengikuti jadwal LR (Muon memakai grid terpisah).
    pub fn step_schedule(
        &mut self,
        store: &mut ParamStore,
        sched: &LrSchedule,
        step: u64,
    ) -> usize {
        let lr_adamw = sched.lr_f32(step);
        let lr_muon = sched.lr_f32(step) * self.settings.muon_lr_scale;
        self.step(store, lr_muon, lr_adamw)
    }
}

/// Clip norma gradien global; kembalikan norma sebelum clipping.
pub fn clip_grad_norm(store: &mut ParamStore, max_norm: f32) -> f32 {
    let norm = store.grad_norm();
    if max_norm > 0.0 && norm > max_norm && norm.is_finite() {
        let scale = max_norm / norm;
        for s in store.slots.iter_mut() {
            if s.frozen {
                continue;
            }
            for g in s.grad.iter_mut() {
                *g *= scale;
            }
        }
    }
    norm
}

/// Uji numerik Muon (PRD langkah 5): koefisien NS, nilai singular hasil, RMS.
pub fn muon_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let bad = |m: String| Err(VirError::Internal(m));
    let (a, b, c) = NS_COEFFS;
    if (a + b + c - 0.701).abs() > 1e-3 {
        return bad(format!("a+b+c = {} != 0,701", a + b + c));
    }
    // Matriks acak 32×16: setelah NS5, norma kolom XᵀX mendekati 1 dan
    // kolom berbeda hampir ortogonal (toleransi, bukan ortogonal eksak).
    let mut x = vec![0.0f32; 32 * 16];
    let mut rng = crate::rng::Rng::new(5);
    for v in x.iter_mut() {
        *v = rng.next_f32() - 0.5;
    }
    newton_schulz(&mut x, 32, 16);
    let xt = transpose(&x, 32, 16);
    let g = matmul(&xt, 16, 32, &x, 16);
    for i in 0..16 {
        for j in 0..16 {
            let v = g[i * 16 + j];
            if i == j && !(0.4..=1.6).contains(&v) {
                return bad(format!("norma kolom {i} = {v} di luar [0,4; 1,6]"));
            }
            if i != j && v.abs() > 0.4 {
                return bad(format!("kolom {i} dan {j} tidak ortogonal: {v}"));
            }
        }
    }
    // RMS update Muon ≈ 0,2 (Lemma 1 + faktor skala).
    let mut slot = ParamSlot::new(
        "core.q.weight",
        32,
        16,
        ParamRole::Generator,
        vec![0.1; 32 * 16],
    );
    let mut seed = 99u64;
    for g in slot.grad.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *g = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
    }
    let mut muon = Muon::new(1, &MuonParams::default());
    let before: Vec<f32> = slot.data.clone();
    if !muon.step(0, &mut slot, 1e-3, 0.1) {
        return bad("Muon menolak slot valid".to_string());
    }
    let delta: Vec<f32> = slot
        .data
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect();
    let rms = (delta.iter().map(|d| d * d).sum::<f32>() / delta.len() as f32).sqrt();
    // Matriks hasil NS berkolom ortonormal: RMS ≈ 1/√max(A,B); dengan faktor
    // 0,2·√max(A,B) dan η=1e-3, RMS update ≈ η·0,2 (Lemma 1 + PRD langkah 4).
    let expect_rms = 1e-3 * MUON_SCALE;
    if !(rms > expect_rms * 0.55 && rms < expect_rms * 1.7) {
        return bad(format!("RMS update Muon {rms} di luar dugaan {expect_rms}"));
    }
    Ok(())
}

/// Uji asap AdamW: langkah pertama mengikuti tanda gradien.
pub fn adam_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let mut store = ParamStore::new();
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Generator,
        vec![1.0, 1.0],
    ));
    store.add(ParamSlot::new(
        "core.q.weight",
        4,
        4,
        ParamRole::Generator,
        vec![0.0; 16],
    ));
    if store.get(0).map(|s| s.group) != Some(OptGroup::AdamW) {
        return Err(VirError::Internal("norm salah klasifikasi".into()));
    }
    if store.get(1).map(|s| s.group) != Some(OptGroup::Muon) {
        return Err(VirError::Internal("matriks blok salah klasifikasi".into()));
    }
    if !store.set_grad(0, &[1.0, -1.0]) {
        return Err(VirError::Internal("set_grad gagal".into()));
    }
    if !store.set_grad(1, &[0.01; 16]) {
        return Err(VirError::Internal("set_grad gagal".into()));
    }
    let mut opt = Hybrid::new(&store, &OptimSettings::default());
    let updated = opt.step_single(&mut store, 1e-2);
    if updated != 2 {
        return Err(VirError::Internal("jumlah slot terupdate salah".into()));
    }
    let s0 = store.get(0).map(|s| s.data.clone()).unwrap_or_default();
    if !(s0[0] < 1.0 && s0[1] > 1.0) {
        return Err(VirError::Internal(
            "AdamW tidak mengikuti tanda gradien".into(),
        ));
    }
    let frozen = store.freeze_role(ParamRole::Vq);
    if frozen != 0 {
        return Err(VirError::Internal("freeze_role salah hitung".into()));
    }
    Ok(())
}
