//! E2E: MFS + RLM + sandbox (F3.7).

use vir_agent::{RlmConfig, ScriptedLm, run_rlm};
use vir_mfs::MemoryStore;

#[test]
fn e2e_memory_then_rlm() {
    // tulis ingatan
    let store = MemoryStore::open_in_memory().unwrap();
    let vid = store
        .write("/semantik/catatan", "rust adalah bahasa cepat")
        .unwrap();
    let content = store.read(vid).unwrap();
    assert_eq!(content, "rust adalah bahasa cepat");

    // RLM dengan scripted LM yang mencetak FINAL
    let lm = ScriptedLm::new(vec!["print('FINAL(ok)')".into()]);
    let cfg = RlmConfig {
        max_depth: 2,
        max_calls: 2,
    };
    let r = run_rlm(&lm, "ctx", &cfg).unwrap();
    assert!(r.final_answer.is_some());
}

#[test]
fn e2e_find_hybrid() {
    let store = MemoryStore::open_in_memory().unwrap();
    store.write("/a", "rust programming").unwrap();
    store.write("/b", "python data").unwrap();
    let hits = store.find("rust", 5).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].1.contains("rust"));
}
