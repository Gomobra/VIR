//! Data contoh untuk snapshot: JSONL metrik, eval, log, RLM, generate.
#![allow(dead_code)]

pub const METRICS_LINES: &str = concat!(
    r#"{"event":"metric","step":10,"loss":2.50,"lr":0.000300,"throughput":1180.5,"mem_mb":3400,"eta_secs":120,"backend":"cpu"}"#,
    "\n",
    r#"{"event":"metric","step":20,"loss":2.10,"lr":0.000280,"throughput":1290.0,"mem_mb":3520,"eta_secs":90,"backend":"cpu"}"#,
    "\n",
    r#"{"event":"metric","step":30,"loss":1.80,"lr":0.000250,"throughput":1310.0,"mem_mb":3600,"eta_secs":45,"backend":"cpu","#,
    r#""levels":[{"name":"L0","loss":1.90,"codebook_util":[0.10,0.55,0.92]},{"name":"L1","loss":1.70,"codebook_util":[0.30,0.62,0.20]}],"#,
    r#""gates":[{"name":"G1","state":"pass","detail":"rouge 0.42"}]}"#,
    "\n",
    r#"{"event":"checkpoint","name":"ckpt-000030","step":30,"loss":1.80,"path":"/tmp/ckpt-000030"}"#,
    "\n",
);

pub const EVAL_LINES: &str = concat!(
    r#"{"task":"ruler-4k","n":32,"c":8,"accuracy":0.75,"n_loops":4,"context_len":4096}"#,
    "\n",
    r#"{"task":"ruler-8k","n":32,"c":12,"accuracy":0.50,"n_loops":8,"context_len":8192}"#,
    "\n",
    r#"{"task":"arc-1k","n":32,"c":4,"accuracy":0.90,"n_loops":2,"context_len":1024}"#,
    "\n",
);

pub const LOG_LINES: &str = concat!(
    r#"{"time":"12:00:00","level":"info","target":"vir_train","message":"step 30 loss 1.80"}"#,
    "\n",
    r#"{"time":"12:00:01","level":"warn","target":"vir_eval","message":"gerbang G1 tunggu"}"#,
    "\n",
    r#"{"time":"12:00:02","level":"error","target":"vir_rlm","message":"sandbox gagal"}"#,
    "\n",
    r#"{"time":"12:00:03","level":"debug","target":"vir_model","message":"cache aktif"}"#,
    "\n",
    r#"{"time":"12:00:04","level":"trace","target":"vir_tui","message":"frame siap"}"#,
    "\n",
);

pub const RLM_LINES: &str = concat!(
    r#"{"event":"budget","max_depth":6,"max_calls":64}"#,
    "\n",
    r#"{"event":"step","id":0,"depth":0,"calls":1,"node":"akar","code":"for i in range(3): mem.find(i)","done":true}"#,
    "\n",
    r#"{"event":"step","id":1,"depth":1,"calls":2,"node":"mem.find('rust')","mem":true,"output":"path=/semantik/rust"}"#,
    "\n",
    r#"{"event":"final","answer":"42"}"#,
    "\n",
);

pub const GENERATE_LINES: &str = concat!(
    r#"{"event":"level","level":1,"text":"gagasan kasar tentang memori","halting":"continue","confidence":[0.20,0.55,0.91]}"#,
    "\n",
    r#"{"event":"level","level":2,"text":"gagasan halus tentang memori","halting":"stop","confidence":[0.45,0.62,0.95]}"#,
    "\n",
    r#"{"event":"candidate","text":"kandidat A","score":0.81}"#,
    "\n",
    r#"{"event":"final"}"#,
    "\n",
);
