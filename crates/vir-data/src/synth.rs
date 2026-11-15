//! Korpus sintetis: aritmetika, kode, kalimat templat multibahasa (PRD F1.3 langkah 5).
//!
//! Semua generator deterministik: keluaran hanya bergantung pada `n`, sehingga
//! korpus uji dan smoke test selalu identik. Dipakai sebagai korpus kecil berlisensi
//! bersih (tanpa unduhan) untuk tes BPE, shard, dan pipeline data.

/// Jumlah pasang bahasa pada `synth_sentences`.
const LANGUAGES: [&str; 4] = ["id", "en", "ru", "ja"];

/// Deret aritmetika `a + b = c` (API lama, dipakai `vir smoke`).
pub fn synth_arithmetic(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let a = (i * 7 + 3) % 20;
            let b = (i * 11 + 5) % 20;
            let sum = a + b;
            format!("{a} + {b} = {sum}")
        })
        .collect()
}

/// Soal aritmetika campur (tambah, kurang, kali) dengan kunci jawaban.
pub fn synth_questions(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let a = (i * 13 + 7) % 50;
            let b = (i * 5 + 3) % 12 + 1;
            match i % 3 {
                0 => {
                    let answer = a + b;
                    format!("Q: berapa {a} + {b}? A: {answer}")
                }
                1 => {
                    let answer = a as i64 - b as i64;
                    format!("Q: what is {a} - {b}? A: {answer}")
                }
                _ => {
                    let answer = a * b;
                    format!("Q: сколько будет {a} * {b}? A: {answer}")
                }
            }
        })
        .collect()
}

/// Program kecil dengan fungsi dan loop.
pub fn synth_code(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let name = format!("f{i}");
            let acc = i % 5 + 1;
            format!(
                "def {name}(a, b):\n    total = a * b + {acc}\n    for k in range(3):\n        total += k\n    return total"
            )
        })
        .collect()
}

/// Kalimat templat multibahasa (Indonesia, Inggris, Rusia, Jepang).
pub fn synth_sentences(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let language = LANGUAGES[i % LANGUAGES.len()];
            let number = i % 10;
            let name = i % 3;
            let day = i + 1;
            match language {
                "id" => {
                    let who = nama(name);
                    format!("Pada hari ke-{day}, {who} membeli {number} buku di pasar.")
                }
                "en" => {
                    let who = nama_inggris(name);
                    format!("On day {day}, {who} bought {number} books at the market.")
                }
                "ru" => {
                    let who = nama_rusia(name);
                    format!("В день {day} {who} купил {number} книг на рынке.")
                }
                _ => {
                    let who = nama_jepang(name);
                    format!("{day}日目に{who}は市場で{number}冊の本を買いました。")
                }
            }
        })
        .collect()
}

/// Korpus campuran: aritmetika, kode, dan kalimat templat.
pub fn synth_corpus(n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n * 3);
    out.extend(synth_arithmetic(n));
    out.extend(synth_code(n));
    out.extend(synth_sentences(n));
    out
}

fn nama(index: usize) -> &'static str {
    match index % 3 {
        0 => "Ani",
        1 => "Budi",
        _ => "Sari",
    }
}

fn nama_inggris(index: usize) -> &'static str {
    match index % 3 {
        0 => "Ann",
        1 => "Bob",
        _ => "Cara",
    }
}

fn nama_rusia(index: usize) -> &'static str {
    match index % 3 {
        0 => "Анна",
        1 => "Борис",
        _ => "Вера",
    }
}

fn nama_jepang(index: usize) -> &'static str {
    match index % 3 {
        0 => "太郎",
        1 => "花子",
        _ => "一郎",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aritmetika_api_lama_tetap() {
        let texts = synth_arithmetic(50);
        assert_eq!(texts.len(), 50);
        assert_eq!(texts[0], "3 + 5 = 8");
        assert_eq!(texts[1], "10 + 16 = 26");
    }

    #[test]
    fn setiap_generator_deterministik() {
        assert_eq!(synth_arithmetic(10), synth_arithmetic(10));
        assert_eq!(synth_questions(10), synth_questions(10));
        assert_eq!(synth_code(10), synth_code(10));
        assert_eq!(synth_sentences(10), synth_sentences(10));
        assert_eq!(synth_corpus(3), synth_corpus(3));
    }

    #[test]
    fn kalimat_meliputi_empat_bahasa() {
        let texts = synth_sentences(8);
        assert_eq!(texts.len(), 8);
        let all: String = texts.concat();
        for word in ["Pada hari", "On day", "В день", "日目に"] {
            assert!(all.contains(word), "bahasa {word} tidak ada");
        }
    }

    #[test]
    fn kode_mengandung_fungsi() {
        let texts = synth_code(3);
        assert!(texts[0].contains("def f0(a, b):"));
        assert!(texts[2].contains("def f2(a, b):"));
    }

    #[test]
    fn korpus_campuran_berisi_semua() {
        let corpus = synth_corpus(4);
        assert_eq!(corpus.len(), 12);
        assert!(corpus.iter().any(|t| t.contains("def ")));
        assert!(corpus.iter().any(|t| t.contains(" = ")));
    }
}
