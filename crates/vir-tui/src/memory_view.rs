//! Layar memori: pohon MFS, pratinjau `read`, dan hasil `find`.

use vir_mfs::{MemoryStore, ReadTarget, SearchHit, TreeNode};

pub struct MemoryView {
    path: String,
    store: Option<MemoryStore>,
    tree: Option<TreeNode>,
    pub rows: Vec<(usize, String, String)>,
    pub selected: usize,
    pub selected_path: String,
    pub preview: String,
    pub search: String,
    pub hits: Vec<SearchHit>,
    pub status: String,
    pub error: Option<String>,
}

impl MemoryView {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let mut view = Self {
            path,
            store: None,
            tree: None,
            rows: Vec::new(),
            selected: 0,
            selected_path: "/".into(),
            preview: "Pilih berkas lalu Enter untuk membaca.".into(),
            search: String::new(),
            hits: Vec::new(),
            status: "memori belum dibuka".into(),
            error: None,
        };
        view.refresh();
        view
    }

    pub fn refresh(&mut self) {
        self.error = None;
        if self.store.is_none() {
            match MemoryStore::open(&self.path) {
                Ok(store) => self.store = Some(store),
                Err(error) => {
                    self.error = Some(error.to_string());
                    self.status = "gagal membuka MFS".into();
                    return;
                }
            }
        }
        self.reload_tree();
    }

    fn reload_tree(&mut self) {
        let Some(store) = self.store.as_ref() else {
            return;
        };
        match vir_mfs::tree(store, "/") {
            Ok(tree) => {
                self.rows.clear();
                flatten(&tree, 0, &mut self.rows);
                self.tree = Some(tree);
                let count = vir_mfs::ls(store, "/")
                    .map(|items| items.len())
                    .unwrap_or(0);
                self.status = format!("{} entri · {} versi tampil", count, self.rows.len());
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let len = self.rows.len() as isize;
        self.selected = (self.selected as isize + delta).rem_euclid(len) as usize;
        self.sync_selection();
    }

    fn sync_selection(&mut self) {
        if let Some((_, path, _)) = self.rows.get(self.selected) {
            self.selected_path = path.clone();
        }
    }

    pub fn open_selected(&mut self) {
        let Some(store) = self.store.as_ref() else {
            return;
        };
        match vir_mfs::read(store, ReadTarget::Path(&self.selected_path)) {
            Ok(result) => {
                self.preview = format!(
                    "{} · versi {} · {}\n{}",
                    result.path, result.version_id, result.hash, result.content
                );
                self.status = format!("baca {}", result.path);
                self.error = None;
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    pub fn set_search(&mut self, query: impl Into<String>) {
        self.search = query.into();
        self.selected = 0;
    }

    pub fn search(&mut self) {
        let query = self.search.trim();
        if query.is_empty() {
            self.hits.clear();
            self.status = "kata kunci kosong".into();
            return;
        }
        let Some(store) = self.store.as_ref() else {
            return;
        };
        match vir_mfs::find(store, query, 8) {
            Ok(hits) => {
                self.hits = hits;
                self.status = format!("{} hasil untuk '{query}'", self.hits.len());
                if let Some(first) = self.hits.first() {
                    self.preview = format!(
                        "{} · skor {:.3} · BM25 {:?} · vektor {:?}\n{}",
                        first.path, first.score, first.fts_rank, first.vector_rank, first.content
                    );
                    self.selected_path = first.path.clone();
                }
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    pub fn hit_line(&self, hit: &SearchHit) -> String {
        format!(
            "{:.3}  {}  v{}  bm25={:?} vec={:?}",
            hit.score, hit.path, hit.version_id, hit.fts_rank, hit.vector_rank
        )
    }
}

fn flatten(node: &TreeNode, depth: usize, rows: &mut Vec<(usize, String, String)>) {
    let kind = if node.children.is_empty() {
        "file"
    } else {
        "dir"
    };
    rows.push((depth, node.path.clone(), kind.to_string()));
    for child in &node.children {
        flatten(child, depth + 1, rows);
    }
}
