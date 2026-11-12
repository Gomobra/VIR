//! Operasi tingkat pengguna: `ls`, `tree`, `find`, `read`, `write`, `link`,
//! `tag`, `history`, `diff`, `forget`, `restore`.

use std::collections::BTreeMap;

use crate::catalog::{FileEntry, VersionInfo};
use crate::diff::{Diff, DiffKind, diff_lines};
use crate::meta::LinkKind;
use crate::path::normalize_path;
use crate::query::{SearchHit, SearchOptions};
use crate::store::{MemoryStore, WriteOutcome};
use crate::{Result, VirError};

/// Simpul pohon ruang nama.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeNode {
    pub name: String,
    pub path: String,
    pub entry: Option<FileEntry>,
    pub children: Vec<TreeNode>,
}

#[derive(Default, Clone)]
struct Node {
    name: String,
    path: String,
    entry: Option<FileEntry>,
    children: BTreeMap<String, Node>,
}

impl Node {
    fn root() -> Self {
        Self {
            name: "/".to_string(),
            path: "/".to_string(),
            entry: None,
            children: BTreeMap::new(),
        }
    }

    fn child_path(&self, name: &str) -> String {
        if self.path == "/" {
            format!("/{name}")
        } else {
            format!("{}/{name}", self.path)
        }
    }

    fn insert(&mut self, parts: &[&str], entry: FileEntry) {
        if parts.is_empty() {
            self.entry = Some(entry);
            return;
        }
        let name = parts[0];
        let path = self.child_path(name);
        let child = self
            .children
            .entry(name.to_string())
            .or_insert_with(|| Node {
                name: name.to_string(),
                path: path.clone(),
                entry: None,
                children: BTreeMap::new(),
            });
        child.path = path;
        child.insert(&parts[1..], entry);
    }

    fn find(&self, parts: &[&str]) -> Option<&Node> {
        if parts.is_empty() {
            return Some(self);
        }
        self.children.get(parts[0])?.find(&parts[1..])
    }

    fn into_tree(self) -> TreeNode {
        TreeNode {
            name: self.name,
            path: self.path,
            entry: self.entry,
            children: self.children.into_values().map(Node::into_tree).collect(),
        }
    }
}

/// Daftar berkas di bawah prefiks path (versi terbaru, belum dilupakan).
pub fn ls(store: &MemoryStore, prefix: &str) -> Result<Vec<FileEntry>> {
    let prefix = normalize_path(prefix)?;
    let mut out = Vec::new();
    for entry in store.entries()? {
        if entry.deleted || !under_prefix(&entry.path, &prefix) {
            continue;
        }
        out.push(entry);
    }
    Ok(out)
}

/// Cocokkan path dengan prefiks segmen (bukan sekadar awalan teks).
pub(crate) fn under_prefix(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Pohon ruang nama di bawah `root`.
pub fn tree(store: &MemoryStore, root: &str) -> Result<TreeNode> {
    let root = normalize_path(root)?;
    let mut all = Node::root();
    for entry in store.entries()? {
        if entry.deleted {
            continue;
        }
        let path = entry.path.trim_matches('/').to_string();
        if path.is_empty() {
            continue;
        }
        let parts: Vec<&str> = path.split('/').collect();
        all.insert(&parts, entry);
    }
    if root == "/" {
        return Ok(all.into_tree());
    }
    let parts: Vec<&str> = root.trim_matches('/').split('/').collect();
    let node = all.find(&parts).cloned().unwrap_or_else(|| Node {
        name: parts.last().copied().unwrap_or("/").to_string(),
        path: root.clone(),
        entry: None,
        children: BTreeMap::new(),
    });
    Ok(node.into_tree())
}

/// Pencarian hibrida bawaan.
pub fn find(store: &MemoryStore, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let opts = SearchOptions {
        query: query.to_string(),
        limit,
        ..SearchOptions::default()
    };
    store.search(&opts)
}

/// Target operasi baca/lupa: path atau id versi.
#[derive(Debug, Clone, Copy)]
pub enum ReadTarget<'a> {
    Path(&'a str),
    Version(i64),
}

/// Hasil operasi baca.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadResult {
    pub version_id: i64,
    pub file_id: i64,
    pub path: String,
    pub hash: String,
    pub content: String,
}

/// Baca versi terbaru path atau satu versi; isi tidak diubah.
pub fn read(store: &MemoryStore, target: ReadTarget<'_>) -> Result<ReadResult> {
    let version_id = match target {
        ReadTarget::Path(path) => {
            let path = normalize_path(path)?;
            let vid = store
                .current_version(&path)?
                .ok_or_else(|| VirError::Config(format!("berkas {path} tidak ada")))?;
            if store.is_forgotten(vid)? {
                return Err(VirError::Config(format!("berkas {path} sudah dilupakan")));
            }
            vid
        }
        ReadTarget::Version(vid) => {
            if !store.version_exists(vid)? {
                return Err(VirError::Config(format!("versi {vid} tidak ada")));
            }
            vid
        }
    };
    let info = store.version_info(version_id)?;
    let content = store.read_version(version_id)?;
    Ok(ReadResult {
        version_id,
        file_id: info.file_id,
        path: info.path,
        hash: info.hash,
        content,
    })
}

/// Tulis isi sebagai versi baru.
pub fn write(store: &MemoryStore, path: &str, content: &str) -> Result<WriteOutcome> {
    store.write_version(path, content, None)
}

/// Buat tautan antar dua versi.
pub fn link(store: &MemoryStore, from: i64, to: i64, kind: LinkKind) -> Result<()> {
    store.link(from, to, kind)
}

/// Tautan keluar sebuah versi.
pub fn links(store: &MemoryStore, version_id: i64) -> Result<Vec<(i64, LinkKind)>> {
    store.links_from(version_id)
}

/// Tetangga graf tautan (dua arah).
pub fn neighbors(store: &MemoryStore, version_id: i64) -> Result<Vec<i64>> {
    store.neighbors(version_id)
}

/// Beri tag pada berkas (path atau id berkas); mengembalikan id berkas.
pub fn tag(store: &MemoryStore, target: &str, tag: &str) -> Result<i64> {
    let file_id = resolve_file(store, target)?;
    store.tag(file_id, tag)?;
    Ok(file_id)
}

/// Hapus tag dari berkas.
pub fn untag(store: &MemoryStore, target: &str, tag: &str) -> Result<i64> {
    let file_id = resolve_file(store, target)?;
    store.untag(file_id, tag)?;
    Ok(file_id)
}

/// Metadata seluruh versi sebuah path.
pub fn history(store: &MemoryStore, path: &str) -> Result<Vec<VersionInfo>> {
    store.versions(path)
}

/// Diff dua versi (id versi).
pub fn diff(store: &MemoryStore, a: i64, b: i64) -> Result<Diff> {
    let left = store.read_version(a)?;
    let right = store.read_version(b)?;
    let lines = diff_lines(&left, &right);
    let added = lines.iter().filter(|l| l.kind == DiffKind::Add).count();
    let removed = lines.iter().filter(|l| l.kind == DiffKind::Remove).count();
    Ok(Diff {
        a,
        b,
        lines,
        added,
        removed,
    })
}

/// Hapus lunak versi (path atau id); mengembalikan id versi.
pub fn forget(store: &MemoryStore, target: &str) -> Result<i64> {
    let version_id = resolve_version(store, target)?;
    store.forget(version_id)?;
    Ok(version_id)
}

/// Pulihkan versi yang dilupakan.
pub fn restore(store: &MemoryStore, target: &str) -> Result<i64> {
    let version_id = resolve_version(store, target)?;
    store.restore(version_id)?;
    Ok(version_id)
}

fn resolve_version(store: &MemoryStore, target: &str) -> Result<i64> {
    if let Ok(id) = target.trim().parse::<i64>() {
        if store.version_exists(id)? {
            return Ok(id);
        }
        return Err(VirError::Config(format!("versi {id} tidak ada")));
    }
    let path = normalize_path(target)?;
    store
        .current_version(&path)?
        .ok_or_else(|| VirError::Config(format!("berkas {path} tidak ada")))
}

fn resolve_file(store: &MemoryStore, target: &str) -> Result<i64> {
    if let Ok(id) = target.trim().parse::<i64>() {
        if store.path_of(id)?.is_some() {
            return Ok(id);
        }
        return Err(VirError::Config(format!("berkas {id} tidak ada")));
    }
    let path = normalize_path(target)?;
    store
        .file_id(&path)?
        .ok_or_else(|| VirError::Config(format!("berkas {path} tidak ada")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::DEFAULT_DEDUP_THRESHOLD;

    fn store() -> MemoryStore {
        MemoryStore::open_in_memory()
            .unwrap()
            .with_dedup_threshold(DEFAULT_DEDUP_THRESHOLD)
    }

    #[test]
    fn ls_dan_tree() {
        let s = store();
        s.write("/episodik/e1", "satu").unwrap();
        s.write("/episodik/sub/e2", "dua").unwrap();
        s.write("/episodikx/e3", "tiga").unwrap();
        s.write("/semantik/s1", "empat").unwrap();
        let listed = ls(&s, "/episodik").unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().all(|e| e.path != "/episodikx/e3"));
        let root = tree(&s, "/").unwrap();
        assert_eq!(root.children.len(), 3);
        let ep = tree(&s, "/episodik").unwrap();
        assert_eq!(ep.children.len(), 2);
        assert_eq!(ep.children[0].name, "e1");
        assert!(ep.children[1].children[0].path.ends_with("/sub/e2"));
    }

    #[test]
    fn tulis_baca_riwayat_diff() {
        let s = store();
        let v1 = write(&s, "/a.txt", "satu\ndua\n").unwrap().version_id;
        let v2 = write(&s, "/a.txt", "satu\nx\n").unwrap().version_id;
        let r = read(&s, ReadTarget::Path("/a.txt")).unwrap();
        assert_eq!(r.version_id, v2);
        let h = history(&s, "/a.txt").unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].version_id, v1);
        let d = diff(&s, v1, v2).unwrap();
        assert_eq!(d.added, 1);
        assert_eq!(d.removed, 1);
    }

    #[test]
    fn tautan_tag_lupa_pulih() {
        let s = store();
        let v1 = write(&s, "/a.txt", "satu").unwrap().version_id;
        let v2 = write(&s, "/b.txt", "dua").unwrap().version_id;
        link(&s, v1, v2, LinkKind::DerivedFrom).unwrap();
        assert_eq!(links(&s, v1).unwrap(), vec![(v2, LinkKind::DerivedFrom)]);
        assert_eq!(neighbors(&s, v2).unwrap(), vec![v1]);
        let fid = tag(&s, "/a.txt", "kota").unwrap();
        assert_eq!(s.tags(fid).unwrap(), vec!["kota".to_string()]);
        assert!(read(&s, ReadTarget::Version(v1)).is_ok());
        forget(&s, "/a.txt").unwrap();
        assert!(read(&s, ReadTarget::Path("/a.txt")).is_err());
        restore(&s, &v1.to_string()).unwrap();
        assert_eq!(
            read(&s, ReadTarget::Path("/a.txt")).unwrap().content,
            "satu"
        );
    }
}
