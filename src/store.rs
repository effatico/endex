//! Disk cache: bincode-serialized index with a magic header, written
//! atomically (tmp file + rename) so a crash never corrupts the cache.
//!
//! Alongside the index we keep a tiny JSON *manifest* (`manifest.json`)
//! recording the embedding provider and the corpus fingerprint. On load we
//! can then cheaply answer: is this cache for the same provider? is it
//! fresh for the current tree? — without deserializing the full index.
//!
//! Caches live in `~/.endex/cache/<repo_name>-<hash_of_project_path>/`
//! (falling back to the project dir itself when there is no home dir), so
//! indexed project directories stay clean.

use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io;
use std::path::Path;

const MAGIC: &[u8; 10] = b"ENDEXIDX\x04\x00"; // v4: + per-block content hash

/// Cache format version of the current MAGIC header (bumped on schema change).
pub const CACHE_VERSION: u32 = 4;

/// Filenames the tool writes into its cache directory (`~/.endex/cache/...`).
/// Watchers and walkers must ignore these names or the index re-ingests its
/// own output forever (relevant when the cache falls back into the project
/// dir, or when the project dir is a cache dir).
pub const SELF_WRITTEN: &[&str] = &[
    "index.bin",
    "index.bin.tmp",
    "manifest.json",
    "manifest.json.tmp",
];

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    /// Embedding provider identity (e.g. "http:qwen3-embedding@http://..."),
    /// or "" if no vectors were stored.
    pub embedding_provider: String,
    /// XOR of all file content hashes + file count — staleness canary.
    pub corpus_fingerprint: u64,
    pub files: usize,
    pub blocks: usize,
    pub embedding_vectors: usize,
    pub embedding_dim: usize,
}

/// Directory that holds this project's cache files:
/// `~/.endex/cache/<repo_name>-<hash_of_project_path>` (falls back to the
/// project dir itself when there is no home directory). The hash suffix
/// keeps two different projects with the same directory name apart.
fn cache_dir(root: &Path) -> std::path::PathBuf {
    let home = env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(std::path::PathBuf::from));
    let Some(home) = home else {
        return root.to_path_buf();
    };
    let canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let name = canon
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".into());
    // Sanitize: filename chars are safe to use as-is in a path component.
    let name: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let hash = crate::embed::fnv64(&canon.to_string_lossy());
    home.join(".endex")
        .join("cache")
        .join(format!("{name}-{hash:016x}"))
}

pub fn cache_path(root: &Path) -> std::path::PathBuf {
    cache_dir(root).join("index.bin")
}

pub fn manifest_path(root: &Path) -> std::path::PathBuf {
    cache_dir(root).join("manifest.json")
}

pub fn save(index: &Index, root: &Path) -> io::Result<()> {
    let path = cache_path(root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let t0 = std::time::Instant::now();
    let mut data = bincode::serialize(index).map_err(io::Error::other)?;
    let mut buf = Vec::with_capacity(data.len() + MAGIC.len());
    buf.extend_from_slice(MAGIC);
    buf.append(&mut data);

    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &buf)?;
    fs::rename(&tmp, &path)?; // atomic on POSIX
    eprintln!(
        "  cache saved: {:.1} MB in {:?}",
        buf.len() as f64 / (1024.0 * 1024.0),
        t0.elapsed()
    );

    // Manifest: small, human-readable staleness canary. Best-effort.
    let manifest = Manifest {
        embedding_provider: index.embeddings.provider_id.clone(),
        corpus_fingerprint: index.corpus_fingerprint(),
        files: index.file_count(),
        blocks: index.block_count(),
        embedding_vectors: index.embeddings.map.len(),
        embedding_dim: index.embeddings.dim,
    };
    if let Ok(json) = serde_json::to_string_pretty(&manifest) {
        let mtmp = manifest_path(root).with_extension("tmp");
        if fs::write(&mtmp, &json).is_ok() {
            let _ = fs::rename(&mtmp, manifest_path(root));
        }
    }
    Ok(())
}

/// Read the manifest without touching the big index. Returns None if the
/// manifest is missing/corrupt — callers should then fall back to a full
/// load.
/// Cache file metadata without deserializing the index: path, size,
/// modified time. Combined with `load_manifest` by callers.
pub struct CacheInfo {
    pub path: String,
    pub bytes: u64,
    /// seconds since the cache was last written (None if unavailable)
    pub age_seconds: Option<u64>,
}

pub fn cache_info(root: &Path) -> Option<CacheInfo> {
    let path = cache_path(root);
    let meta = std::fs::metadata(&path).ok()?;
    let modified = meta.modified().ok();
    let age = modified
        .and_then(|m| std::time::SystemTime::now().duration_since(m).ok())
        .map(|d| d.as_secs());
    Some(CacheInfo {
        path: path.to_string_lossy().into_owned(),
        bytes: meta.len(),
        age_seconds: age,
    })
}

pub fn load_manifest(root: &Path) -> Option<Manifest> {
    let s = fs::read_to_string(manifest_path(root)).ok()?;
    serde_json::from_str(&s).ok()
}

pub fn load(root: &Path) -> Option<Index> {
    let path = cache_path(root);
    let buf = fs::read(&path).ok()?;
    if buf.len() < MAGIC.len() || &buf[..MAGIC.len()] != MAGIC {
        return None; // unknown/corrupt cache
    }
    bincode::deserialize(&buf[MAGIC.len()..]).ok()
}
