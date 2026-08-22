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
//! indexed project directories stay clean. `ENDEX_CACHE_DIR` overrides the
//! `~/.endex/cache` root (useful for tests and sandboxed environments).
//!
//! The v5 payload carries a trailing CRC-32 checksum so `load` can detect a
//! corrupt or truncated cache and fall back to a rebuild. This is an
//! accident detector, NOT a security boundary: anyone who can write the
//! cache file can trivially recompute the checksum, exactly as they could
//! recompute a bare SHA-256. Since a keyed MAC is the only thing that would
//! change that, and we aren't doing key management for a rebuildable cache,
//! we use the cheapest adequate check — `crc32fast` is already in the
//! dependency tree via `ureq -> flate2`, whereas SHA-256 pulled in seven
//! extra crates to solve a threat model we don't have.
//!
//! v4 files (payload only) are still *read* so an upgrade does not throw
//! away expensive embedding vectors; they are rewritten as v5 on the next
//! save.

use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Current format: MAGIC | bincode payload | CRC-32(payload), little-endian.
const MAGIC: &[u8; 10] = b"ENDEXIDX\x05\x00";
/// Legacy format: MAGIC | bincode payload. Same schema as v5, no checksum.
/// Read-only — never written.
const MAGIC_V4: &[u8; 10] = b"ENDEXIDX\x04\x00";

/// Bytes in the trailing CRC-32 checksum appended to the payload.
pub const CHECKSUM_LEN: usize = 4;

fn checksum(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

/// Cache format version of the current MAGIC header (bumped on schema change).
pub const CACHE_VERSION: u32 = 5;

pub const INDEX_FILE: &str = "index.bin";
pub const MANIFEST_FILE: &str = "manifest.json";

/// Filenames the tool writes into its cache directory. Walkers and watchers
/// must skip them *inside that directory* or the index re-ingests its own
/// output forever — see `is_self_written`, which is path-scoped on purpose:
/// `manifest.json` is a perfectly ordinary source file in most projects and
/// must never be filtered by name alone.
pub const SELF_WRITTEN: &[&str] = &[
    INDEX_FILE,
    "index.bin.tmp",
    MANIFEST_FILE,
    "manifest.json.tmp",
];

/// True if `path` is one of our own cache artifacts, i.e. it lives directly
/// in `cache_dir` and has one of the `SELF_WRITTEN` names. Callers should
/// compute `cache_dir(root)` once and reuse it (it canonicalizes).
pub fn is_self_written(cache_dir: &Path, path: &Path) -> bool {
    path.parent() == Some(cache_dir)
        && path
            .file_name()
            .is_some_and(|n| SELF_WRITTEN.contains(&n.to_string_lossy().as_ref()))
}

/// Sibling temp path for atomic writes: `index.bin` -> `index.bin.tmp`.
/// (`Path::with_extension` would produce `index.tmp`, which does not match
/// `SELF_WRITTEN` and would leak into the index.)
fn tmp_path(p: &Path) -> PathBuf {
    let mut s: OsString = p.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

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
/// `ENDEX_CACHE_DIR` overrides the `~/.endex/cache` root.
///
/// Canonicalizes `root`, so callers in hot paths should call it once.
pub fn cache_dir(root: &Path) -> PathBuf {
    let base = match env::var_os("ENDEX_CACHE_DIR").filter(|v| !v.is_empty()) {
        Some(v) => Some(PathBuf::from(v)),
        None => env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .or_else(|| env::var_os("USERPROFILE").filter(|v| !v.is_empty()))
            .map(|h| PathBuf::from(h).join(".endex").join("cache")),
    };
    let Some(base) = base else {
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
    base.join(format!("{name}-{hash:016x}"))
}

pub fn cache_path(root: &Path) -> PathBuf {
    cache_dir(root).join(INDEX_FILE)
}

pub fn manifest_path(root: &Path) -> PathBuf {
    cache_dir(root).join(MANIFEST_FILE)
}

pub fn save(index: &Index, root: &Path) -> io::Result<()> {
    // One canonicalizing lookup for both files.
    let dir = cache_dir(root);
    let path = dir.join(INDEX_FILE);
    fs::create_dir_all(&dir)?;
    let t0 = std::time::Instant::now();
    // Layout: MAGIC | bincode payload | CRC-32(payload) little-endian.
    let data = bincode::serialize(index).map_err(io::Error::other)?;
    let mut buf = Vec::with_capacity(data.len() + MAGIC.len() + CHECKSUM_LEN);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&data);
    buf.extend_from_slice(&checksum(&data).to_le_bytes());

    let tmp = tmp_path(&path);
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
        let mpath = dir.join(MANIFEST_FILE);
        let mtmp = tmp_path(&mpath);
        if fs::write(&mtmp, &json).is_ok() {
            let _ = fs::rename(&mtmp, &mpath);
        }
    }
    Ok(())
}

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

/// Read the manifest without touching the big index. Returns None if the
/// manifest is missing/corrupt — callers should then fall back to a full
/// load.
pub fn load_manifest(root: &Path) -> Option<Manifest> {
    let s = fs::read_to_string(manifest_path(root)).ok()?;
    serde_json::from_str(&s).ok()
}

pub fn load(root: &Path) -> Option<Index> {
    let path = cache_path(root);
    let buf = fs::read(&path).ok()?;
    if buf.len() < MAGIC.len() {
        return None;
    }
    let (header, payload) = buf.split_at(MAGIC.len());
    // The header alone decides the framing — never guess from length, or a
    // legacy payload gets read as checksum-protected and always "fails".
    let body = match header {
        h if h == MAGIC => {
            let (b, tag) = payload.split_at(payload.len().checked_sub(CHECKSUM_LEN)?);
            if checksum(b).to_le_bytes() != tag {
                return None; // corrupt or truncated → rebuild
            }
            b
        }
        // v4: identical schema, no checksum. Accepted so upgrades keep their
        // embedding vectors; the next save rewrites the file as v5.
        h if h == MAGIC_V4 => payload,
        _ => return None, // unknown/corrupt cache
    };
    bincode::deserialize(body).ok()
}
