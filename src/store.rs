//! Disk cache: bincode-serialized index with a magic header, written
//! atomically (tmp file + rename) so a crash never corrupts the cache.
//!
//! Alongside the index we keep a tiny JSON *manifest* (`.endex-manifest.json`)
//! recording the embedding provider and the corpus fingerprint. On load we
//! can then cheaply answer: is this cache for the same provider? is it
//! fresh for the current tree? — without deserializing the full index.

use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

const MAGIC: &[u8; 10] = b"ENDEXIDX\x03\x00"; // v3: + per-file content_hash, manifest

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

pub fn cache_path(root: &Path) -> std::path::PathBuf {
    root.join(".endex-index.bin")
}

pub fn manifest_path(root: &Path) -> std::path::PathBuf {
    root.join(".endex-manifest.json")
}

pub fn save(index: &Index, root: &Path) -> io::Result<()> {
    let path = cache_path(root);
    let t0 = std::time::Instant::now();
    let mut data =
        bincode::serialize(index).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
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
