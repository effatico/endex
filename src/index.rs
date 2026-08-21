//! Trigram inverted index over "code blocks" (blank-line-separated chunks),
//! plus the knowledge graph and embedding store derived from them.
//!
//! Design (same family as Google Code Search / Zoekt):
//!   - The corpus is split into *blocks* (contiguous non-blank lines, capped).
//!   - Every 3-byte window (lowercased) of a block maps to a sorted posting
//!     list of block ids. A substring query intersects the posting lists of
//!     its trigrams, yielding a tiny candidate set that is then verified.
//!   - This makes arbitrary substring search run in microseconds-to-ms even
//!     over multi-GB corpora.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::embed::Embeddings;
use crate::graph::{self, Def, Graph};

pub const MAX_FILE_SIZE: u64 = 5 * 1024 * 1024; // skip files larger than 5 MB
pub const BLOCK_MAX_LINES: usize = 80; // split blocks longer than this
pub const TOMBSTONE_FILE: u32 = u32::MAX;

#[derive(Serialize, Deserialize)]
pub struct FileEntry {
    pub id: u32,
    pub mtime: u64,
    pub len: u64,
    pub blocks: Vec<u32>,
    /// definitions extracted at index time (sorted by line)
    #[serde(default)]
    pub defs: Vec<Def>,
    /// content hash (fnv64 of the file's text). Used for the corpus
    /// fingerprint in the cache manifest — a cheap staleness canary.
    /// Change *detection* for reindexing is mtime+len based (above).
    #[serde(default)]
    pub content_hash: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Block {
    pub file: u32,
    pub line: u32, // 1-based line of block start
    pub text: String,
    /// fnv64 of `text`, computed once at parse time. Embedding lookups,
    /// coverage checks and GC all key off this — recomputing it per query
    /// costs a full corpus hash pass on every call.
    #[serde(default)]
    pub hash: u64,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Index {
    pub root: PathBuf,
    pub files: HashMap<PathBuf, FileEntry>,
    pub file_ids: Vec<String>, // file id -> path ("" = tombstone)
    pub free_files: Vec<u32>,
    pub blocks: Vec<Block>, // block id -> block (file == TOMBSTONE_FILE = free)
    pub free_blocks: Vec<u32>,
    /// trigram (3 bytes packed into u32) -> sorted list of block ids
    pub postings: HashMap<u32, Vec<u32>>,
    /// knowledge graph (symbols, call edges, imports)
    #[serde(default)]
    pub graph: Graph,
    /// semantic embeddings, keyed by block content hash
    #[serde(default)]
    pub embeddings: Embeddings,
}

// ---------- parsing ----------

struct ParsedBlock {
    line: u32,
    text: String,
    trigrams: Vec<u32>,
    hash: u64,
}

struct ParsedFile {
    path: PathBuf,
    mtime: u64,
    len: u64,
    blocks: Vec<ParsedBlock>,
    defs: Vec<Def>,
    content_hash: u64,
}

/// Split text into blank-line-separated blocks (capped at BLOCK_MAX_LINES).
fn parse_blocks(text: &str) -> Vec<ParsedBlock> {
    let mut blocks = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut start_line: u32 = 1;

    fn flush(cur: &mut Vec<&str>, start_line: u32, blocks: &mut Vec<ParsedBlock>) {
        if cur.is_empty() {
            return;
        }
        let text = cur.join("\n");
        let trigrams = block_trigrams(&text);
        let hash = crate::embed::fnv64(&text);
        blocks.push(ParsedBlock {
            line: start_line,
            text,
            trigrams,
            hash,
        });
        cur.clear();
    }

    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            flush(&mut cur, start_line, &mut blocks);
            start_line = (i + 2) as u32;
        } else {
            cur.push(line);
            if cur.len() >= BLOCK_MAX_LINES {
                flush(&mut cur, start_line, &mut blocks);
                start_line = (i + 2) as u32;
            }
        }
    }
    flush(&mut cur, start_line, &mut blocks);
    blocks
}

/// Unique lowercased byte-trigrams of a block's text.
fn block_trigrams(text: &str) -> Vec<u32> {
    let lower = text.to_lowercase();
    let b = lower.as_bytes();
    if b.len() < 3 {
        return Vec::new();
    }
    let mut set: HashSet<u32> = HashSet::with_capacity(b.len() / 2);
    for w in b.windows(3) {
        set.insert((w[0] as u32) << 16 | (w[1] as u32) << 8 | w[2] as u32);
    }
    set.into_iter().collect()
}

pub fn is_binary(bytes: &[u8]) -> bool {
    let n = bytes.len().min(8192);
    bytes[..n].contains(&0)
}

fn mtime_of(meta: &fs::Metadata) -> u64 {
    use std::time::UNIX_EPOCH;
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// ---------- index operations ----------

impl Index {
    pub fn new(root: &Path) -> Self {
        Index {
            root: root.to_path_buf(),
            ..Default::default()
        }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len() - self.free_blocks.len()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn path_of(&self, file_id: u32) -> &str {
        self.file_ids
            .get(file_id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// Tombstone a block: the slot is recycled via `free_blocks`. Its stale
    /// posting-list entries are deliberately NOT removed — query-time
    /// verification checks the block's current text anyway, and ids are
    /// recycled, so stale entries are transient and correctness never
    /// depends on their removal. This turns a reindex from
    /// O(trigrams x posting-list-length) into O(trigrams).
    fn tombstone_block(&mut self, block_id: u32) {
        self.blocks[block_id as usize] = Block {
            file: TOMBSTONE_FILE,
            line: 0,
            text: String::new(),
            hash: 0,
        };
        self.free_blocks.push(block_id);
    }

    fn alloc_block_id(&mut self) -> u32 {
        self.free_blocks.pop().unwrap_or_else(|| {
            self.blocks.push(Block {
                file: TOMBSTONE_FILE,
                line: 0,
                text: String::new(),
                hash: 0,
            });
            (self.blocks.len() - 1) as u32
        })
    }

    /// Incrementally (re)index a single file. No-op if unchanged.
    /// Returns true if the index changed.
    pub fn index_file(&mut self, path: &Path) -> bool {
        let meta = match fs::metadata(path) {
            Ok(m) if m.is_file() => m,
            _ => {
                return self.remove_file(path);
            }
        };
        let mtime = mtime_of(&meta);
        let len = meta.len();
        if let Some(e) = self.files.get(path) {
            if e.mtime == mtime && e.len == len {
                return false; // unchanged
            }
        }
        if len > MAX_FILE_SIZE {
            return self.remove_file(path);
        }
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(_) => return self.remove_file(path),
        };
        if is_binary(&bytes) {
            return self.remove_file(path);
        }
        let text = String::from_utf8_lossy(&bytes);
        let content_hash = crate::embed::fnv64(&text);
        let parsed = parse_blocks(&text);
        let defs = graph::extract_defs(&path.to_string_lossy(), &text);

        // Remove old blocks for this file (tombstone only — see above).
        if let Some(old) = self.files.remove(path) {
            for &b in &old.blocks {
                self.tombstone_block(b);
            }
            self.file_ids[old.id as usize] = String::new();
            self.free_files.push(old.id);
        }

        // Assign file id.
        let file_id = self.free_files.pop().unwrap_or_else(|| {
            self.file_ids.push(String::new());
            (self.file_ids.len() - 1) as u32
        });
        self.file_ids[file_id as usize] = path.to_string_lossy().into_owned();

        let mut block_ids = Vec::with_capacity(parsed.len());
        for pb in parsed {
            let id = self.alloc_block_id();
            self.blocks[id as usize] = Block {
                file: file_id,
                line: pb.line,
                text: pb.text,
                hash: pb.hash,
            };
            for t in pb.trigrams {
                let v = self.postings.entry(t).or_default();
                // Guard against duplicates: recycled block ids may already
                // have a stale entry for this trigram.
                match v.binary_search(&id) {
                    Ok(_) => {}
                    Err(pos) => v.insert(pos, id),
                }
            }
            block_ids.push(id);
        }
        self.files.insert(
            path.to_path_buf(),
            FileEntry {
                id: file_id,
                mtime,
                len,
                blocks: block_ids,
                defs,
                content_hash,
            },
        );
        true
    }

    /// Remove a file and all its blocks from the index. Returns true if changed.
    pub fn remove_file(&mut self, path: &Path) -> bool {
        if let Some(old) = self.files.remove(path) {
            for &b in &old.blocks {
                self.tombstone_block(b);
            }
            self.file_ids[old.id as usize] = String::new();
            self.free_files.push(old.id);
            true
        } else {
            false
        }
    }

    /// Full parallel build over a directory tree (respecting .gitignore etc).
    pub fn build(&mut self, root: &Path) {
        *self = Index::new(root); // full rebuild: drop any stale state
        let paths: Vec<PathBuf> = walk_files(root);

        let t0 = std::time::Instant::now();
        let parsed: Vec<Option<ParsedFile>> = paths
            .par_iter()
            .map(|p| {
                let meta = fs::metadata(p).ok()?;
                if !meta.is_file() || meta.len() > MAX_FILE_SIZE {
                    return None;
                }
                let bytes = fs::read(p).ok()?;
                if is_binary(&bytes) {
                    return None;
                }
                let text = String::from_utf8_lossy(&bytes);
                Some(ParsedFile {
                    path: p.clone(),
                    mtime: mtime_of(&meta),
                    len: meta.len(),
                    content_hash: crate::embed::fnv64(&text),
                    blocks: parse_blocks(&text),
                    defs: graph::extract_defs(&p.to_string_lossy(), &text),
                })
            })
            .collect();

        let n_files = parsed.iter().flatten().count();
        let n_blocks: usize = parsed.iter().flatten().map(|f| f.blocks.len()).sum();

        // Sequential merge: block ids assigned in ascending order -> posting
        // lists come out sorted for free.
        for file in parsed.into_iter().flatten() {
            self.files.remove(&file.path); // fresh build; drop any stale entry
            let file_id = self.file_ids.len() as u32;
            self.file_ids.push(file.path.to_string_lossy().into_owned());
            let mut ids = Vec::with_capacity(file.blocks.len());
            for pb in file.blocks {
                let id = self.blocks.len() as u32;
                self.blocks.push(Block {
                    file: file_id,
                    line: pb.line,
                    text: pb.text,
                    hash: pb.hash,
                });
                for t in pb.trigrams {
                    self.postings.entry(t).or_default().push(id);
                }
                ids.push(id);
            }
            self.files.insert(
                file.path,
                FileEntry {
                    id: file_id,
                    mtime: file.mtime,
                    len: file.len,
                    blocks: ids,
                    defs: file.defs,
                    content_hash: file.content_hash,
                },
            );
        }
        let t1 = std::time::Instant::now();
        let g = graph::rebuild(self);
        self.graph = g;
        eprintln!(
            "  parsed {} files / {} blocks in {:?}, graph {} symbols / {} call edges / {} imports in {:?} ({} trigram posting lists)",
            n_files,
            n_blocks,
            t0.elapsed(),
            self.graph.symbols.len(),
            self.graph.call_edge_count(),
            self.graph.file_imports.len(),
            t1.elapsed(),
            self.postings.len()
        );
    }

    /// Refresh against disk: reindex changed/new files, drop deleted ones.
    pub fn refresh(&mut self, root: &Path) -> usize {
        let current: HashSet<PathBuf> = walk_files(root).into_iter().collect();
        let mut changed = 0usize;

        // Removed files.
        let stale: Vec<PathBuf> = self
            .files
            .keys()
            .filter(|p| !current.contains(*p))
            .cloned()
            .collect();
        for p in stale {
            if self.remove_file(&p) {
                changed += 1;
            }
        }
        // New or modified files.
        for p in &current {
            if self.index_file(p) {
                changed += 1;
            }
        }
        if changed > 0 {
            self.finalize();
        }
        changed
    }

    /// Recompute derived state after incremental updates: rebuild the
    /// knowledge graph (pure in-memory pass) and GC dead embeddings.
    pub fn finalize(&mut self) {
        let g = graph::rebuild(self);
        self.graph = g;
        let live: HashSet<u64> = self
            .blocks
            .iter()
            .filter(|b| b.file != TOMBSTONE_FILE)
            .map(|b| b.hash)
            .collect();
        self.embeddings.gc(&live);
    }

    /// Corpus fingerprint: XOR of all file content hashes, combined with the
    /// file count. Two indexes of the same tree have the same fingerprint;
    /// any content change flips it. Used by the cache manifest to detect
    /// staleness at a glance.
    pub fn corpus_fingerprint(&self) -> u64 {
        let mut h = self.files.len() as u64;
        for e in self.files.values() {
            h ^= e.content_hash;
        }
        h
    }
}

/// Collect indexable files using the `ignore` walker (gitignore-aware,
/// skips hidden dirs and .git).
pub fn walk_files(root: &Path) -> Vec<PathBuf> {
    use ignore::WalkBuilder;
    let mut out = Vec::new();
    for entry in WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .follow_links(false)
        // Never index our own cache/manifest files, even in repos that
        // don't gitignore them.
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !crate::store::SELF_WRITTEN.contains(&name.as_ref())
        })
        .build()
        .flatten()
    {
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            out.push(entry.into_path());
        }
    }
    out
}
