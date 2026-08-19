//! Semantic embeddings with pluggable providers, plus hybrid (lexical +
//! semantic) search via reciprocal rank fusion.
//!
//! Providers:
//! - `hash`   — deterministic feature-hashing embedding, fully offline.
//!   Gives fuzzy lexical matching (typos, word variants) with zero
//!   dependencies and instant speed. Not truly semantic.
//! - `openai` — any OpenAI-compatible `/embeddings` endpoint: OpenAI,
//!   Ollama (`http://localhost:11434/v1`), LM Studio, vLLM, ...
//!   This is where real semantic search comes from, local or remote.
//!
//! Embeddings are cached in the index by content hash, so file edits only
//! re-embed changed blocks and moved code keeps its vectors.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::index::{Index, TOMBSTONE_FILE};
use crate::search;

// ---------- hashing ----------

fn fnv64_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

pub fn fnv64(s: &str) -> u64 {
    fnv64_bytes(s.as_bytes())
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Deterministic offline embedding: hashed word tokens + character trigrams,
/// L2-normalized.
fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(16);
    let t = text.to_lowercase();
    let mut v = vec![0.0f32; dim];
    let add = |v: &mut Vec<f32>, bytes: &[u8], w: f32| {
        let h = fnv64_bytes(bytes);
        v[(h % dim as u64) as usize] += w;
        v[((h >> 32) % dim as u64) as usize] += w * 0.5;
    };
    for word in t.split(|c: char| !c.is_alphanumeric()) {
        if word.len() >= 2 {
            add(&mut v, word.as_bytes(), 1.0);
        }
    }
    let b = t.as_bytes();
    if b.len() >= 3 {
        for i in 0..b.len() - 2 {
            add(&mut v, &b[i..i + 3], 0.5);
        }
    }
    normalize(&mut v);
    v
}

// ---------- provider ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProviderKind {
    Hash,
    Http,
}

pub struct Provider {
    pub kind: ProviderKind,
    pub url: String,
    pub model: String,
    pub key: String,
    pub batch: usize,
    pub dim: usize,
}

impl Provider {
    /// Resolve provider settings from CLI options with env-var fallbacks:
    /// EMBED_PROVIDER, EMBED_URL, EMBED_MODEL, EMBED_API_KEY / OPENAI_API_KEY,
    /// EMBED_DIM, EMBED_BATCH.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        provider: Option<&str>,
        model: Option<&str>,
        url: Option<&str>,
        key: Option<&str>,
        dim: Option<usize>,
        batch: Option<usize>,
    ) -> Provider {
        let env = |name: &str| std::env::var(name).ok().filter(|s| !s.is_empty());
        let name = provider
            .map(str::to_string)
            .or_else(|| env("EMBED_PROVIDER"))
            .unwrap_or_else(|| "hash".into());
        match name.as_str() {
            "openai" | "http" | "remote" => Provider {
                kind: ProviderKind::Http,
                url: url
                    .map(str::to_string)
                    .or_else(|| env("EMBED_URL"))
                    .unwrap_or_else(|| "https://api.openai.com/v1".into()),
                model: model
                    .map(str::to_string)
                    .or_else(|| env("EMBED_MODEL"))
                    .unwrap_or_else(|| "text-embedding-3-small".into()),
                key: key
                    .map(str::to_string)
                    .or_else(|| env("EMBED_API_KEY"))
                    .or_else(|| env("OPENAI_API_KEY"))
                    .unwrap_or_default(),
                batch: batch
                    .or_else(|| env("EMBED_BATCH").and_then(|s| s.parse().ok()))
                    .unwrap_or(64),
                dim: 0,
            },
            _ => Provider {
                kind: ProviderKind::Hash,
                url: String::new(),
                model: "hash".into(),
                key: String::new(),
                batch: batch
                    .or_else(|| env("EMBED_BATCH").and_then(|s| s.parse().ok()))
                    .unwrap_or(4096),
                dim: dim
                    .or_else(|| env("EMBED_DIM").and_then(|s| s.parse().ok()))
                    .unwrap_or(256),
            },
        }
    }

    /// Stable identity of this provider's embedding space. Embeddings from a
    /// different provider/model are invalid and must be recomputed.
    pub fn id(&self) -> String {
        match self.kind {
            ProviderKind::Hash => format!("hash:{}", self.dim),
            ProviderKind::Http => format!("http:{}@{}", self.model, self.url),
        }
    }

    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        match self.kind {
            ProviderKind::Hash => Ok(texts.iter().map(|t| hash_embed(t, self.dim)).collect()),
            ProviderKind::Http => self.http_embed(texts),
        }
    }

    fn http_embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        #[derive(Deserialize)]
        struct EmbedResp {
            data: Vec<EmbedDatum>,
        }
        #[derive(Deserialize)]
        struct EmbedDatum {
            embedding: Vec<f32>,
            #[serde(default)]
            index: usize,
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch.max(1)) {
            let mut req = ureq::post(&format!("{}/embeddings", self.url));
            if !self.key.is_empty() {
                req = req.set("Authorization", &format!("Bearer {}", self.key));
            }
            let resp = req
                .timeout(Duration::from_secs(300))
                .send_json(json!({ "model": self.model, "input": chunk }))
                .map_err(|e| format!("embedding request failed: {e}"))?;
            let parsed: EmbedResp = resp
                .into_json()
                .map_err(|e| format!("bad embedding response: {e}"))?;
            let mut data = parsed.data;
            data.sort_by_key(|d| d.index); // stable sort keeps order if index missing
            if data.len() != chunk.len() {
                return Err(format!(
                    "embedding count mismatch: sent {}, got {}",
                    chunk.len(),
                    data.len()
                ));
            }
            for mut d in data {
                normalize(&mut d.embedding);
                out.push(d.embedding);
            }
        }
        Ok(out)
    }
}

// ---------- embedding store ----------

#[derive(Serialize, Deserialize, Default)]
pub struct Embeddings {
    /// Provider identity these vectors belong to.
    pub provider_id: String,
    pub dim: usize,
    /// content hash (fnv64 of block text) -> L2-normalized vector
    pub map: HashMap<u64, Vec<f32>>,
}

/// Ensure every live block has an embedding, computing the missing ones.
pub fn ensure(
    emb: &mut Embeddings,
    prov: &Provider,
    index: &Index,
    progress: bool,
) -> Result<(), String> {
    if emb.provider_id != prov.id() {
        *emb = Embeddings {
            provider_id: prov.id(),
            dim: match prov.kind {
                ProviderKind::Hash => prov.dim,
                ProviderKind::Http => 0,
            },
            map: HashMap::new(),
        };
    }
    let missing: HashMap<u64, String> = index
        .blocks
        .par_iter()
        .filter(|b| b.file != TOMBSTONE_FILE)
        .filter(|b| !emb.map.contains_key(&fnv64(&b.text)))
        .map(|b| (fnv64(&b.text), b.text.clone()))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let mut items: Vec<(u64, String)> = missing.into_iter().collect();
    items.sort_unstable_by_key(|(h, _)| *h); // deterministic order
    let total = items.len();
    if progress {
        eprintln!("  embedding {total} block(s) with {} ...", prov.id());
    }
    let mut done = 0usize;
    for chunk in items.chunks(prov.batch.max(1)) {
        let refs: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        let vecs = prov.embed(&refs)?;
        if emb.dim == 0 {
            if let Some(v) = vecs.first() {
                emb.dim = v.len();
            }
        }
        for ((h, _), v) in chunk.iter().zip(vecs) {
            emb.map.insert(*h, v);
        }
        done += chunk.len();
        if progress && total > 2000 {
            eprintln!("    {done}/{total}");
        }
    }
    Ok(())
}

/// Ensure all embeddings exist for `idx` (handles the field borrow).
pub fn ensure_all(idx: &mut Index, prov: &Provider) -> Result<(), String> {
    let mut emb = std::mem::take(&mut idx.embeddings);
    let r = ensure(&mut emb, prov, idx, true);
    idx.embeddings = emb;
    r
}

impl Embeddings {
    /// Drop vectors for blocks that no longer exist.
    pub fn gc(&mut self, live: &HashSet<u64>) {
        self.map.retain(|k, _| live.contains(k));
    }
}

// ---------- vector + hybrid search ----------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Top-k blocks by cosine similarity (vectors are normalized, so dot product).
pub fn top_k(index: &Index, emb: &Embeddings, q: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut scores: Vec<(f32, u32)> = index
        .blocks
        .par_iter()
        .enumerate()
        .filter(|(_, b)| b.file != TOMBSTONE_FILE)
        .filter_map(|(i, b)| emb.map.get(&fnv64(&b.text)).map(|v| (dot(q, v), i as u32)))
        .collect();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    scores.truncate(k);
    scores
}

/// Hybrid search: fuse lexical (trigram) and semantic rankings with
/// reciprocal rank fusion. Returns hits with their fused score.
pub fn ask(
    idx: &mut Index,
    prov: &Provider,
    query: &str,
    limit: usize,
) -> Result<(Vec<(f32, search::Hit)>, bool), String> {
    let before = idx.embeddings.map.len();
    let mut emb = std::mem::take(&mut idx.embeddings);
    let result = (|| -> Result<Vec<(f32, search::Hit)>, String> {
        ensure(&mut emb, prov, idx, true)?;
        let qv = prov.embed(&[query])?.remove(0);
        let sem = top_k(idx, &emb, &qv, 200);
        let lex = search::search(idx, query, 200);
        Ok(fuse(idx, &lex, &sem, query, limit))
    })();
    idx.embeddings = emb;
    result.map(|hits| (hits, idx.embeddings.map.len() != before))
}

fn fuse(
    idx: &Index,
    lex: &[search::Hit],
    sem: &[(f32, u32)],
    query: &str,
    limit: usize,
) -> Vec<(f32, search::Hit)> {
    // Reciprocal rank fusion.
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for (r, h) in lex.iter().enumerate() {
        *scores.entry(h.block_id).or_default() += 1.0 / (60.0 + r as f32);
    }
    for (r, (_, bid)) in sem.iter().enumerate() {
        *scores.entry(*bid).or_default() += 1.0 / (60.0 + r as f32);
    }
    let mut ranked: Vec<(f32, u32)> = scores.into_iter().map(|(bid, s)| (s, bid)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(limit);
    let q = query.to_lowercase();
    ranked
        .into_iter()
        .map(|(score, bid)| {
            let b = &idx.blocks[bid as usize];
            let occurrences = b.text.to_lowercase().match_indices(&q).count();
            (
                score,
                search::Hit {
                    block_id: bid,
                    file_id: b.file,
                    line: b.line,
                    text: b.text.clone(),
                    occurrences,
                },
            )
        })
        .collect()
}
