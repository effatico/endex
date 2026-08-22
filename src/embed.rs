//! Semantic embeddings with pluggable providers, plus hybrid (lexical +
//! semantic) search via reciprocal rank fusion and optional reranking.
//!
//! Providers:
//! - `cohere` (default) — Cohere `/embed` (v2): asymmetric search_document/
//!   search_query embeddings, plus the `/rerank` endpoint which reorders
//!   the fused candidates of every `ask` query. Needs COHERE_API_KEY.
//! - `openai` — any OpenAI-compatible `/embeddings` endpoint: OpenAI,
//!   Ollama (`http://localhost:11434/v1`), LM Studio, vLLM, ...
//! - `hash`   — deterministic feature-hashing embedding, fully offline.
//!   Gives fuzzy lexical matching (typos, word variants) with zero
//!   dependencies and instant speed. Not truly semantic.
//!
//! Only Cohere is actively maintained; the others remain as fallbacks.
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
    Cohere,
}

#[derive(Clone)]
pub struct Provider {
    pub kind: ProviderKind,
    pub url: String,
    pub model: String,
    pub key: String,
    pub batch: usize,
    pub dim: usize,
    /// Cohere rerank model ("rerank-v3.5"); empty when the provider has
    /// no reranking endpoint (all non-Cohere providers).
    pub rerank_model: String,
}

/// Options for `Provider::resolve`. Every field is optional; anything unset
/// falls back to the EMBED_* environment variables, then to defaults.
#[derive(Default, Clone)]
pub struct ProviderOpts {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub url: Option<String>,
    pub key: Option<String>,
    pub dim: Option<usize>,
    pub batch: Option<usize>,
    pub rerank_model: Option<String>,
}

impl Provider {
    /// Resolve provider settings from CLI options with env-var fallbacks:
    /// EMBED_PROVIDER, EMBED_URL, EMBED_MODEL, EMBED_API_KEY /
    /// COHERE_API_KEY / OPENAI_API_KEY, EMBED_DIM, EMBED_BATCH,
    /// EMBED_RERANK_MODEL. The default provider is Cohere.
    pub fn resolve(opts: &ProviderOpts) -> Provider {
        Self::resolve_checked(opts).0
    }

    /// `resolve`, plus a one-line notice to show the user when the choice
    /// was silently changed.
    ///
    /// Cohere is the default but needs an API key. When it was *defaulted*
    /// into (not explicitly requested) and no key is configured, fall back
    /// to the offline `hash` provider: otherwise every query pays a failing
    /// network round-trip, and a default should never ship code off the
    /// machine without the user opting in. An explicit `cohere` request is
    /// always honored — the error then tells the user what is wrong.
    pub fn resolve_checked(opts: &ProviderOpts) -> (Provider, Option<String>) {
        let env = |name: &str| std::env::var(name).ok().filter(|s| !s.is_empty());
        let explicit = opts.provider.clone().or_else(|| env("EMBED_PROVIDER"));
        let defaulted = explicit.is_none();
        let prov = Self::build(opts, explicit.unwrap_or_else(|| "cohere".into()));
        if defaulted && prov.kind == ProviderKind::Cohere && prov.key.is_empty() {
            let fallback = Self::build(opts, "hash".into());
            return (
                fallback,
                Some(
                    "no COHERE_API_KEY / EMBED_API_KEY set — using the offline 'hash' provider; \
                     set a key (or --embed-provider cohere) for real semantic search"
                        .into(),
                ),
            );
        }
        (prov, None)
    }

    fn build(opts: &ProviderOpts, name: String) -> Provider {
        let env = |name: &str| std::env::var(name).ok().filter(|s| !s.is_empty());
        match name.as_str() {
            "cohere" => Provider {
                kind: ProviderKind::Cohere,
                url: opts
                    .url
                    .clone()
                    .or_else(|| env("EMBED_URL"))
                    .unwrap_or_else(|| "https://api.cohere.com/v2".into()),
                model: opts
                    .model
                    .clone()
                    .or_else(|| env("EMBED_MODEL"))
                    .unwrap_or_else(|| "embed-v4.0".into()),
                key: opts
                    .key
                    .clone()
                    .or_else(|| env("EMBED_API_KEY"))
                    .or_else(|| env("COHERE_API_KEY"))
                    .unwrap_or_default(),
                batch: opts
                    .batch
                    .or_else(|| env("EMBED_BATCH").and_then(|s| s.parse().ok()))
                    .unwrap_or(96), // Cohere's per-request limit
                dim: 0,
                rerank_model: opts
                    .rerank_model
                    .clone()
                    .or_else(|| env("EMBED_RERANK_MODEL"))
                    .unwrap_or_else(|| "rerank-v3.5".into()),
            },
            "openai" | "http" | "remote" => Provider {
                kind: ProviderKind::Http,
                url: opts
                    .url
                    .clone()
                    .or_else(|| env("EMBED_URL"))
                    .unwrap_or_else(|| "https://api.openai.com/v1".into()),
                model: opts
                    .model
                    .clone()
                    .or_else(|| env("EMBED_MODEL"))
                    .unwrap_or_else(|| "text-embedding-3-small".into()),
                key: opts
                    .key
                    .clone()
                    .or_else(|| env("EMBED_API_KEY"))
                    .or_else(|| env("OPENAI_API_KEY"))
                    .unwrap_or_default(),
                batch: opts
                    .batch
                    .or_else(|| env("EMBED_BATCH").and_then(|s| s.parse().ok()))
                    .unwrap_or(64),
                dim: 0,
                rerank_model: String::new(),
            },
            _ => Provider {
                kind: ProviderKind::Hash,
                url: String::new(),
                model: "hash".into(),
                key: String::new(),
                batch: opts
                    .batch
                    .or_else(|| env("EMBED_BATCH").and_then(|s| s.parse().ok()))
                    .unwrap_or(4096),
                dim: opts
                    .dim
                    .or_else(|| env("EMBED_DIM").and_then(|s| s.parse().ok()))
                    .unwrap_or(256),
                rerank_model: String::new(),
            },
        }
    }

    /// Stable identity of this provider's embedding space. Embeddings from a
    /// different provider/model are invalid and must be recomputed.
    pub fn id(&self) -> String {
        match self.kind {
            ProviderKind::Hash => format!("hash:{}", self.dim),
            ProviderKind::Http => format!("http:{}@{}", self.model, self.url),
            ProviderKind::Cohere => format!("cohere:{}@{}", self.model, self.url),
        }
    }

    /// Whether this provider can rerank candidate documents. Only Cohere
    /// exposes a reranking endpoint; other providers keep the hybrid RRF
    /// order and are not maintained further.
    pub fn supports_rerank(&self) -> bool {
        self.kind == ProviderKind::Cohere && !self.rerank_model.is_empty()
    }

    /// Rerank `docs` against `query`, returning `(doc_index, relevance)`
    /// pairs sorted best-first. Returns None when the provider has no
    /// reranker or the call fails — reranking is a quality upgrade, never
    /// a hard dependency (the caller keeps the RRF order on None).
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Option<Vec<(usize, f32)>> {
        if !self.supports_rerank() || docs.is_empty() {
            return None;
        }
        match self.cohere_rerank(query, docs) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("rerank failed ({e}); keeping hybrid order");
                None
            }
        }
    }

    /// Cohere `/rerank` (v2): query + documents -> per-document relevance
    /// scores. One request reranks the whole candidate pool.
    fn cohere_rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, String> {
        #[derive(Deserialize)]
        struct RerankResp {
            results: Vec<RerankResult>,
        }
        #[derive(Deserialize)]
        struct RerankResult {
            index: usize,
            relevance_score: f32,
        }
        let mut req = ureq::post(&format!("{}/rerank", self.url));
        if !self.key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.key));
        }
        let resp = req
            .timeout(Duration::from_secs(60))
            .send_json(json!({
                "model": self.rerank_model,
                "query": query,
                "documents": docs,
                "top_n": docs.len(),
            }))
            .map_err(|e| format!("cohere rerank request failed: {e}"))?;
        let mut scored: Vec<(usize, f32)> = resp
            .into_json::<RerankResp>()
            .map_err(|e| format!("bad cohere rerank response: {e}"))?
            .results
            .into_iter()
            .filter(|r| r.index < docs.len())
            .map(|r| (r.index, r.relevance_score))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }

    /// Embed a batch of documents (code blocks).
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        match self.kind {
            ProviderKind::Hash => Ok(texts.iter().map(|t| hash_embed(t, self.dim)).collect()),
            ProviderKind::Http => self.http_embed(texts),
            ProviderKind::Cohere => self.cohere_embed(texts, "search_document"),
        }
    }

    /// Embed a single search query. Providers with an asymmetric
    /// document/query distinction (Cohere) use it; others reuse `embed`.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, String> {
        match self.kind {
            ProviderKind::Cohere => Ok(self.cohere_embed(&[text], "search_query")?.remove(0)),
            _ => Ok(self.embed(&[text])?.remove(0)),
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

    /// Cohere `/embed` (v2): texts + input_type, float embeddings.
    /// `input_type` is "search_document" for blocks and "search_query" for
    /// queries — Cohere models are trained asymmetrically and this
    /// materially improves retrieval quality.
    fn cohere_embed(&self, texts: &[&str], input_type: &str) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        #[derive(Deserialize)]
        struct CohereResp {
            embeddings: CohereEmbeddings,
        }
        #[derive(Deserialize)]
        struct CohereEmbeddings {
            float: Vec<Vec<f32>>,
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch.max(1)) {
            let mut req = ureq::post(&format!("{}/embed", self.url));
            if !self.key.is_empty() {
                req = req.set("Authorization", &format!("Bearer {}", self.key));
            }
            let resp = req
                .timeout(Duration::from_secs(300))
                .send_json(json!({
                    "model": self.model,
                    "texts": chunk,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                }))
                .map_err(|e| format!("cohere embedding request failed: {e}"))?;
            let parsed: CohereResp = resp
                .into_json()
                .map_err(|e| format!("bad cohere embedding response: {e}"))?;
            let mut embs = parsed.embeddings.float;
            if embs.len() != chunk.len() {
                return Err(format!(
                    "cohere embedding count mismatch: sent {}, got {}",
                    chunk.len(),
                    embs.len()
                ));
            }
            for v in embs.iter_mut() {
                normalize(v);
            }
            out.extend(embs);
        }
        Ok(out)
    }
}

// ---------- embedding store ----------

#[derive(Serialize, Deserialize, Default, Clone)]
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
    reset_for_provider(emb, prov);
    let missing: HashMap<u64, String> = index
        .blocks
        .par_iter()
        .filter(|b| b.file != TOMBSTONE_FILE)
        .filter(|b| !emb.map.contains_key(&b.hash))
        .map(|b| (b.hash, b.text.clone()))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let total = missing.len();
    if progress {
        eprintln!("  embedding {total} block(s) with {} ...", prov.id());
    }
    embed_texts(emb, prov, missing, progress)
}

/// Reset the store if the provider changed, so vectors from different
/// models are never mixed.
pub fn reset_for_provider(emb: &mut Embeddings, prov: &Provider) {
    if emb.provider_id != prov.id() {
        *emb = Embeddings {
            provider_id: prov.id(),
            dim: match prov.kind {
                ProviderKind::Hash => prov.dim,
                ProviderKind::Http | ProviderKind::Cohere => 0,
            },
            map: HashMap::new(),
        };
    }
}

/// Embed a set of (content-hash, text) pairs into the store. This is the
/// slow HTTP part, deliberately separated so callers can run it without
/// holding the index lock.
pub fn embed_texts(
    emb: &mut Embeddings,
    prov: &Provider,
    missing: HashMap<u64, String>,
    progress: bool,
) -> Result<(), String> {
    let mut items: Vec<(u64, String)> = missing.into_iter().collect();
    items.sort_unstable_by_key(|(h, _)| *h); // deterministic order
    let total = items.len();
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

/// Collect the (hash, text) pairs that still lack vectors, reading the
/// index. Cheap: a parallel scan; callers should run this under a short
/// lock, then release the lock before calling `embed_texts`.
pub fn collect_missing(index: &Index, emb: &Embeddings) -> HashMap<u64, String> {
    index
        .blocks
        .par_iter()
        .filter(|b| b.file != TOMBSTONE_FILE)
        .filter(|b| !emb.map.contains_key(&b.hash))
        .map(|b| (b.hash, b.text.clone()))
        .collect()
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
        .filter_map(|(i, b)| emb.map.get(&b.hash).map(|v| (dot(q, v), i as u32)))
        .collect();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    scores.truncate(k);
    scores
}

/// Result of the full `ask` path.
pub struct AskOutcome {
    /// Fused (and, when the provider supports it, reranked) hits with
    /// their scores — fused RRF weight or Cohere relevance score.
    pub hits: Vec<(f32, search::Hit)>,
    /// Whether new block embeddings were computed (cache should be saved).
    pub changed: bool,
    /// Whether the provider's reranker reordered the fused candidates.
    pub reranked: bool,
}

/// Hard ceiling on the candidate pool sent to the reranker: one request,
/// bounded cost and latency.
const MAX_RERANK_POOL: usize = 100;

/// Candidate pool size fed to the reranker: 4× the requested limit, capped
/// at `MAX_RERANK_POOL`. The pool is always >= limit, so a large `limit`
/// still gets reranked (it just has less headroom to reorder — it never
/// silently disables reranking). Providers without a reranker get 1×.
pub fn rerank_pool(prov: &Provider, limit: usize) -> usize {
    if !prov.supports_rerank() {
        return limit;
    }
    limit.max((limit * 4).min(MAX_RERANK_POOL))
}

/// Cap per candidate text before sending it to the reranker: beyond this
/// the API truncates anyway; smaller requests are cheaper and faster.
const RERANK_DOC_CHARS: usize = 4000;

/// Ask the provider to rank `texts` (candidate block bodies) against the
/// query, returning `(index, relevance)` best-first. Texts are truncated to
/// `RERANK_DOC_CHARS` here so callers don't have to. `None` = no reranker,
/// nothing to rank, or the call failed — callers keep their existing order.
///
/// This performs the HTTP round-trip and touches no index state, so servers
/// can call it with every lock released.
pub fn rerank_order(prov: &Provider, query: &str, texts: &[&str]) -> Option<Vec<(usize, f32)>> {
    // <=1 candidate: nothing to reorder, don't spend a request on it.
    if !prov.supports_rerank() || texts.len() <= 1 {
        return None;
    }
    let docs: Vec<String> = texts
        .iter()
        .map(|t| t.chars().take(RERANK_DOC_CHARS).collect())
        .collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    prov.rerank(query, &refs).filter(|o| !o.is_empty())
}

/// Apply a `rerank_order` result to any vector of scored items, replacing
/// each item's score with its relevance and truncating to `limit`.
/// Generic so servers can reorder already-enriched records instead of
/// re-resolving ids against a possibly-changed index afterwards.
pub fn apply_rerank_order<T>(
    items: Vec<(f32, T)>,
    order: Vec<(usize, f32)>,
    limit: usize,
) -> Vec<(f32, T)> {
    // Move items out by index (T is not necessarily Clone).
    let mut slots: Vec<Option<(f32, T)>> = items.into_iter().map(Some).collect();
    order
        .into_iter()
        .take(limit)
        .filter_map(|(i, score)| {
            slots
                .get_mut(i)
                .and_then(Option::take)
                .map(|(_, v)| (score, v))
        })
        .collect()
}

/// Rerank fused hits against the query, truncating to `limit`. Falls back
/// to the original RRF order when the provider has no reranker or the call
/// fails — reranking must never fail a query. Returns `(hits, reranked)`.
pub fn rerank_fused(
    prov: &Provider,
    mut fused: Vec<(f32, search::Hit)>,
    query: &str,
    limit: usize,
) -> (Vec<(f32, search::Hit)>, bool) {
    let texts: Vec<&str> = fused.iter().map(|(_, h)| h.text.as_str()).collect();
    match rerank_order(prov, query, &texts) {
        Some(order) => (apply_rerank_order(fused, order, limit), true),
        None => {
            fused.truncate(limit);
            (fused, false)
        }
    }
}

/// Hybrid search: fuse lexical (trigram) and semantic rankings with
/// reciprocal rank fusion, then rerank the fused candidates with the
/// provider's reranker when it has one (Cohere).
///
/// NOTE: this embeds every missing block inline before answering — fine for
/// the CLI where the user expects to wait once, but too slow for latency-
/// sensitive servers. Those should use `ask_fast` and pre-embed in the
/// background via `ensure_all`.
pub fn ask(
    idx: &mut Index,
    prov: &Provider,
    query: &str,
    limit: usize,
) -> Result<AskOutcome, String> {
    let before = idx.embeddings.map.len();
    let pool = rerank_pool(prov, limit);
    let mut emb = std::mem::take(&mut idx.embeddings);
    let result = (|| -> Result<Vec<(f32, search::Hit)>, String> {
        ensure(&mut emb, prov, idx, true)?;
        let qv = prov.embed_query(query)?;
        let sem = top_k(idx, &emb, &qv, 200);
        let lex = search::search(idx, query, 200);
        Ok(fuse(idx, &lex, &sem, query, pool))
    })();
    idx.embeddings = emb;
    let changed = idx.embeddings.map.len() != before;
    result.map(|fused| {
        let (hits, reranked) = rerank_fused(prov, fused, query, limit);
        AskOutcome {
            hits,
            changed,
            reranked,
        }
    })
}

/// Low-latency hybrid search: NEVER embeds the corpus inline. Uses whatever
/// vectors are already cached (blocks without vectors simply contribute only
/// to the lexical ranking), so it stays fast even while a background
/// embedding pass is still warming up. Only the query itself is embedded
/// (one HTTP round-trip).
///
/// Returns `(hits, coverage)` where coverage is the fraction of live blocks
/// that already have vectors (1.0 = fully warm).
pub fn ask_fast(
    idx: &Index,
    emb: &Embeddings,
    prov: &Provider,
    query: &str,
    limit: usize,
) -> Result<(Vec<(f32, search::Hit)>, f32), String> {
    let qv = prov.embed_query(query)?;
    Ok(ask_fast_with_qv(idx, emb, &qv, query, limit))
}

/// `ask_fast` with a pre-embedded query vector. Splitting the HTTP call
/// (embed the query) from the scoring lets servers run the network part
/// WITHOUT holding any index/embedding lock.
pub fn ask_fast_with_qv(
    idx: &Index,
    emb: &Embeddings,
    qv: &[f32],
    query: &str,
    limit: usize,
) -> (Vec<(f32, search::Hit)>, f32) {
    let coverage = coverage_of(idx, emb);
    let sem = top_k(idx, emb, qv, 200);
    let lex = search::search(idx, query, 200);
    (fuse(idx, &lex, &sem, query, limit), coverage)
}

/// Fraction of live blocks that currently have vectors (1.0 = fully warm).
pub fn coverage_of(idx: &Index, emb: &Embeddings) -> f32 {
    let live = idx
        .blocks
        .iter()
        .filter(|b| b.file != TOMBSTONE_FILE)
        .count();
    if live == 0 {
        return 1.0;
    }
    let covered = idx
        .blocks
        .iter()
        .filter(|b| b.file != TOMBSTONE_FILE)
        .filter(|b| emb.map.contains_key(&b.hash))
        .count();
    covered as f32 / live as f32
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
