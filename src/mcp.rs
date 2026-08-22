//! MCP (Model Context Protocol) server over stdio, so AI coding assistants
//! (Claude Code, Cursor, ...) can query this codebase's index: lexical
//! search, hybrid semantic search, symbol graphs, call flows, and clues.
//!
//! Design goal: LOW LATENCY. The index is held in memory behind a shared
//! mutex, a filesystem watcher keeps it fresh in the background, and a
//! background embedder keeps semantic vectors warm. `endex_ask` never
//! embeds the corpus inline — it embeds only the query (one HTTP round
//! trip) and uses whatever vectors are already cached, degrading to
//! lexical-only results while the embedder is still warming up.
//!
//! Protocol: JSON-RPC 2.0, one message per line (`Content-Length` headers
//! are also accepted and auto-detected). Only `initialize`, `ping`,
//! `tools/list` and `tools/call` are implemented; notifications (no `id`)
//! are processed without a response.

use crate::embed::{self, Embeddings, Provider};
use crate::graph::Graph;
use crate::index::Index;
use crate::{search, store, watch};
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

const PROTOCOL_VERSION: &str = "2025-06-18";

/// Locks that survive a poisoned mutex: after `panic = "abort"` was dropped
/// from the release profile, a panicking tool call must not take down every
/// subsequent request with it.
fn mlock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn rlock<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

fn wlock<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}

/// Global state shared by all tool calls and the background tasks.
struct Shared {
    idx: Index,
    root: PathBuf,
    dirty: bool,
    stats: ServerStats,
}

struct ServerStats {
    started_unix: u64,
    indexed_files: usize,
    last_index_unix: Option<u64>,
    last_embed_unix: Option<u64>,
    last_embed_blocks: usize,
    embed_runs: usize,
    last_save_unix: Option<u64>,
    index_runs: usize,
}

static SHARED: OnceLock<Arc<Mutex<Shared>>> = OnceLock::new();
/// Live semantic vectors, owned by the background embedder and swapped in
/// atomically. Queries read through an RwLock and are NEVER blocked by an
/// in-progress embedding pass (the embedder works on a cloned snapshot and
/// only takes the write lock for the final swap).
static EMB: OnceLock<Arc<RwLock<Embeddings>>> = OnceLock::new();

fn shared() -> Arc<Mutex<Shared>> {
    SHARED.get().expect("MCP server not initialized").clone()
}

fn emb() -> Arc<RwLock<Embeddings>> {
    EMB.get().expect("MCP server not initialized").clone()
}

/// Counter bumped by the watcher whenever files change; the embedder thread
/// waits on it to re-embed only when needed.
static EPOCH: AtomicUsize = AtomicUsize::new(0);

fn change_epoch() -> &'static AtomicUsize {
    &EPOCH
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// All exposed tools, with JSON Schemas for their arguments. Tools always
/// operate on the directory the server was started with (a deliberately
/// single-root design — one server per repo).
fn tool_defs() -> Value {
    json!([
        {
            "name": "endex_index",
            "description": "Build or refresh the endex code index. IMPORTANT: call this FIRST when starting work in a repo where endex has not been run yet — it enables all other endex_* tools (subsequent calls are incremental, only changed files are re-parsed, and the other tools auto-load the index anyway). Skip it if endex_stats already returns stats for this directory.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "endex_search",
            "description": "Fast substring search over the whole codebase, ranked by relevance. PREFER THIS over Grep/Glob when looking for where an identifier, function name, error message, or literal string is used: results are pre-ranked, deduplicated per block, and each hit INCLUDES the full code block text (typically the whole function), so you usually do NOT need a follow-up Read of the file. Sub-millisecond even on large repos.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Case-insensitive substring to find — e.g. a function name, type name, log message, or error string." },
                    "limit": { "type": "integer", "description": "Max results (default 20, max 100)." }
                },
                "required": ["query"]
            }
        },
        {
            "name": "endex_ask",
            "description": "Semantic search over the codebase in NATURAL LANGUAGE — use this whenever you do not know the exact identifier: 'how do we handle retries', 'where is rate limiting enforced', 'authentication middleware'. STRONGLY PREFER this as the FIRST step when exploring an unfamiliar codebase or concept, instead of guessing grep patterns. Each hit includes the full code block text, so follow-up file reads are rarely needed. Results are reranked for relevance when the provider supports it; vectors are kept warm by a background embedder (if coverage < 1.0 the index is still warming up and results are lexical-leaning — retry shortly).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language question or description of the code you are looking for." },
                    "limit": { "type": "integer" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "endex_graph",
            "description": "Knowledge-graph neighborhood of a symbol: what it calls (callees), who calls it (callers), and which files import it, with file:line for every entry. USE THIS before editing a shared function to see all its call sites and dependents — it answers 'what breaks if I change X?' far better than grep, because edges are real call/import relationships, not text matches. Also great after endex_search/endex_ask to orient around a discovered symbol.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": { "type": "string", "description": "Symbol name (function, method, class, struct, ...)." }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "endex_flow",
            "description": "Trace EXECUTION FLOWS through the codebase: finds call-graph paths between two symbols (e.g. from 'main' to 'save', from an HTTP handler to the DB write). This is the PRIMARY tool for questions like 'how does X reach Y?', 'what is the code path for feature Z?', 'trace the request lifecycle'. Returns up to 5 shortest paths; every hop has file:line AND the full source block inline (disable with include_blocks=false), so a single call often answers a flow question completely — no manual file hopping needed. Tip: use endex_search or endex_ask first to discover the endpoint symbol names if unsure.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Source symbol name — an entry point like main, a handler, an exported API." },
                    "to": { "type": "string", "description": "Target symbol name — the downstream function you want to trace into." },
                    "include_blocks": { "type": "boolean", "description": "Also include the source text of the block each hop is defined in (default true — keep it on unless the response is too large)." },
                    "max_depth": { "type": "integer", "description": "Max path length (default 8)." }
                },
                "required": ["from", "to"]
            }
        },
        {
            "name": "endex_clues",
            "description": "Reconnaissance by concept: blocks mentioning a term, each annotated with the symbols DEFINED in that block plus their callers/callees. Use when you have a topic word ('cache', 'auth', 'retry') and want both the matching code AND the key symbols involved — the returned symbol names are ideal follow-up inputs for endex_graph and endex_flow.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "term": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["term"]
            }
        },
        {
            "name": "endex_stats",
            "description": "Server statistics: index size (files/blocks/symbols/edges), embedding provider + coverage, cache version/path/bytes/age, server uptime, last index/embed/save timestamps. Call this to verify setup before relying on endex_ask.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ])
}

// ---------- JSON-RPC plumbing ----------

fn rpc_ok(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_err(id: &Value, code: i64, msg: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg } })
}

fn tool_ok(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Build the response meta: wraps the payload with timing + cache info.
/// Callers construct the payload first, then pass it with the measured
/// duration. Includes cache version/format, server uptime and — when
/// available — the embedding provider identity.
fn meta(sh: &Shared, payload: Value, duration_ms: f64) -> Value {
    let info = store::cache_info(&sh.root);
    let mut m = json!({
        "meta": {
            "data": payload,
            // Numeric, rounded to 2 decimals: compact and machine-readable.
            "duration_ms": (duration_ms * 100.0).round() / 100.0,
            "cache_version": store::CACHE_VERSION,
        }
    });
    if let Some(i) = info {
        m["meta"]["cache_bytes"] = json!(i.bytes);
        if let Some(age) = i.age_seconds {
            m["meta"]["cache_age_seconds"] = json!(age);
        }
    }
    m
}

/// Full server stats payload for the endex_stats tool.
fn stats_payload(sh: &Shared, emb: &Embeddings, provider: &Provider) -> Value {
    let idx = &sh.idx;
    let s = &sh.stats;
    let now = unix_now();
    let info = store::cache_info(&sh.root);
    json!({
        "index": {
            "dir": sh.root,
            "files": idx.file_count(),
            "blocks": idx.block_count(),
            "symbols": idx.graph.symbols.len(),
            "call_edges": idx.graph.call_edge_count(),
            "import_edges": idx.graph.file_imports.len(),
            "corpus_fingerprint": format!("{:#x}", idx.corpus_fingerprint()),
        },
        "embeddings": {
            "provider": provider.id(),
            "rerank_model": if provider.supports_rerank() {
                json!(provider.rerank_model)
            } else {
                Value::Null
            },
            "vectors": emb.map.len(),
            "dim": emb.dim,
            "coverage": coverage(idx, emb),
        },
        "cache": {
            "version": store::CACHE_VERSION,
            "path": info.as_ref().map(|i| i.path.clone()),
            "bytes": info.as_ref().map(|i| i.bytes),
            "age_seconds": info.as_ref().and_then(|i| i.age_seconds),
        },
        "server": {
            "started_unix": s.started_unix,
            "uptime_seconds": now.saturating_sub(s.started_unix),
            "last_index_unix": s.last_index_unix,
            "indexed_since_start": s.indexed_files,
            "index_runs": s.index_runs,
            "last_embed_unix": s.last_embed_unix,
            "last_embed_blocks": s.last_embed_blocks,
            "embed_runs": s.embed_runs,
            "last_save_unix": s.last_save_unix,
            "revision": env!("CARGO_PKG_VERSION"),
        },
    })
}

fn tool_err(msg: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": msg }],
        "isError": true
    })
}

/// Cap on a single framed message body — guards against absurd
/// Content-Length values allocating gigabytes.
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Extract the JSON payload of one inbound message. Supports
/// newline-delimited JSON (single objects AND JSON-RPC batch arrays) and
/// `Content-Length`-framed messages.
///
/// Robustness contract: a malformed line yields `Value::Null` (skipped by
/// the caller) and NEVER corrupts the stream for subsequent messages.
pub fn read_message(stdin: &mut dyn BufRead, buf: &mut Vec<u8>) -> io::Result<Option<Value>> {
    buf.clear();
    if stdin.read_until(b'\n', buf)? == 0 {
        return Ok(None); // EOF
    }
    let line = String::from_utf8_lossy(buf);
    let line = line.trim();
    if line.is_empty() {
        return Ok(Some(Value::Null));
    }
    if line.starts_with('{') || line.starts_with('[') {
        // Newline-delimited JSON (object or batch array).
        return Ok(Some(serde_json::from_str(line).unwrap_or(Value::Null)));
    }
    // Header framing — only entered when the first header declares a length,
    // so random garbage lines are skipped instead of eating the stream.
    let lower = line.to_lowercase();
    let Some(rest) = lower.strip_prefix("content-length:") else {
        return Ok(Some(Value::Null));
    };
    let Some(n) = rest.trim().parse::<usize>().ok() else {
        return Ok(Some(Value::Null));
    };
    if n > MAX_MESSAGE_BYTES {
        return Ok(Some(Value::Null));
    }
    // Consume remaining header lines up to the blank separator.
    loop {
        buf.clear();
        if stdin.read_until(b'\n', buf)? == 0 {
            return Ok(None); // EOF mid-headers
        }
        if String::from_utf8_lossy(buf).trim().is_empty() {
            break;
        }
    }
    let mut body = vec![0u8; n];
    if stdin.read_exact(&mut body).is_err() {
        return Ok(None); // EOF mid-body
    }
    Ok(Some(serde_json::from_slice(&body).unwrap_or(Value::Null)))
}

fn write_message(out: &mut dyn io::Write, msg: &Value) {
    let s = msg.to_string();
    // Newline-delimited JSON: simplest framing accepted by all stdio clients.
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}

// ---------- tool implementations ----------

fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn arg_usize(args: &Value, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn arg_bool(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Round a score to 4 decimals as f64: compact, LLM-friendly output
/// without f32 rendering noise in JSON.
fn round_score(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

/// Fully resolve one `ask` hit into its JSON record: file path plus the
/// symbols defined in the block (ideal follow-up inputs for endex_graph /
/// endex_flow). MUST be called while holding the lock that produced the
/// hit — `file_id` / `block_id` are recycled on reindex.
fn ask_hit_json(idx: &Index, h: &search::Hit, score: f32) -> Value {
    let g = &idx.graph;
    let syms: Vec<Value> = g
        .by_block
        .get(&h.block_id)
        .map(|ids| {
            ids.iter()
                .take(8)
                .filter_map(|&id| g.symbols.get(id as usize))
                .map(|s| json!({ "name": s.name, "kind": s.kind.label() }))
                .collect()
        })
        .unwrap_or_default();
    json!({
        "file": idx.path_of(h.file_id),
        "line": h.line,
        "score": round_score(score),
        "symbols": syms,
        "text": h.text,
    })
}

fn resolve_symbol(idx: &Index, g: &Graph, id: u32) -> Value {
    let s = &g.symbols[id as usize];
    json!({
        "name": s.name,
        "kind": s.kind.label(),
        "file": idx.path_of(s.file),
        "line": s.line
    })
}

fn symbol_summary(idx: &Index, g: &Graph, id: u32) -> Value {
    let s = &g.symbols[id as usize];
    let callees: Vec<Value> = g
        .callees(id)
        .iter()
        .take(50)
        .map(|&c| resolve_symbol(idx, g, c))
        .collect();
    let callers: Vec<Value> = g
        .callers(id)
        .iter()
        .take(50)
        .map(|&c| resolve_symbol(idx, g, c))
        .collect();
    let importers: Vec<&str> = g
        .file_imports
        .iter()
        .filter(|(_, to)| *to == s.file)
        .map(|(from, _)| idx.path_of(*from))
        .take(20)
        .collect();
    json!({
        "name": s.name,
        "kind": s.kind.label(),
        "file": idx.path_of(s.file),
        "line": s.line,
        "calls": callees,
        "called_by": callers,
        "imported_by": importers
    })
}

/// Fraction of live blocks that currently have semantic vectors.
fn coverage(idx: &Index, emb: &Embeddings) -> f32 {
    embed::coverage_of(idx, emb)
}

fn exec_tool(provider: &Provider, name: &str, args: &Value) -> Value {
    let t0 = std::time::Instant::now();
    match name {
        "endex_index" => {
            let sh_arc = shared();
            let mut sh = mlock(&sh_arc);
            // A watcher keeps the index fresh; a manual call forces a refresh.
            let root = sh.root.clone();
            let changed = sh.idx.refresh(&root);
            if changed > 0 {
                sh.stats.index_runs += 1;
                sh.stats.last_index_unix = Some(unix_now());
                change_epoch().fetch_add(1, Ordering::SeqCst);
            }
            let idx = &sh.idx;
            let emb_arc = emb();
            let emb = rlock(&emb_arc);
            let payload = json!({
                "dir": sh.root,
                "refreshed_files": changed,
                "files": idx.file_count(),
                "blocks": idx.block_count(),
                "symbols": idx.graph.symbols.len(),
                "call_edges": idx.graph.call_edge_count(),
                "import_edges": idx.graph.file_imports.len(),
                // Named `coverage` everywhere (endex_ask, endex_stats).
                "coverage": coverage(idx, &emb),
                "message": "index ready (filesystem watcher active; changes are picked up automatically)"
            });
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_stats" => {
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let emb_store = emb();
            let emb = rlock(&emb_store);
            let payload = stats_payload(&sh, &emb, provider);
            // Wrapped in the same meta envelope as every other tool.
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_search" => {
            let query = match arg_str(args, "query") {
                Some(q) if !q.is_empty() => q,
                _ => return tool_err("missing required argument: query"),
            };
            let limit = arg_usize(args, "limit", 20).min(100);
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let idx = &sh.idx;
            let hits = search::search(idx, query, limit);
            let out: Vec<Value> = hits
                .iter()
                .map(|h| {
                    json!({
                        "file": idx.path_of(h.file_id),
                        "line": h.line,
                        "occurrences": h.occurrences,
                        "text": h.text,
                    })
                })
                .collect();
            let payload = json!({ "query": query, "count": hits.len(), "hits": out });
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_ask" => {
            let query = match arg_str(args, "query") {
                Some(q) if !q.is_empty() => q,
                _ => return tool_err("missing required argument: query"),
            };
            let limit = arg_usize(args, "limit", 20).min(100);
            // 1. Embed the query BEFORE taking any lock: this HTTP
            //    round-trip must never block the watcher, the embedder,
            //    or other tools.
            let qv = provider.embed_query(query);
            let pool = embed::rerank_pool(provider, limit);
            // 2. Fuse lexical + semantic rankings into a candidate pool
            //    sized for the reranker, and fully resolve every hit
            //    (path + symbols) WHILE STILL HOLDING THE LOCK. File and
            //    block ids are recycled by the watcher on reindex, so they
            //    must never outlive the guard that produced them.
            let (mut cands, cov, warn) = {
                let sh_arc = shared();
                let sh = mlock(&sh_arc);
                let idx = &sh.idx;
                let emb_arc = emb();
                let emb = rlock(&emb_arc);
                let (hits, cov, warn) = match &qv {
                    Ok(qv) => {
                        let (hits, cov) = embed::ask_fast_with_qv(idx, &emb, qv, query, pool);
                        (hits, cov, None)
                    }
                    Err(e) => (
                        // Provider unreachable: degrade to lexical, say so.
                        search::search(idx, query, limit)
                            .into_iter()
                            .map(|h| (0.0f32, h))
                            .collect(),
                        0.0,
                        Some(format!(
                            "semantic provider unavailable ({e}); results are lexical only"
                        )),
                    ),
                };
                let cands: Vec<(f32, Value)> = hits
                    .into_iter()
                    .map(|(score, h)| (score, ask_hit_json(idx, &h, score)))
                    .collect();
                (cands, cov, warn)
            };
            // 3. Rerank with NO locks held (this is an HTTP call). Skipped
            //    when the embedding call already failed; internally a no-op
            //    for providers without a reranker. Only scores and order
            //    change here — the records are already resolved.
            let mut reranked = false;
            if warn.is_none() {
                let texts: Vec<&str> = cands
                    .iter()
                    .map(|(_, v)| v["text"].as_str().unwrap_or(""))
                    .collect();
                if let Some(order) = embed::rerank_order(provider, query, &texts) {
                    cands = embed::apply_rerank_order(cands, order, limit);
                    reranked = true;
                }
            }
            cands.truncate(limit);
            let out: Vec<Value> = cands
                .into_iter()
                .map(|(score, mut v)| {
                    // Reranking replaced the RRF weight with a relevance score.
                    v["score"] = json!(round_score(score));
                    v
                })
                .collect();
            let mut payload = json!({
                "query": query,
                "provider": provider.id(),
                "coverage": (cov * 100.0).round() / 100.0,
                "ranked_by": if warn.is_some() {
                    "lexical"
                } else if reranked {
                    "rerank"
                } else {
                    "rrf"
                },
                "count": out.len(),
                "hits": out,
            });
            // Conditional fields only: keep the default output compact.
            if let Some(w) = warn {
                payload["warning"] = json!(w);
            }
            if cov < 0.95 && payload.get("warning").is_none() {
                payload["warming_up"] = json!(true);
            }
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_graph" => {
            let sym = match arg_str(args, "symbol") {
                Some(s) if !s.is_empty() => s,
                _ => return tool_err("missing required argument: symbol"),
            };
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let idx = &sh.idx;
            let g = &idx.graph;
            let ids = g.find_all(sym);
            if ids.is_empty() {
                let sugg = g.suggest(sym);
                return tool_err(&format!(
                    "no symbol named '{sym}'.{}",
                    if sugg.is_empty() {
                        String::new()
                    } else {
                        format!(" did you mean: {} ?", sugg.join(", "))
                    }
                ));
            }
            let out: Vec<Value> = ids.iter().map(|&id| symbol_summary(idx, g, id)).collect();
            let payload = json!({ "symbol": sym, "matches": out });
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_clues" => {
            let term = match arg_str(args, "term") {
                Some(t) if !t.is_empty() => t,
                _ => return tool_err("missing required argument: term"),
            };
            let limit = arg_usize(args, "limit", 15).min(100);
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let idx = &sh.idx;
            let g = &idx.graph;
            let hits = search::search(idx, term, limit);
            let out: Vec<Value> = hits
                .iter()
                .map(|h| {
                    let syms: Vec<Value> = g
                        .by_block
                        .get(&h.block_id)
                        .map(|ids| {
                            ids.iter()
                                .map(|&id| {
                                    let s = &g.symbols[id as usize];
                                    let callers: Vec<String> = g
                                        .callers(id)
                                        .iter()
                                        .take(6)
                                        .map(|&c| g.symbols[c as usize].name.clone())
                                        .collect();
                                    let callees: Vec<String> = g
                                        .callees(id)
                                        .iter()
                                        .take(6)
                                        .map(|&c| g.symbols[c as usize].name.clone())
                                        .collect();
                                    json!({
                                        "name": s.name,
                                        "kind": s.kind.label(),
                                        "called_by": callers,
                                        "calls": callees,
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    json!({
                        "file": idx.path_of(h.file_id),
                        "line": h.line,
                        "occurrences": h.occurrences,
                        "symbols_defined": syms,
                        "text": h.text,
                    })
                })
                .collect();
            let payload = json!({ "term": term, "count": hits.len(), "hits": out });
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        "endex_flow" => {
            let from = match arg_str(args, "from") {
                Some(s) if !s.is_empty() => s,
                _ => return tool_err("missing required argument: from"),
            };
            let to = match arg_str(args, "to") {
                Some(s) if !s.is_empty() => s,
                _ => return tool_err("missing required argument: to"),
            };
            let max_depth = arg_usize(args, "max_depth", 8).clamp(2, 32);
            let include_blocks = arg_bool(args, "include_blocks", true);
            let sh_arc = shared();
            let sh = mlock(&sh_arc);
            let idx = &sh.idx;
            let g = &idx.graph;
            let sources = g.find_all(from);
            let targets: std::collections::HashSet<u32> = g.find_all(to).into_iter().collect();
            if sources.is_empty() {
                let sugg = g.suggest(from);
                return tool_err(&format!(
                    "no symbol named '{from}'.{}",
                    if sugg.is_empty() {
                        String::new()
                    } else {
                        format!(" did you mean: {} ?", sugg.join(", "))
                    }
                ));
            }
            if targets.is_empty() {
                let sugg = g.suggest(to);
                return tool_err(&format!(
                    "no symbol named '{to}'.{}",
                    if sugg.is_empty() {
                        String::new()
                    } else {
                        format!(" did you mean: {} ?", sugg.join(", "))
                    }
                ));
            }
            let paths = g.find_paths(&sources, &targets, max_depth, 5);
            if paths.is_empty() {
                let payload = json!({
                    "from": from,
                    "to": to,
                    "paths": [],
                    "message": format!("no call path found (max depth {max_depth})")
                });
                let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
                return tool_ok(m.to_string());
            }
            let out: Vec<Value> = paths
                .iter()
                .map(|p| {
                    let hops: Vec<Value> = p
                        .iter()
                        .map(|&sid| {
                            let s = &g.symbols[sid as usize];
                            let mut v = json!({
                                "name": s.name,
                                "kind": s.kind.label(),
                                "file": idx.path_of(s.file),
                                "line": s.line,
                            });
                            if include_blocks {
                                if let Some(blk) = idx.blocks.get(s.block as usize) {
                                    v["block_text"] = json!(blk.text);
                                }
                            }
                            v
                        })
                        .collect();
                    json!({ "hops": hops })
                })
                .collect();
            let payload = json!({
                "from": from,
                "to": to,
                "path_count": paths.len(),
                "paths": out
            });
            let m = meta(&sh, payload, t0.elapsed().as_secs_f64() * 1000.0);
            tool_ok(m.to_string())
        }

        _ => {
            let _ = t0; // silence unused for dispatch fallthrough
            tool_err(&format!("unknown tool: {name}"))
        }
    }
}

// ---------- background tasks ----------

/// Filesystem watcher: reindexes changed files, rebuilds the graph, saves
/// the cache, and pokes the embedder via the change epoch. Changed paths
/// are filtered through the same ignore rules as full walks — otherwise
/// gitignored secrets (`.env`) and build output (`target/`, `node_modules`)
/// would leak into the index (and from there to the embedding provider).
fn spawn_watcher() {
    let sh = shared();
    let root = mlock(&sh).root.clone();
    std::thread::spawn(move || {
        let rx = match watch::watch(&root) {
            Ok(rx) => rx,
            Err(e) => {
                eprintln!("endex MCP: watcher unavailable: {e} (index will not auto-refresh)");
                return;
            }
        };
        let mut ignores = watch::Ignores::new(&root);
        for batch in rx.iter() {
            // A panicking batch must not kill the watcher thread.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle_watch_batch(&batch, &mut ignores)
            }));
            if r.is_err() {
                eprintln!("endex MCP: watcher panicked on a change batch; continuing");
            }
        }
    });
}

fn handle_watch_batch(batch: &[PathBuf], ignores: &mut watch::Ignores) {
    let sh_arc = shared();
    let mut sh = mlock(&sh_arc);
    let mut n = 0usize;
    for p in batch {
        if ignores.is_ignored(p, p.is_dir()) {
            continue;
        }
        if !p.is_file() {
            sh.idx.remove_file(p);
            continue;
        }
        if sh.idx.index_file(p) {
            n += 1;
        }
    }
    if n > 0 {
        sh.idx.finalize(); // rebuild graph + GC embeddings in memory
        let now = unix_now();
        sh.stats.indexed_files += n;
        sh.stats.index_runs += 1;
        sh.stats.last_index_unix = Some(now);
        sh.dirty = true;
        let root = sh.root.clone();
        if store::save(&sh.idx, &root).is_ok() {
            // Saved: only clear dirty on SUCCESS so the shutdown flush still
            // persists after a failed save.
            sh.stats.last_save_unix = Some(now);
            sh.dirty = false;
        }
        change_epoch().fetch_add(1, Ordering::SeqCst);
        eprintln!("endex MCP: reindexed {n} changed file(s)");
    }
}

/// Background embedder: keeps block vectors warm. Runs once at startup and
/// again whenever the watcher reports changes; never blocks tool calls.
/// Embeds into a delta store and merges it back under a brief write lock —
/// no full-store clones except the one snapshot needed to persist the cache.
fn spawn_embedder(provider: Provider) {
    std::thread::spawn(move || {
        let mut seen_epoch = 0usize;
        loop {
            // Wait until there is something to do.
            let epoch = change_epoch().load(Ordering::SeqCst);
            if epoch == seen_epoch {
                std::thread::sleep(std::time::Duration::from_millis(250));
                continue;
            }
            seen_epoch = epoch;

            let pass =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| embed_pass(&provider)));
            match pass {
                Ok(Ok((added, total))) => {
                    if total > 0 {
                        eprintln!("endex MCP: embedded {added}/{total} block(s) in background");
                    }
                }
                Ok(Err(e)) => {
                    // Provider down (e.g. Ollama not running): back off so we
                    // don't spin; the watcher epoch will retry us later.
                    eprintln!("endex MCP: background embed failed: {e} (retrying in 30s)");
                    std::thread::sleep(std::time::Duration::from_secs(30));
                    seen_epoch = 0; // force retry (epoch is always >= 1; cf. run)
                }
                Err(_) => {
                    eprintln!("endex MCP: embedder panicked on a pass; continuing");
                }
            }
        }
    });
}

/// One embed pass. Returns (added, total-missing); error = provider failure.
fn embed_pass(provider: &Provider) -> Result<(usize, usize), String> {
    // 1. Short locks: reset-for-provider in place + collect missing texts.
    let (missing, root) = {
        let sh_arc = shared();
        let sh = mlock(&sh_arc);
        {
            let emb_arc = emb();
            let mut w = wlock(&emb_arc);
            embed::reset_for_provider(&mut w, provider);
        }
        let emb_arc = emb();
        let guard = rlock(&emb_arc);
        let m = embed::collect_missing(&sh.idx, &guard);
        (m, sh.root.clone())
    };
    if missing.is_empty() {
        return Ok((0, 0));
    }
    let total = missing.len();
    eprintln!("endex MCP: embedding {total} block(s) in background");

    // 2. NO lock held: slow HTTP embedding into a fresh delta store.
    let mut delta = Embeddings {
        provider_id: provider.id(),
        dim: rlock(&emb()).dim,
        map: std::collections::HashMap::new(),
    };
    embed::embed_texts(&mut delta, provider, missing, false)?;
    let added = delta.map.len();

    // 3. Brief write lock to merge the delta, then persist a snapshot
    //    through the index copy for cache round-trips.
    {
        let emb_arc = emb();
        let mut w = wlock(&emb_arc);
        if w.dim == 0 {
            w.dim = delta.dim;
        }
        w.map.extend(delta.map);
    }
    let snapshot = rlock(&emb()).clone();
    let sh_arc = shared();
    let mut sh = mlock(&sh_arc);
    sh.idx.embeddings = snapshot;
    let now = unix_now();
    if store::save(&sh.idx, &root).is_ok() {
        sh.stats.last_save_unix = Some(now);
    }
    sh.stats.embed_runs += 1;
    sh.stats.last_embed_unix = Some(now);
    sh.stats.last_embed_blocks = added;
    Ok((added, total))
}

// ---------- main loop ----------

pub fn run(dir: String, use_cache: bool, provider: Provider) {
    let root = PathBuf::from(&dir)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&dir));

    // Manifest check (cheap): is the on-disk cache for the same embedding
    // provider? If not, its vectors are useless — start the embedding store
    // empty rather than trusting stale vectors in a different model space.
    let manifest = store::load_manifest(&root);
    if let Some(m) = &manifest {
        if !m.embedding_provider.is_empty() && m.embedding_provider != provider.id() {
            eprintln!(
                "endex MCP: cache embeddings are for '{}' but provider is '{}' — re-embedding in background",
                m.embedding_provider,
                provider.id()
            );
        }
    }

    // Load or build the index (fast: cache hit ~ms, full build ~hundreds of ms).
    let mut idx = if use_cache {
        match store::load(&root) {
            Some(i) => i,
            None => {
                let mut i = Index::new(&root);
                i.build(&root);
                let _ = store::save(&i, &root);
                i
            }
        }
    } else {
        let mut i = Index::new(&root);
        i.build(&root);
        i
    };

    // Partial invalidation: if the cached tree differs from the current one,
    // `refresh` re-parses only changed files (content-hash keying keeps the
    // vectors of every unchanged block). Log how much changed.
    let pre = idx.corpus_fingerprint();
    let changed = idx.refresh(&root);
    if changed > 0 {
        eprintln!(
            "endex MCP: partial refresh — {changed} file(s) changed since cache (fingerprint {pre:#x} -> {:#x})",
            idx.corpus_fingerprint()
        );
        let _ = store::save(&idx, &root);
    } else if idx.graph.symbols.is_empty() && !idx.files.is_empty() {
        idx.finalize();
    }

    // The live embedding store starts from whatever the cache had — UNLESS
    // the provider changed, in which case we start empty and let the
    // background embedder repopulate it in the correct vector space.
    let cached_emb = if idx.embeddings.provider_id == provider.id() {
        idx.embeddings.clone()
    } else {
        Embeddings {
            provider_id: provider.id(),
            ..Embeddings::default()
        }
    };
    let emb_state = Arc::new(RwLock::new(cached_emb));
    let shared_state = Arc::new(Mutex::new(Shared {
        root: root.clone(),
        idx,
        dirty: false,
        stats: ServerStats {
            started_unix: unix_now(),
            indexed_files: changed,
            last_index_unix: None,
            last_embed_unix: None,
            last_embed_blocks: 0,
            embed_runs: 0,
            last_save_unix: None,
            index_runs: usize::from(changed > 0),
        },
    }));
    if SHARED.set(shared_state).is_err() {
        panic!("shared state installed exactly once");
    }
    if EMB.set(emb_state).is_err() {
        panic!("embedding state installed exactly once");
    }

    if provider.kind == embed::ProviderKind::Cohere && provider.key.is_empty() {
        eprintln!(
            "endex MCP: Cohere was requested but no API key is set \
             (COHERE_API_KEY / EMBED_API_KEY) — queries degrade to lexical \
             results until a key is configured; use --embed-provider hash \
             for fully offline search"
        );
    }

    eprintln!(
        "endex MCP server ready (dir: {}, provider: {})",
        root.display(),
        provider.id()
    );

    // Always-on background tasks: watch for changes, keep embeddings warm.
    spawn_watcher();
    // Bump the epoch so the embedder does an initial warm-up pass.
    change_epoch().fetch_add(1, Ordering::SeqCst);
    spawn_embedder(provider.clone());

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut buf = Vec::new();

    loop {
        let msg = match read_message(&mut stdin, &mut buf) {
            Ok(Some(m)) => m,
            Ok(None) => break, // EOF
            Err(_) => break,   // IO error
        };
        if msg.is_null() {
            continue;
        }

        // Batch requests: process each element.
        let requests: Vec<Value> = match &msg {
            Value::Array(v) => v.clone(),
            other => vec![other.clone()],
        };

        for req in requests {
            let method = req.get("method").and_then(Value::as_str).unwrap_or("");
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            let is_notification = req.get("id").is_none();

            let response = match method {
                "initialize" => rpc_ok(
                    &id,
                    json!({
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "endex", "version": env!("CARGO_PKG_VERSION") }
                    }),
                ),
                "ping" => rpc_ok(&id, json!({})),
                "tools/list" => rpc_ok(&id, json!({ "tools": tool_defs() })),
                "tools/call" => {
                    let params = req.get("params").cloned().unwrap_or(json!({}));
                    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = params.get("arguments").cloned().unwrap_or(json!({}));
                    // A panicking tool must return an error, not kill the server
                    // (release profile no longer uses panic = "abort").
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        exec_tool(&provider, name, &args)
                    }))
                    .unwrap_or_else(|_| tool_err("internal error: endex tool panicked"));
                    rpc_ok(&id, result)
                }
                "notifications/initialized" | "notifications/cancelled" => continue,
                _ => {
                    if is_notification {
                        continue;
                    }
                    rpc_err(&id, -32601, &format!("method not found: {method}"))
                }
            };

            if is_notification {
                continue;
            }
            write_message(&mut out, &response);
        }
    }

    // Save on shutdown if the watcher left unsaved changes.
    let sh_arc = shared();
    let mut sh = mlock(&sh_arc);
    if sh.dirty {
        let root = sh.root.clone();
        let _ = store::save(&sh.idx, &root);
        sh.dirty = false;
    }
}
