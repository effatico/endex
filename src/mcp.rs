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
use crate::graph::{Graph, SymbolKind};
use crate::index::Index;
use crate::{search, store, watch};
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

const PROTOCOL_VERSION: &str = "2025-06-18";
/// Files the server itself writes — the watcher must ignore them or it
/// reindexes its own output forever.
const SELF_WRITTEN: &[&str] = &[
    ".endex-index.bin",
    ".endex-index.tmp",
    ".endex-manifest.json",
    ".endex-manifest.tmp",
];

/// Global state shared by all tool calls and the background tasks.
struct Shared {
    idx: Index,
    root: PathBuf,
    dirty: bool,
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

/// All exposed tools, with JSON Schemas for their arguments.
fn tool_defs() -> Value {
    json!([
        {
            "name": "endex_index",
            "description": "Build or refresh the endex code index for a directory. IMPORTANT: call this FIRST when starting work in a repo where endex has not been run yet — it enables all other endex_* tools (subsequent calls are incremental, only changed files are re-parsed, and the other tools auto-load the index anyway). Skip it if endex_status already returns stats for this directory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dir": { "type": "string", "description": "Directory to index (absolute or relative). Default: the directory the server was started with." }
                }
            }
        },
        {
            "name": "endex_search",
            "description": "Fast substring search over the whole codebase, ranked by relevance. PREFER THIS over Grep/Glob when looking for where an identifier, function name, error message, or literal string is used: results are pre-ranked, deduplicated per block, and each hit INCLUDES the full code block text (typically the whole function), so you usually do NOT need a follow-up Read of the file. Sub-millisecond even on large repos.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Case-insensitive substring to find — e.g. a function name, type name, log message, or error string." },
                    "dir": { "type": "string" },
                    "limit": { "type": "integer", "description": "Max results (default 20, max 100)." }
                },
                "required": ["query"]
            }
        },
        {
            "name": "endex_ask",
            "description": "Semantic search over the codebase in NATURAL LANGUAGE — use this whenever you do not know the exact identifier: 'how do we handle retries', 'where is rate limiting enforced', 'authentication middleware'. STRONGLY PREFER this as the FIRST step when exploring an unfamiliar codebase or concept, instead of guessing grep patterns. Each hit includes the full code block text, so follow-up file reads are rarely needed. Combines lexical + embedding similarity (vectors are kept warm by a background embedder; if coverage < 1.0 the index is still warming up and results are lexical-leaning — retry shortly).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language question or description of the code you are looking for." },
                    "dir": { "type": "string" },
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
                    "symbol": { "type": "string", "description": "Symbol name (function, method, class, struct, ...)." },
                    "dir": { "type": "string" }
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
                    "dir": { "type": "string" },
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
                    "dir": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["term"]
            }
        },
        {
            "name": "endex_status",
            "description": "Check whether the endex index is ready for a directory and what it contains (files, blocks, symbols, call edges, embedding vectors, semantic coverage). Call this to verify setup before relying on the other endex_* tools.",
            "inputSchema": {
                "type": "object",
                "properties": { "dir": { "type": "string" } }
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

fn tool_err(msg: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": msg }],
        "isError": true
    })
}

/// Extract the JSON payload of one inbound message. Supports both
/// newline-delimited JSON and `Content-Length`-framed messages.
fn read_message(stdin: &mut dyn BufRead, buf: &mut Vec<u8>) -> io::Result<Option<Value>> {
    buf.clear();
    let mut byte = [0u8; 1];
    // Peek at the first non-whitespace byte to detect framing.
    let first = loop {
        match stdin.read(&mut byte) {
            Ok(0) => return Ok(None), // EOF
            Ok(_) => {
                let c = byte[0];
                if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                    continue;
                }
                break c;
            }
            Err(e) => return Err(e),
        }
    };

    if first == b'{' {
        // Newline-delimited JSON.
        buf.push(first);
        stdin.read_until(b'\n', buf)?;
        let s = String::from_utf8_lossy(buf);
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(Some(Value::Null));
        }
        return Ok(Some(serde_json::from_str(trimmed).unwrap_or(Value::Null)));
    }

    // Assume Content-Length framing.
    let mut content_length: Option<usize> = None;
    buf.push(first);
    loop {
        let mut line = Vec::new();
        stdin.read_until(b'\n', &mut line)?;
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = text.to_lowercase().strip_prefix("content-length:") {
            content_length = rest.trim().parse().ok();
        }
    }
    let n = match content_length {
        Some(n) => n,
        None => return Ok(Some(Value::Null)),
    };
    let mut body = vec![0u8; n];
    let mut read = 0;
    while read < n {
        match stdin.read(&mut body[read..]) {
            Ok(0) => break,
            Ok(k) => read += k,
            Err(e) => return Err(e),
        }
    }
    Ok(Some(
        serde_json::from_slice(&body[..read]).unwrap_or(Value::Null),
    ))
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
    let live = idx
        .blocks
        .iter()
        .filter(|b| b.file != crate::index::TOMBSTONE_FILE)
        .count();
    if live == 0 {
        return 1.0;
    }
    let covered = idx
        .blocks
        .iter()
        .filter(|b| b.file != crate::index::TOMBSTONE_FILE)
        .filter(|b| emb.map.contains_key(&embed::fnv64(&b.text)))
        .count();
    covered as f32 / live as f32
}

fn exec_tool(provider: &Provider, name: &str, args: &Value) -> Value {
    match name {
        "endex_index" => {
            let sh_arc = shared();
            let mut sh = sh_arc.lock().unwrap();
            // A watcher keeps the index fresh; a manual call forces a refresh.
            let root = sh.root.clone();
            let changed = sh.idx.refresh(&root);
            if changed > 0 {
                sh.dirty = true;
                change_epoch().fetch_add(1, Ordering::SeqCst);
            }
            let idx = &sh.idx;
            let emb_arc = emb();
            let emb = emb_arc.read().unwrap();
            tool_ok(
                json!({
                    "dir": sh.root,
                    "refreshed_files": changed,
                    "files": idx.file_count(),
                    "blocks": idx.block_count(),
                    "symbols": idx.graph.symbols.len(),
                    "call_edges": idx.graph.call_edge_count(),
                    "import_edges": idx.graph.file_imports.len(),
                    "semantic_coverage": coverage(idx, &emb),
                    "message": "index ready (filesystem watcher active; changes are picked up automatically)"
                })
                .to_string(),
            )
        }

        "endex_status" => {
            let sh_arc = shared();
            let sh = sh_arc.lock().unwrap();
            let emb_store = emb();
            let emb = emb_store.read().unwrap();
            let idx = &sh.idx;
            let g = &idx.graph;
            let kind_counts: Value = {
                let mut m = serde_json::Map::new();
                for k in [
                    SymbolKind::Func,
                    SymbolKind::Method,
                    SymbolKind::Class,
                    SymbolKind::Struct,
                    SymbolKind::Enum,
                    SymbolKind::Trait,
                    SymbolKind::Type,
                    SymbolKind::Interface,
                    SymbolKind::Impl,
                ] {
                    let n = g.symbols.iter().filter(|s| s.kind == k).count();
                    if n > 0 {
                        m.insert(k.label().to_string(), Value::from(n));
                    }
                }
                Value::Object(m)
            };
            tool_ok(
                json!({
                    "dir": sh.root,
                    "files": idx.file_count(),
                    "blocks": idx.block_count(),
                    "symbols": g.symbols.len(),
                    "symbol_kinds": kind_counts,
                    "call_edges": g.call_edge_count(),
                    "import_edges": g.file_imports.len(),
                    "embedding_vectors": emb.map.len(),
                    "embedding_dim": emb.dim,
                    "embedding_provider": provider.id(),
                    "semantic_coverage": coverage(idx, &emb),
                    "corpus_fingerprint": format!("{:#x}", idx.corpus_fingerprint()),
                })
                .to_string(),
            )
        }

        "endex_search" => {
            let query = match arg_str(args, "query") {
                Some(q) if !q.is_empty() => q,
                _ => return tool_err("missing required argument: query"),
            };
            let limit = arg_usize(args, "limit", 20).min(100);
            let sh_arc = shared();
            let sh = sh_arc.lock().unwrap();
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
            tool_ok(json!({ "query": query, "count": hits.len(), "hits": out }).to_string())
        }

        "endex_ask" => {
            let query = match arg_str(args, "query") {
                Some(q) if !q.is_empty() => q,
                _ => return tool_err("missing required argument: query"),
            };
            let limit = arg_usize(args, "limit", 20).min(100);
            let sh_arc = shared();
            let sh = sh_arc.lock().unwrap();
            let idx = &sh.idx;
            let emb = emb().read().unwrap().clone(); // snapshot, cheap vs HTTP
                                                     // ask_fast: embeds ONLY the query, never the corpus. The
                                                     // background embedder keeps block vectors warm; coverage tells
                                                     // the caller how semantic the current results are.
            match embed::ask_fast(idx, &emb, provider, query, limit) {
                Ok((hits, cov)) => {
                    let out: Vec<Value> = hits
                        .iter()
                        .map(|(score, h)| {
                            json!({
                                "file": idx.path_of(h.file_id),
                                "line": h.line,
                                "score": score,
                                "text": h.text,
                            })
                        })
                        .collect();
                    tool_ok(
                        json!({
                            "query": query,
                            "provider": provider.id(),
                            "semantic_coverage": cov,
                            "warming_up": cov < 0.95,
                            "count": hits.len(),
                            "hits": out
                        })
                        .to_string(),
                    )
                }
                Err(e) => {
                    // Provider unreachable: degrade to lexical, say so.
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
                    tool_ok(
                        json!({
                            "query": query,
                            "warning": format!("semantic provider unavailable ({e}); results are lexical only"),
                            "semantic_coverage": 0.0,
                            "count": hits.len(),
                            "hits": out
                        })
                        .to_string(),
                    )
                }
            }
        }

        "endex_graph" => {
            let sym = match arg_str(args, "symbol") {
                Some(s) if !s.is_empty() => s,
                _ => return tool_err("missing required argument: symbol"),
            };
            let sh_arc = shared();
            let sh = sh_arc.lock().unwrap();
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
            tool_ok(json!({ "symbol": sym, "matches": out }).to_string())
        }

        "endex_clues" => {
            let term = match arg_str(args, "term") {
                Some(t) if !t.is_empty() => t,
                _ => return tool_err("missing required argument: term"),
            };
            let limit = arg_usize(args, "limit", 15).min(100);
            let sh_arc = shared();
            let sh = sh_arc.lock().unwrap();
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
            tool_ok(json!({ "term": term, "count": hits.len(), "hits": out }).to_string())
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
            let sh = sh_arc.lock().unwrap();
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
                return tool_ok(
                    json!({
                        "from": from,
                        "to": to,
                        "paths": [],
                        "message": format!("no call path found (max depth {max_depth})")
                    })
                    .to_string(),
                );
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
            tool_ok(
                json!({
                    "from": from,
                    "to": to,
                    "path_count": paths.len(),
                    "paths": out
                })
                .to_string(),
            )
        }

        _ => tool_err(&format!("unknown tool: {name}")),
    }
}

// ---------- background tasks ----------

/// Filesystem watcher: reindexes changed files, rebuilds the graph, saves
/// the cache, and pokes the embedder via the change epoch.
fn spawn_watcher() {
    let sh = shared();
    let root = sh.lock().unwrap().root.clone();
    std::thread::spawn(move || {
        let rx = match watch::watch(&root) {
            Ok(rx) => rx,
            Err(e) => {
                eprintln!("endex MCP: watcher unavailable: {e} (index will not auto-refresh)");
                return;
            }
        };
        for batch in rx.iter() {
            let sh_arc = shared();
            let mut sh = sh_arc.lock().unwrap();
            let mut n = 0usize;
            for p in &batch {
                if p.file_name()
                    .map(|f| SELF_WRITTEN.iter().any(|s| f == *s))
                    .unwrap_or(false)
                    || p.components().any(|c| c.as_os_str() == ".git")
                {
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
                sh.dirty = true;
                let root = sh.root.clone();
                let _ = store::save(&sh.idx, &root);
                sh.dirty = false;
                change_epoch().fetch_add(1, Ordering::SeqCst);
                eprintln!("endex MCP: reindexed {n} changed file(s)");
            }
        }
    });
}

/// Background embedder: keeps block vectors warm. Runs once at startup and
/// again whenever the watcher reports changes; never blocks tool calls.
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

            // 1. Short lock: reset-for-provider + collect missing texts.
            let (missing, root) = {
                let sh_arc = shared();
                let sh = sh_arc.lock().unwrap();
                let mut snapshot = emb().read().unwrap().clone();
                embed::reset_for_provider(&mut snapshot, &provider);
                // Write back the (possibly reset) snapshot so coverage is
                // honest while we work.
                let m = embed::collect_missing(&sh.idx, &snapshot);
                *emb().write().unwrap() = snapshot;
                (m, sh.root.clone())
            };
            if missing.is_empty() {
                continue;
            }
            let total = missing.len();
            eprintln!("endex MCP: embedding {total} block(s) in background");

            // 2. NO lock held: slow HTTP embedding into a working copy.
            let mut work = emb().read().unwrap().clone();
            let before = work.map.len();
            match embed::embed_texts(&mut work, &provider, missing, false) {
                Ok(()) => {
                    let added = work.map.len().saturating_sub(before);
                    // 3. Brief write lock to swap in the new vectors, then
                    // persist through the index copy for cache round-trips.
                    *emb().write().unwrap() = work.clone();
                    let sh_arc = shared();
                    let mut sh = sh_arc.lock().unwrap();
                    sh.idx.embeddings = work;
                    let _ = store::save(&sh.idx, &root);
                    eprintln!("endex MCP: embedded {added}/{total} block(s) in background");
                }
                Err(e) => {
                    // Provider down (e.g. Ollama not running): back off so we
                    // don't spin; the watcher epoch will retry us later.
                    eprintln!("endex MCP: background embed failed: {e} (retrying in 30s)");
                    std::thread::sleep(std::time::Duration::from_secs(30));
                    seen_epoch = 0; // force retry
                }
            }
        }
    });
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
    }));
    if SHARED.set(shared_state).is_err() {
        panic!("shared state installed exactly once");
    }
    if EMB.set(emb_state).is_err() {
        panic!("embedding state installed exactly once");
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
                    let result = exec_tool(&provider, name, &args);
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
    let mut sh = sh_arc.lock().unwrap();
    if sh.dirty {
        let root = sh.root.clone();
        let _ = store::save(&sh.idx, &root);
        sh.dirty = false;
    }
}
