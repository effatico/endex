//! Tests for the knowledge graph, embeddings, and hybrid search.

use endex::{embed, index::Index, search, store};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);
static CACHE_ENV: Once = Once::new();

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        // Redirect the on-disk cache into the system temp dir: tests must
        // never write into the developer's real ~/.endex/cache. Set once
        // per test process, before any store path is resolved.
        CACHE_ENV.call_once(|| {
            let d = std::env::temp_dir().join(format!("endex-test-cache-{}", std::process::id()));
            std::env::set_var("ENDEX_CACHE_DIR", &d);
        });
        let n = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!(
            "endex-graph-test-{}-{}-{}",
            std::process::id(),
            n,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }

    fn write(&self, rel: &str, contents: &str) -> PathBuf {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, contents).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Resolve the cache dir BEFORE deleting the project (it canonicalizes
        // the root), so no cache leaks behind after the test run.
        let cache = store::cache_dir(&self.0);
        let _ = fs::remove_dir_all(&cache);
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn hash_provider() -> embed::Provider {
    embed::Provider::resolve(&embed::ProviderOpts {
        provider: Some("hash".into()),
        dim: Some(256),
        ..Default::default()
    })
}

// ---------- graph ----------

#[test]
fn graph_extracts_defs_calls_and_imports() {
    let tmp = TempDir::new();
    tmp.write(
        "a.ts",
        "import { serve } from './b';\n\nfunction main() {\n  serve();\n}\n",
    );
    tmp.write(
        "b.ts",
        "export function serve() {\n  listen();\n}\n\nexport function listen() {}\n",
    );

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let g = &idx.graph;

    let main = g.find_all("main");
    let serve = g.find_all("serve");
    let listen = g.find_all("listen");
    assert_eq!(main.len(), 1);
    assert_eq!(serve.len(), 1);
    assert_eq!(listen.len(), 1);

    // call edges: main -> serve -> listen
    assert!(g.callees(main[0]).contains(&serve[0]));
    assert!(g.callees(serve[0]).contains(&listen[0]));
    assert!(g.callers(listen[0]).contains(&serve[0]));

    // kinds
    assert_eq!(
        g.symbols[serve[0] as usize].kind,
        endex::graph::SymbolKind::Func
    );

    // file import edge a.ts -> b.ts
    let file_ids: Vec<u32> = idx.files.values().map(|fe| fe.id).collect();
    assert_eq!(g.file_imports.len(), 1);
    assert!(file_ids.contains(&g.file_imports[0].1));
}

#[test]
fn graph_rust_defs_and_calls() {
    let tmp = TempDir::new();
    tmp.write(
        "lib.rs",
        "fn compute(x: u64) -> u64 {\n    x * 2\n}\n\nfn caller() {\n    compute(21);\n}\n",
    );
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let g = &idx.graph;
    let compute = g.find_all("compute");
    let caller = g.find_all("caller");
    assert_eq!(compute.len(), 1);
    assert_eq!(caller.len(), 1);
    assert!(g.callees(caller[0]).contains(&compute[0]));
    assert!(g.callers(compute[0]).contains(&caller[0]));
}

#[test]
fn flow_finds_shortest_call_path() {
    let tmp = TempDir::new();
    tmp.write(
        "a.ts",
        "import { serve } from './b';\n\nfunction main() {\n  serve();\n}\n",
    );
    tmp.write(
        "b.ts",
        "export function serve() {\n  listen();\n}\n\nexport function listen() {}\n",
    );
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let g = &idx.graph;
    let main = g.find_all("main");
    let targets: HashSet<u32> = g.find_all("listen").into_iter().collect();
    let paths = g.find_paths(&main, &targets, 8, 5);
    assert!(!paths.is_empty());
    assert_eq!(paths[0].len(), 3); // main -> serve -> listen
    let names: Vec<&str> = paths[0]
        .iter()
        .map(|&id| g.symbols[id as usize].name.as_str())
        .collect();
    assert_eq!(names, vec!["main", "serve", "listen"]);
}

#[test]
fn graph_updates_incrementally() {
    let tmp = TempDir::new();
    let file = tmp.write(
        "a.rs",
        "fn target() {}\n\nfn caller() {\n    target();\n}\n",
    );
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert!(idx.graph.find_all("target").len() == 1);
    assert!(idx.graph.callers(idx.graph.find_all("target")[0]).len() == 1);

    // Remove the call; graph must reflect it after finalize().
    fs::write(&file, "fn target() {}\n\nfn caller() {\n}\n").unwrap();
    assert!(idx.index_file(&file));
    idx.finalize();
    assert!(idx
        .graph
        .callers(idx.graph.find_all("target")[0])
        .is_empty());

    // Delete the file; its symbols must disappear.
    fs::remove_file(&file).unwrap();
    idx.remove_file(&file);
    idx.finalize();
    assert!(idx.graph.find_all("target").is_empty());
}

#[test]
fn graph_prefers_same_file_definitions_for_common_names() {
    let tmp = TempDir::new();
    tmp.write(
        "a.ts",
        "function save(x: number) { return x; }\n\nfunction mainA() {\n  save(1);\n}\n",
    );
    tmp.write(
        "b.ts",
        "function save(x: number) { return x; }\n\nfunction mainB() {\n  save(2);\n}\n",
    );
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let g = &idx.graph;

    // `save` is defined in both files; each caller must link to the LOCAL
    // definition only, not to every same-named symbol in the repo.
    let saves = g.find_all("save");
    assert_eq!(saves.len(), 2);
    let save_a = saves
        .iter()
        .find(|&&id| idx.path_of(g.symbols[id as usize].file).ends_with("a.ts"))
        .copied()
        .unwrap();
    let save_b = saves
        .iter()
        .find(|&&id| idx.path_of(g.symbols[id as usize].file).ends_with("b.ts"))
        .copied()
        .unwrap();
    let caller_names_a: Vec<&str> = g
        .callers(save_a)
        .iter()
        .map(|&c| g.symbols[c as usize].name.as_str())
        .collect();
    let caller_names_b: Vec<&str> = g
        .callers(save_b)
        .iter()
        .map(|&c| g.symbols[c as usize].name.as_str())
        .collect();
    assert_eq!(caller_names_a, vec!["mainA"]);
    assert_eq!(caller_names_b, vec!["mainB"]);
}

// ---------- embeddings ----------

#[test]
fn hash_embedding_is_deterministic_and_normalized() {
    let p = hash_provider();
    let a = p.embed(&["hello world"]).unwrap().remove(0);
    let b = p.embed(&["hello world"]).unwrap().remove(0);
    assert_eq!(a.len(), 256);
    assert_eq!(a, b);
    let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
    let c = p.embed(&["something else entirely"]).unwrap().remove(0);
    assert_ne!(a, c);
}

#[test]
fn hybrid_search_finds_relevant_blocks() {
    let tmp = TempDir::new();
    tmp.write(
        "billing.ts",
        "function processInvoicePayment(amount: number) {\n  return amount * 1.25;\n}\n",
    );
    tmp.write("other.ts", "const fibonacci = [1, 1, 2, 3, 5];\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let prov = hash_provider();

    // No lexical hit for "invoice payment processing" as a substring, but
    // the fused hash-embedding ranking should surface the invoice block.
    let outcome = embed::ask(&mut idx, &prov, "invoice payment processing", 5).unwrap();
    assert!(outcome.changed);
    // The hash provider has no reranker, so RRF order is kept.
    assert!(!outcome.reranked);
    assert!(!outcome.hits.is_empty());
    assert!(idx
        .path_of(outcome.hits[0].1.file_id)
        .contains("billing.ts"));

    // Pure lexical search still works independently.
    assert_eq!(search::search(&idx, "processInvoicePayment", 5).len(), 1);
}

#[test]
fn embeddings_survive_cache_round_trip_and_gc() {
    let tmp = TempDir::new();
    let a = tmp.write("a.txt", "persist me\n");
    tmp.write("b.txt", "drop me later\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let prov = hash_provider();
    embed::ensure_all(&mut idx, &prov).unwrap();
    assert_eq!(idx.embeddings.map.len(), 2);

    endex::store::save(&idx, tmp.path()).unwrap();
    let mut loaded = endex::store::load(tmp.path()).expect("cache loads");
    assert_eq!(loaded.embeddings.map.len(), 2);
    assert_eq!(loaded.embeddings.provider_id, prov.id());

    // Deleting a file must GC its embedding after finalize().
    fs::remove_file(tmp.path().join("b.txt")).unwrap();
    let b = tmp.path().join("b.txt");
    loaded.remove_file(&b);
    loaded.finalize();
    assert_eq!(loaded.embeddings.map.len(), 1);
    let _ = a;
}

#[test]
fn provider_switch_invalidates_embeddings() {
    let tmp = TempDir::new();
    tmp.write("a.txt", "some content\n");
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());

    let p256 = embed::Provider::resolve(&embed::ProviderOpts {
        provider: Some("hash".into()),
        dim: Some(256),
        ..Default::default()
    });
    embed::ensure_all(&mut idx, &p256).unwrap();
    assert_eq!(idx.embeddings.map.len(), 1);

    // Different dim -> different provider id -> embeddings reset.
    let p128 = embed::Provider::resolve(&embed::ProviderOpts {
        provider: Some("hash".into()),
        dim: Some(128),
        ..Default::default()
    });
    embed::ensure_all(&mut idx, &p128).unwrap();
    assert_eq!(idx.embeddings.provider_id, p128.id());
    assert_eq!(idx.embeddings.map.len(), 1);
}

// ---------- reranking ----------

fn cohere_provider() -> embed::Provider {
    // Resolved with an explicit key so `supports_rerank()` is true; no
    // request is ever made in these tests.
    embed::Provider::resolve(&embed::ProviderOpts {
        provider: Some("cohere".into()),
        key: Some("test-key".into()),
        ..Default::default()
    })
}

#[test]
fn rerank_pool_never_disables_reranking_for_large_limits() {
    let cohere = cohere_provider();
    assert!(cohere.supports_rerank());
    // Small limits get 4x headroom...
    assert_eq!(embed::rerank_pool(&cohere, 5), 20);
    assert_eq!(embed::rerank_pool(&cohere, 20), 80);
    // ...large ones are capped, but the pool is ALWAYS >= limit, so a big
    // limit still gets reranked instead of silently falling back to RRF.
    for limit in [39, 40, 50, 100] {
        let pool = embed::rerank_pool(&cohere, limit);
        assert!(pool >= limit, "pool {pool} must cover limit {limit}");
    }
    // Providers without a reranker never over-fetch.
    let hash = hash_provider();
    assert_eq!(embed::rerank_pool(&hash, 50), 50);
}

#[test]
fn apply_rerank_order_reorders_scores_and_truncates() {
    let items: Vec<(f32, &str)> = vec![(0.1, "a"), (0.2, "b"), (0.3, "c")];
    // Reranker says: c best, then a.
    let order = vec![(2usize, 0.9f32), (0usize, 0.5f32), (1usize, 0.1f32)];
    let out = embed::apply_rerank_order(items, order, 2);
    assert_eq!(out.len(), 2, "truncated to limit");
    assert_eq!(out[0].1, "c");
    assert_eq!(out[1].1, "a");
    assert!((out[0].0 - 0.9).abs() < 1e-6, "score replaced by relevance");
}

#[test]
fn apply_rerank_order_ignores_out_of_range_and_duplicate_indices() {
    // A malformed provider response must not panic or duplicate hits.
    let items: Vec<(f32, &str)> = vec![(0.1, "a"), (0.2, "b")];
    let order = vec![(99usize, 1.0f32), (1usize, 0.8f32), (1usize, 0.7f32)];
    let out = embed::apply_rerank_order(items, order, 10);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, "b");
}

#[test]
fn rerank_order_is_none_without_a_reranker_or_candidates() {
    let hash = hash_provider();
    assert!(embed::rerank_order(&hash, "q", &["a", "b"]).is_none());
    // A single candidate is not worth a request.
    let cohere = cohere_provider();
    assert!(embed::rerank_order(&cohere, "q", &["only"]).is_none());
    assert!(embed::rerank_order(&cohere, "q", &[]).is_none());
}

// ---------- provider defaults ----------

#[test]
fn cohere_default_without_key_falls_back_to_offline_hash() {
    // Guard the privacy/behavior default: a bare `endex ask` with no key
    // must not try to ship code to an external API on every query.
    let saved: Vec<(&str, Option<String>)> = ["EMBED_PROVIDER", "EMBED_API_KEY", "COHERE_API_KEY"]
        .iter()
        .map(|k| (*k, std::env::var(k).ok()))
        .collect();
    for (k, _) in &saved {
        std::env::remove_var(k);
    }

    let (prov, notice) = embed::Provider::resolve_checked(&embed::ProviderOpts::default());
    assert_eq!(prov.kind, embed::ProviderKind::Hash);
    assert!(notice.is_some(), "the downgrade must be announced");

    // An EXPLICIT cohere request is always honored, key or not.
    let (explicit, _) = embed::Provider::resolve_checked(&embed::ProviderOpts {
        provider: Some("cohere".into()),
        ..Default::default()
    });
    assert_eq!(explicit.kind, embed::ProviderKind::Cohere);

    // With a key, the default stays Cohere.
    let (keyed, notice) = embed::Provider::resolve_checked(&embed::ProviderOpts {
        key: Some("k".into()),
        ..Default::default()
    });
    assert_eq!(keyed.kind, embed::ProviderKind::Cohere);
    assert!(notice.is_none());

    for (k, v) in saved {
        match v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
}
