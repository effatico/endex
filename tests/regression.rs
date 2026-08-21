//! Regression tests for: unicode-safe highlighting (no char-boundary panic),
//! MCP message framing robustness, watcher ignore rules, and posting-list
//! consistency under repeated edits.

use endex::{index::Index, mcp, output, search, watch};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let n = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!(
            "endex-regtest-{}-{}-{}",
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
        let _ = fs::remove_dir_all(&self.0);
    }
}

// ---------- unicode-safe highlighting ----------

#[test]
fn match_ranges_stay_on_char_boundaries() {
    // 'İ' (U+0130) lowercases to "i̇" — 2 bytes become 3. Offsets found in
    // the lowercased string must never be used to slice the original.
    let r = output::match_ranges("İ日本語abc", "abc");
    assert_eq!(r, vec![(11, 14)]);

    // İ=2B ×3, é=2B -> 日 at bytes 8..11 in the ORIGINAL string.
    let r = output::match_ranges("İİİé日", "日");
    assert_eq!(r, vec![(8, 11)]);

    // This exact input used to panic at a non-char-boundary slice.
    output::print_block_matches("İİİé日\n", 1, "日", 6);

    // Plain ASCII behaves as before.
    let r = output::match_ranges("let s = TOKEN;", "token");
    assert_eq!(r, vec![(8, 13)]);

    // No match -> empty.
    assert!(output::match_ranges("nothing here", "zzz").is_empty());
}

// ---------- MCP message framing ----------

#[test]
fn read_message_survives_garbage_lines() {
    let input = "garbage line\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n";
    let mut cur = std::io::Cursor::new(input.as_bytes().to_vec());
    let mut buf = Vec::new();
    let first = mcp::read_message(&mut cur, &mut buf).unwrap().unwrap();
    assert!(first.is_null()); // garbage skipped, stream NOT corrupted
    let second = mcp::read_message(&mut cur, &mut buf).unwrap().unwrap();
    assert_eq!(second["id"], 2);
}

#[test]
fn read_message_parses_batch_arrays() {
    let input = "[{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"ping\"},{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\"}]\n";
    let mut cur = std::io::Cursor::new(input.as_bytes().to_vec());
    let mut buf = Vec::new();
    let msg = mcp::read_message(&mut cur, &mut buf).unwrap().unwrap();
    assert!(msg.is_array());
    assert_eq!(msg.as_array().unwrap().len(), 2);
}

#[test]
fn read_message_parses_content_length_framing() {
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    let input = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
    let mut cur = std::io::Cursor::new(input.into_bytes());
    let mut buf = Vec::new();
    let msg = mcp::read_message(&mut cur, &mut buf).unwrap().unwrap();
    assert_eq!(msg["method"], "ping");
}

#[test]
fn read_message_reports_eof() {
    let mut cur = std::io::Cursor::new(Vec::new());
    let mut buf = Vec::new();
    assert!(mcp::read_message(&mut cur, &mut buf).unwrap().is_none());
}

// ---------- watcher ignore rules ----------

#[test]
fn ignores_skip_gitignored_hidden_and_self_written_files() {
    let tmp = TempDir::new();
    tmp.write(".gitignore", "target/\n");
    // .gitignore rules only apply inside a git repository (walk semantics).
    fs::create_dir(tmp.path().join(".git")).unwrap();
    let mut ig = watch::Ignores::new(tmp.path());

    assert!(ig.is_ignored(&tmp.path().join("target/out.rs"), false));
    assert!(ig.is_ignored(&tmp.path().join("target"), true));
    assert!(ig.is_ignored(&tmp.path().join(".env"), false));
    assert!(ig.is_ignored(&tmp.path().join(".git/HEAD"), false));
    assert!(ig.is_ignored(&tmp.path().join(".endex-index.bin"), false));
    assert!(ig.is_ignored(&tmp.path().join(".endex-manifest.json"), false));
    assert!(!ig.is_ignored(&tmp.path().join("src/main.rs"), false));
    assert!(!ig.is_ignored(&tmp.path().join("a.txt"), false));
}

#[test]
fn watcher_ignored_files_never_enter_the_index() {
    let tmp = TempDir::new();
    tmp.write(".gitignore", "target/\n");
    fs::create_dir(tmp.path().join(".git")).unwrap();
    let env = tmp.write(".env", "SUPERSECRET=1\n");
    let gen = tmp.write("target/junk.rs", "fn ignored_symbol() {}\n");

    // Simulate what the watcher does now: filter first, index only the rest.
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let mut ig = watch::Ignores::new(tmp.path());
    for p in [&env, &gen] {
        if ig.is_ignored(p, false) {
            continue;
        }
        idx.index_file(p);
    }
    assert!(search::search(&idx, "SUPERSECRET", 10).is_empty());
    assert!(search::search(&idx, "ignored_symbol", 10).is_empty());
}

// ---------- posting-list consistency under churn ----------

#[test]
fn repeated_edits_and_recreated_files_keep_postings_consistent() {
    let tmp = TempDir::new();
    let f = tmp.write("a.rs", "fn version_one() {}\n");
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());

    // Many edits: each tombstones the old block and recycles the id.
    for i in 2..20 {
        fs::write(&f, format!("fn version_{i}() {{}}\n")).unwrap();
        assert!(idx.index_file(&f));
    }
    // Only the latest content is searchable, exactly once.
    assert_eq!(search::search(&idx, "version_19", 10).len(), 1);
    assert!(search::search(&idx, "version_one", 10).is_empty());
    assert!(search::search(&idx, "version_5", 10).is_empty());
    assert_eq!(idx.block_count(), 1);

    // Delete, then recreate: recycled ids must not resurrect stale hits.
    fs::remove_file(&f).unwrap();
    idx.remove_file(&f);
    fs::write(&f, "fn version_one() {}\n").unwrap();
    assert!(idx.index_file(&f));
    assert_eq!(search::search(&idx, "version_one", 10).len(), 1);
    assert!(search::search(&idx, "version_19", 10).is_empty());
}
