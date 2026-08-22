//! End-to-end tests: build → search → mutate → re-search, plus cache
//! round-trips. Uses throwaway temp directories.

use endex::{index::Index, search, store};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

/// Unique temp dir per test, cleaned up on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let n = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!(
            "endex-test-{}-{}-{}",
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

fn hit_paths(idx: &Index, q: &str) -> Vec<String> {
    search::search(idx, q, 100)
        .into_iter()
        .map(|h| idx.path_of(h.file_id).to_string())
        .collect()
}

// ---------- indexing & search ----------

#[test]
fn builds_and_finds_substrings_case_insensitively() {
    let tmp = TempDir::new();
    tmp.write(
        "src/lib.rs",
        "fn calculate_tax(income: u64) -> u64 {\n    income / 4\n}\n",
    );
    tmp.write("src/other.ts", "const unrelated = 1;\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert_eq!(idx.file_count(), 2);

    let hits = search::search(&idx, "calculate_tax", 10);
    assert_eq!(hits.len(), 1);
    // Separator-agnostic: Windows paths use backslashes.
    assert!(idx
        .path_of(hits[0].file_id)
        .replace('\\', "/")
        .ends_with("src/lib.rs"));
    assert_eq!(hits[0].line, 1);

    // Case-insensitive.
    assert_eq!(search::search(&idx, "CALCULATE_TAX", 10).len(), 1);
    // Partial substring.
    assert_eq!(search::search(&idx, "ulate_ta", 10).len(), 1);
    // No match.
    assert!(search::search(&idx, "nope_xyz", 10).is_empty());
}

#[test]
fn blocks_are_blank_line_separated() {
    let tmp = TempDir::new();
    tmp.write(
        "a.txt",
        "first block\nsecond line\n\nthird block\n\n\nfourth block\n",
    );

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    // 3 blocks, and the third one starts at line 4 (line 3 is blank).
    let hits = search::search(&idx, "third block", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].line, 4);
}

#[test]
fn short_queries_fall_back_to_scan() {
    let tmp = TempDir::new();
    tmp.write("a.txt", "ab cd\n");
    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert_eq!(search::search(&idx, "cd", 10).len(), 1);
    assert_eq!(search::search(&idx, "z", 10).len(), 0);
}

#[test]
fn skips_binaries_and_oversized_files() {
    let tmp = TempDir::new();
    tmp.write("binary.bin", "text\0with nul\0bytes\n");
    let big = "x".repeat(6 * 1024 * 1024);
    tmp.write("big.txt", &big);

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert_eq!(idx.file_count(), 0);
}

// ---------- incremental updates ----------

#[test]
fn reindex_after_edit_reflects_changes() {
    let tmp = TempDir::new();
    let file = tmp.write("a.rs", "fn alpha_func() {}\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert_eq!(search::search(&idx, "alpha_func", 10).len(), 1);

    // Unchanged file: index_file must be a no-op.
    assert!(!idx.index_file(&file));

    // Edit: old posting must disappear, new one must appear.
    fs::write(&file, "fn beta_func() {}\n").unwrap();
    assert!(idx.index_file(&file));
    assert!(search::search(&idx, "alpha_func", 10).is_empty());
    assert_eq!(search::search(&idx, "beta_func", 10).len(), 1);

    // Delete: everything must disappear.
    fs::remove_file(&file).unwrap();
    idx.remove_file(&file);
    assert!(search::search(&idx, "beta_func", 10).is_empty());
    assert_eq!(idx.file_count(), 0);
}

#[test]
fn refresh_picks_up_new_and_deleted_files() {
    let tmp = TempDir::new();
    tmp.write("a.txt", "keep_me\n");
    let b = tmp.write("b.txt", "remove_me\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    assert_eq!(idx.file_count(), 2);

    fs::remove_file(&b).unwrap();
    let c = tmp.write("c.txt", "new_arrival\n");
    let changed = idx.refresh(tmp.path());
    assert_eq!(changed, 2); // b removed, c added
    assert_eq!(idx.file_count(), 2);
    assert!(hit_paths(&idx, "remove_me").is_empty());
    let hits = hit_paths(&idx, "new_arrival");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0], c.to_str().unwrap());
    assert_eq!(search::search(&idx, "keep_me", 10).len(), 1);
    assert_eq!(search::search(&idx, "new_arrival", 10).len(), 1);
}

// ---------- cache ----------

#[test]
fn cache_round_trip_preserves_search_results() {
    let tmp = TempDir::new();
    tmp.write("a.rs", "fn find_me() {}\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    store::save(&idx, tmp.path()).unwrap();

    let loaded = store::load(tmp.path()).expect("cache should load");
    assert_eq!(loaded.file_count(), idx.file_count());
    assert_eq!(loaded.block_count(), idx.block_count());
    assert_eq!(search::search(&loaded, "find_me", 10).len(), 1);
    assert!(search::search(&loaded, "missing", 10).is_empty());
}

#[test]
fn cache_rejects_garbage() {
    let tmp = TempDir::new();
    let p = store::cache_path(tmp.path());
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, b"not a valid cache").unwrap();
    assert!(store::load(tmp.path()).is_none());
    // Clean up: the cache lives in ~/.endex/cache now, not in the temp dir.
    fs::remove_file(&p).ok();
}

// ---------- ranking ----------

#[test]
fn results_ranked_by_occurrence_count() {
    let tmp = TempDir::new();
    tmp.write("once.txt", "mention spam once\n");
    tmp.write("thrice.txt", "spam here\nspam there\nspam everywhere\n");

    let mut idx = Index::new(tmp.path());
    idx.build(tmp.path());
    let hits = search::search(&idx, "spam", 10);
    assert_eq!(hits.len(), 2);
    assert!(idx.path_of(hits[0].file_id).contains("thrice"));
    assert!(idx.path_of(hits[1].file_id).contains("once"));
}
