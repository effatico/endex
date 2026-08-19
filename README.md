# endex

An extremely fast, cached, self-updating **code indexer** with **millisecond substring search**, written in Rust.

## How it works

- **Trigram inverted index** (the approach behind Google Code Search / Zoekt): every 3-byte window of the corpus maps to a posting list of *block ids*. A query intersects the posting lists of its trigrams, producing a tiny candidate set that is verified exactly — search runs in **microseconds to a few milliseconds** even on huge corpora.
- **Blocks, not lines**: code is chunked into blank-line-separated blocks (max 80 lines), so results are meaningful text blocks (functions, paragraphs, config sections).
- **Disk cache**: the whole index is serialized with bincode to `.endex-index.bin` inside the indexed directory (atomic tmp+rename writes). Startup from cache is ~25 ms for a 3,500-file repo.
- **File watching**: `watch` mode uses `notify` (FSEvents on macOS) with debouncing. Changed files are **reindexed incrementally** in milliseconds — old postings are surgically removed, new ones inserted. The cache is saved automatically on exit (or `:save`).
- **Case-insensitive**, gitignore-aware (`ignore` crate), skips hidden dirs, binaries (NUL-byte detection), and files > 5 MB. Parallel indexing via `rayon`.

## Usage

```bash
cargo build --release

# Build or refresh the cache (incremental if a cache exists)
./target/release/endex index ~/my-repo

# One-shot search (auto-loads cache, auto-refreshes stale files)
./target/release/endex search ~/my-repo "createUserAccount" --limit 20

# Watch mode: live updates + interactive search REPL
./target/release/endex watch ~/my-repo
```

REPL commands: `:limit N`, `:save`, `:stats`, `:quit`.

## Performance (measured on this machine)

langchainjs (3,572 files / 52,730 blocks / 56 MB cache):

| Operation | Time |
|---|---|
| Full index build | 162 ms (parallel) |
| Cache load | ~25 ms |
| Search `chatModel` | 0.45 ms |
| Search `async function` (273 hits) | 0.61 ms |
| Search `the` (2-byte fallback, full scan) | 3.4 ms |
| No-match query (trigram absent) | 0.39 ms |
| Incremental reindex of 1 changed file | ~1 ms |

## Architecture

```
src/
├── main.rs    CLI + interactive watch/REPL loop + hit rendering
├── index.rs   trigram index: block parsing, incremental add/remove, parallel build
├── search.rs  posting-list intersection + parallel verification, ranking
├── store.rs   atomic bincode cache (magic header, tmp+rename)
└── watch.rs   debounced recursive filesystem watcher
```

Search ranking: blocks with more occurrences of the query first, then shorter (more focused) blocks.
