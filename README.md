# endex

An extremely fast, cached, self-updating **code indexer** with **millisecond substring search**, a **knowledge graph** for tracing flows, and **hybrid semantic search** — written in Rust.

## How it works

Three layers, all cached on disk and updated incrementally:

1. **Trigram inverted index** (the approach behind Google Code Search / Zoekt): every 3-byte window of the corpus maps to a posting list of *block ids*. A query intersects the posting lists of its trigrams, producing a tiny candidate set that is verified exactly — arbitrary substring search runs in **microseconds to a few milliseconds** even on huge corpora.
2. **Knowledge graph**: symbols (functions, methods, classes, structs, traits, interfaces...) and **call edges** between them, plus file-level **import edges**. Extracted heuristically per language (Rust, TS/JS, Python, Go, Java/C#/Kotlin) — a full graph rebuild is an in-memory pass of a few ms, so it refreshes live while you watch. Powers `graph` (who calls whom), `flow` (call paths between two symbols) and `clues` (blocks mentioning a term + the symbols defined there).
3. **Embeddings**: every block gets a semantic vector, fused with the lexical ranking via reciprocal rank fusion (`ask`). Vectors are keyed by *content hash*, so editing a file only re-embeds changed blocks and moved code keeps its vectors.

Code is chunked into blank-line-separated **blocks** (max 80 lines), so results are meaningful text blocks (functions, paragraphs, config sections). Everything is case-insensitive, gitignore-aware, skips binaries and files > 5 MB, and indexes in parallel via `rayon`.

### Embedding providers (local or remote)

| Provider | Flag | Notes |
|---|---|---|
| `hash` (default) | `--embed-provider hash` | Deterministic feature-hashing embedding. Fully offline, instant, zero deps. Fuzzy lexical matching (typos, word variants) — not truly semantic. |
| `openai` | `--embed-provider openai` | Any OpenAI-compatible `/embeddings` endpoint: OpenAI, **Ollama** (`--embed-url http://localhost:11434/v1`), LM Studio, vLLM, ... This is where real semantic search comes from — local *or* remote. |

```bash
# remote (OpenAI)
endex ask ~/my-repo "how do we handle retries" --embed-provider openai

# local (Ollama running on your machine)
endex ask ~/my-repo "how do we handle retries" \
  --embed-provider openai \
  --embed-url http://localhost:11434/v1 \
  --embed-model nomic-embed-text
```

Env-var equivalents: `EMBED_PROVIDER`, `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY` / `OPENAI_API_KEY`, `EMBED_DIM`, `EMBED_BATCH`. If the provider is unreachable, `ask` falls back to lexical search with a warning.

## Usage

```bash
cargo build --release

# Build or refresh the cache (incremental if a cache exists)
endex index ~/my-repo

# Fast lexical substring search (auto-loads cache, auto-refreshes stale files)
endex search ~/my-repo "createUserAccount" --limit 20

# Hybrid semantic search (lexical + embeddings)
endex ask ~/my-repo "how do we handle retries"

# Knowledge graph: symbol neighborhood (callers / callees / importers)
endex graph ~/my-repo "chargeCustomer"

# Flows: call-graph paths between two symbols
endex flow ~/my-repo bootstrap listenForConnections

# Clues: blocks mentioning a term + the symbols defined in them
endex clues ~/my-repo "rate limiting"

# Watch mode: live updates + interactive REPL
endex watch ~/my-repo
```

REPL (watch mode): plain `QUERY` = lexical, `? QUERY` = hybrid semantic, plus `:graph N`, `:flow A B`, `:clues T`, `:embed [provider]`, `:limit N`, `:save`, `:stats`, `:quit`. File changes are reindexed in ~1 ms/file; the graph rebuilds in memory and embeddings update lazily on the next `?` query.

## Performance (measured on this machine)

langchainjs (3,572 files / 52,730 blocks / 6,275 symbols / 42k call edges):

| Operation | Time |
|---|---|
| Full index build (incl. graph) | 463 ms (parallel) |
| Knowledge graph rebuild | 21 ms |
| Cache load | ~25 ms |
| Lexical search `chatModel` | 0.45 ms |
| Hybrid semantic search (warm, 42k vectors) | 9 ms |
| Incremental reindex of 1 changed file | ~1 ms |
| Graph rebuild after a file edit | in-memory, ~µs–ms |

## Architecture

```
src/
├── main.rs    CLI + interactive watch/REPL loop + hit rendering
├── index.rs   trigram index: block parsing, incremental add/remove, parallel build
├── graph.rs   knowledge graph: symbol/def extraction, call edges, import resolution, path queries
├── embed.rs   embedding providers (hash / OpenAI-compatible HTTP), content-hash vector cache, RRF fusion
├── search.rs  posting-list intersection + parallel verification, ranking
├── store.rs   atomic bincode cache (versioned magic header, tmp+rename)
└── watch.rs   debounced recursive filesystem watcher
```

Search ranking: lexical hits by occurrence count (then block size); hybrid hits by reciprocal-rank fusion of both rankings.

## Development

```bash
cargo test --release        # 17 integration tests (index, graph, embeddings, cache)
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

MIT licensed. CI runs build + fmt + clippy + tests on Ubuntu and macOS.
