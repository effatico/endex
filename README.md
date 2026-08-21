# endex

An extremely fast, cached, self-updating **code indexer** with **millisecond substring search**, a **knowledge graph** for tracing flows, and **hybrid semantic search** — written in Rust.

Three layers, all cached on disk and updated incrementally:

1. **Trigram inverted index** (the approach behind Google Code Search / Zoekt): every 3-byte window of the corpus maps to a posting list of *block ids*. Queries intersect posting lists and verify a tiny candidate set — arbitrary substring search runs in **microseconds to a few milliseconds** even on huge corpora.
2. **Knowledge graph**: symbols (functions, methods, classes, ...) and **call edges**, plus file-level **import edges**, extracted heuristically per language (Rust, TS/JS, Python, Go, Java/C#/Kotlin). Powers `graph` (who calls whom), `flow` (call paths between two symbols) and `clues` (blocks mentioning a term + their symbols).
3. **Embeddings**: every block gets a semantic vector, fused with lexical ranking via reciprocal rank fusion (`ask`). Vectors are keyed by content hash and persisted in `.endex-index.bin`, so edits only re-embed changed blocks and restarts are instant.

Code is chunked into blank-line-separated **blocks** (max 80 lines), so results are meaningful units (functions, paragraphs, config sections). Case-insensitive, gitignore-aware, skips binaries and files > 5 MB, parallel via `rayon`.

## Install

**Prebuilt binary (no Rust needed)** — from [Releases](https://github.com/effatico/endex/releases), or:

```bash
# macOS / Linux — detects arch, verifies checksum, installs to /usr/local/bin
curl -fsSL https://raw.githubusercontent.com/effatico/endex/main/install.sh | sh

# Homebrew
brew install effatico/endex/endex

# Windows: download endex-x86_64-pc-windows-msvc.zip from Releases
```

**From source:** `cargo install --git https://github.com/effatico/endex` (or `cargo build --release`).

## Use with your AI assistant

endex runs as an MCP server (`endex mcp [DIR]`) exposing seven tools — search, ask, graph, flow, clues, index, status — with an always-on watcher and background embedder.

- **pi** (zero-config native extension — spawns and manages the server for you):
  ```bash
  pi install git:github.com/effatico/endex
  ```
- **Claude Code**:
  ```bash
  claude mcp add endex -- /path/to/endex mcp /path/to/repo
  ```
- **OpenCode** (in `opencode.json`):
  ```json
  { "mcp": { "endex": { "type": "local", "command": ["/path/to/endex", "mcp", "/path/to/repo"] } } }
  ```

Details, scopes, env config, and per-assistant options: [docs/assistants.md](docs/assistants.md).
Semantic search providers (Ollama, OpenAI, Cohere, LiteLLM, offline hash): [docs/embeddings.md](docs/embeddings.md).

## CLI usage

```bash
endex index ~/my-repo                          # build or refresh the cache (incremental)
endex search ~/my-repo "createUserAccount"     # lexical substring search
endex ask ~/my-repo "how do we handle retries" # hybrid semantic search
endex graph ~/my-repo "chargeCustomer"         # callers / callees / importers
endex flow ~/my-repo bootstrap listen          # call paths between two symbols
endex clues ~/my-repo "rate limiting"          # blocks + the symbols defined in them
endex watch ~/my-repo                          # live updates + interactive REPL
endex mcp ~/my-repo                            # MCP server over stdio
```

Watch-mode REPL: plain `QUERY` = lexical, `? QUERY` = semantic, plus `:graph N`, `:flow A B`, `:clues T`, `:embed`, `:limit N`, `:save`, `:stats`, `:quit`.

Semantic search config via env: `EMBED_PROVIDER` (`hash` default, `openai`, `cohere`), `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`.

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

## Architecture

```
src/
├── main.rs    CLI + interactive watch/REPL loop + hit rendering
├── index.rs   trigram index: block parsing, incremental add/remove, parallel build
├── graph.rs   knowledge graph: symbol/def extraction, call edges, import resolution, path queries
├── embed.rs   embedding providers (hash / OpenAI-compatible / Cohere), content-hash vector cache, RRF fusion
├── search.rs  posting-list intersection + parallel verification, ranking
├── store.rs   atomic bincode cache + JSON manifest (provider id, corpus fingerprint)
├── mcp.rs     MCP stdio server: JSON-RPC loop, watcher + background embedder, tool handlers
└── watch.rs   debounced recursive filesystem watcher

extensions/    pi extension: MCP stdio client + native pi tool registrations
```

## Development

```bash
cargo test --release        # integration tests (index, graph, embeddings, cache)
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

# pi extension smoke test (no pi needed):
ENDEX_BIN=./target/release/endex node extensions/test-client.mjs
```

Cutting a release: `git tag v0.2.0 && git push --tags` — the Release workflow builds binaries for Linux (x86_64/aarch64), macOS (Intel/Apple Silicon) and Windows with `.sha256` checksums, then update `Formula/endex.rb`.

MIT licensed. CI runs build + fmt + clippy + tests on Ubuntu and macOS.
