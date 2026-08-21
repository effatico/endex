# endex

An extremely fast, cached, self-updating **code indexer** with **millisecond substring search**, a **knowledge graph** for tracing flows, and **hybrid semantic search** — written in Rust.

## How it works

Three layers, all cached on disk and updated incrementally:

1. **Trigram inverted index** (the approach behind Google Code Search / Zoekt): every 3-byte window of the corpus maps to a posting list of *block ids*. A query intersects the posting lists of its trigrams, producing a tiny candidate set that is verified exactly — arbitrary substring search runs in **microseconds to a few milliseconds** even on huge corpora.
2. **Knowledge graph**: symbols (functions, methods, classes, structs, traits, interfaces...) and **call edges** between them, plus file-level **import edges**. Extracted heuristically per language (Rust, TS/JS, Python, Go, Java/C#/Kotlin) — a full graph rebuild is an in-memory pass of a few ms, so it refreshes live while you watch. Powers `graph` (who calls whom), `flow` (call paths between two symbols) and `clues` (blocks mentioning a term + the symbols defined there).
3. **Embeddings**: every block gets a semantic vector, fused with the lexical ranking via reciprocal rank fusion (`ask`). Vectors are keyed by *content hash* (fnv64 of block text), persisted in `.endex-index.bin`, and guarded by a `.endex-manifest.json` (provider id + corpus fingerprint). Editing a file only re-embeds its changed blocks, moved code keeps its vectors, and a cache written for a different embedding model is detected and ignored. Each source file also carries a `content_hash`, so the server can tell exactly which files invalidated and update just those.

Code is chunked into blank-line-separated **blocks** (max 80 lines), so results are meaningful text blocks (functions, paragraphs, config sections). Everything is case-insensitive, gitignore-aware, skips binaries and files > 5 MB, and indexes in parallel via `rayon`.

### Embedding providers (local or remote)

| Provider | Flag | Notes |
|---|---|---|
| `hash` (default) | `--embed-provider hash` | Deterministic feature-hashing embedding. Fully offline, instant, zero deps. Fuzzy lexical matching (typos, word variants) — not truly semantic. |
| `openai` | `--embed-provider openai` | Any OpenAI-compatible `/embeddings` endpoint: OpenAI, **Ollama** (`--embed-url http://localhost:11434/v1`), **LiteLLM proxy** (`--embed-url http://localhost:4000/v1`), LM Studio, vLLM, ... This is where real semantic search comes from — local *or* remote. |
| `cohere` | `--embed-provider cohere` | Cohere `/embed` API (`embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`). Blocks embed as `search_document`, queries as `search_query` for best retrieval quality. Key via `--embed-key`, `EMBED_API_KEY`, or `COHERE_API_KEY`. |

```bash
# remote (OpenAI)
endex ask ~/my-repo "how do we handle retries" --embed-provider openai

# remote (Cohere)
endex ask ~/my-repo "how do we handle retries" \
  --embed-provider cohere \
  --embed-model embed-v4.0 \
  --embed-key $COHERE_API_KEY

# local (Ollama running on your machine)
endex ask ~/my-repo "how do we handle retries" \
  --embed-provider openai \
  --embed-url http://localhost:11434/v1 \
  --embed-model nomic-embed-text

# via a LiteLLM proxy (gateway to OpenAI / Cohere / Ollama / Bedrock / ...)
endex ask ~/my-repo "how do we handle retries" \
  --embed-provider openai \
  --embed-url http://localhost:4000/v1 \
  --embed-model qwen3-embedding \
  --embed-key sk-litellm-local
```

Env-var equivalents: `EMBED_PROVIDER`, `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY` / `OPENAI_API_KEY` / `COHERE_API_KEY`, `EMBED_DIM`, `EMBED_BATCH`. If the provider is unreachable, `ask` falls back to lexical search with a warning.

### LiteLLM proxy (gateway to any embedding backend)

A [LiteLLM](https://github.com/BerriAI/litellm) proxy exposes an OpenAI-compatible `/v1/embeddings` endpoint, so the `openai` provider works with it unchanged — one gateway for OpenAI, Cohere, Ollama, Bedrock, Azure, Voyage, ... with centralized auth, budgets and fallbacks.

```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen3-embedding            # <- the alias endex uses in EMBED_MODEL
    litellm_params:
      model: ollama/qwen3-embedding
      api_base: http://localhost:11434
  - model_name: cohere-embed
    litellm_params:
      model: cohere/embed-v4.0
      api_key: os.environ/COHERE_API_KEY
general_settings:
  master_key: sk-litellm-local
```

```bash
litellm --config litellm_config.yaml --port 4000
# then: EMBED_PROVIDER=openai EMBED_URL=http://localhost:4000/v1 \
#         EMBED_MODEL=qwen3-embedding EMBED_API_KEY=sk-litellm-local
```

Switching `EMBED_MODEL` between LiteLLM aliases is safe: endex's manifest detects the new model identity and re-embeds in the background automatically. One caveat: Cohere's `search_document`/`search_query` asymmetry is only available when connecting to Cohere *directly* (`EMBED_PROVIDER=cohere`), not through the proxy — a minor quality nuance, irrelevant for Ollama/OpenAI models.

## Install

**Prebuilt binaries (no Rust toolchain needed)** — grab one from [GitHub Releases](https://github.com/effatico/endex/releases), or:

```bash
# macOS / Linux — detects arch, verifies checksum, installs to /usr/local/bin
curl -fsSL https://raw.githubusercontent.com/effatico/endex/main/install.sh | sh

# Homebrew (macOS/Linux)
brew install effatico/endex/endex

# Windows: download endex-x86_64-pc-windows-msvc.zip from Releases
```

**From source** (requires Rust):

```bash
cargo install --git https://github.com/effatico/endex   # or: cargo build --release
```

### Cutting a release (maintainers)

```bash
git tag v0.2.0 && git push --tags
```

The `Release` workflow builds binaries for Linux (x86_64/aarch64), macOS (Intel/Apple Silicon) and Windows, then attaches them with `.sha256` checksums to a new GitHub Release. After it runs, update the `sha256` values in `Formula/endex.rb`.

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

# MCP server: expose the index to Claude Code / Cursor over stdio
endex mcp ~/my-repo
```

## MCP server (AI assistants)

`endex mcp [DIR]` speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC, with `Content-Length` framing auto-detected). Claude Code, Cursor and other MCP clients get seven tools:

| Tool | What it returns |
|---|---|
| `endex_index` | Build/refresh the index (incremental) — rarely needed, the server self-refreshes |
| `endex_search` | Lexical blocks matching a substring (+ file, line, block text) |
| `endex_ask` | Hybrid semantic hits (uses the embedding provider env) |
| `endex_graph` | Symbol neighborhood: kind, file:line, callers, callees, importers |
| `endex_flow` | Call paths A→B with file:line hops **plus the source block of every hop** (`include_blocks`, default true) — perfect for "how does X reach Y" questions |
| `endex_clues` | Blocks mentioning a term, annotated with the symbols defined in them |
| `endex_status` | Index stats (files / blocks / symbols / edges / vectors / semantic coverage) |

Every tool accepts an optional `dir` argument; without it, the directory the server was started with is used.

**Always-on, low-latency by design:** the server holds the index in memory and runs two background threads — a debounced **filesystem watcher** (reindexes changed files in ~1 ms, rebuilds the graph, saves the cache) and a **background embedder** (keeps semantic vectors warm without ever blocking a query). `endex_ask` embeds only the query — never the corpus inline — so it stays fast even while a cold corpus is still embedding; it reports `semantic_coverage` (and `warming_up`) so partial-semantic results are transparent. If the provider is unreachable it degrades to lexical hits with a warning.

**Caching:** vectors are content-hash-keyed and persisted in `.endex-index.bin`; a `.endex-manifest.json` records the provider identity + corpus fingerprint. Restarts reuse all vectors (zero re-embedding), file edits only re-embed the blocks that changed, and a cache written for a different model is detected and rebuilt automatically.

### Claude Code setup

```bash
claude mcp add endex -- /path/to/endex mcp /path/to/your/repo

# with semantic search against local Ollama:
claude mcp add endex \
  -e EMBED_PROVIDER=openai \
  -e EMBED_URL=http://localhost:11434/v1 \
  -e EMBED_MODEL=qwen3-embedding \
  -- /path/to/endex mcp /path/to/your/repo

# ... against Cohere:
claude mcp add endex \
  -e EMBED_PROVIDER=cohere \
  -e EMBED_MODEL=embed-v4.0 \
  -e EMBED_API_KEY=$COHERE_API_KEY \
  -- /path/to/endex mcp /path/to/your/repo

# ... through a LiteLLM proxy:
claude mcp add endex \
  -e EMBED_PROVIDER=openai \
  -e EMBED_URL=http://localhost:4000/v1 \
  -e EMBED_MODEL=qwen3-embedding \
  -e EMBED_API_KEY=sk-litellm-local \
  -- /path/to/endex mcp /path/to/your/repo
```

**Scopes** (`-s, --scope`): `local` (default — this project only), `project` (`.mcp.json`, committed and shared with the team), `user` (all your projects, in `~/.claude.json`). For user scope, omit the dir argument — the server then indexes whatever project directory Claude Code starts it in:

```bash
claude mcp add endex -s user \
  -e EMBED_PROVIDER=openai -e EMBED_URL=http://localhost:11434/v1 \
  -e EMBED_MODEL=qwen3-embedding \
  -- /path/to/endex mcp
```

Or manually in `.mcp.json` / `~/.claude.json`:

```json
{
  "mcpServers": {
    "endex": {
      "command": "/path/to/endex",
      "args": ["mcp", "/path/to/your/repo"],
      "env": {
        "EMBED_PROVIDER": "openai",
        "EMBED_URL": "http://localhost:11434/v1",
        "EMBED_MODEL": "qwen3-embedding"
      }
    }
  }
}
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
├── embed.rs   embedding providers (hash / OpenAI-compatible / Cohere), content-hash vector cache, RRF fusion
├── search.rs  posting-list intersection + parallel verification, ranking
├── store.rs   atomic bincode cache (versioned magic header, tmp+rename) + JSON manifest (provider id, corpus fingerprint)
├── mcp.rs     MCP stdio server: JSON-RPC loop + tool handlers (initialize / tools/list / tools/call)
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
