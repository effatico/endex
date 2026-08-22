# Using endex with AI coding assistants

endex speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdio: any MCP-capable assistant can use it. The pi extension additionally wraps it natively.

The server is **always-on and low-latency**: a background watcher reindexes changed files (~1 ms), a background embedder keeps semantic vectors warm without blocking queries, and `endex_ask` only ever embeds the query (never the corpus inline). It reports `coverage` (0..1, the fraction of blocks with vectors) so you can tell when it's still warming up.

## MCP tools

| Tool | What it returns |
|---|---|
| `endex_search` | Ranked lexical blocks matching a substring (+ file, line, full block text) |
| `endex_ask` | Hybrid semantic hits (lexical + embeddings) for natural-language questions |
| `endex_graph` | Symbol neighborhood: kind, file:line, callers, callees, importers |
| `endex_flow` | Call paths A→B with file:line hops plus the source block of every hop |
| `endex_clues` | Blocks mentioning a term, annotated with the symbols defined in them |
| `endex_index` | Build/refresh the index (rarely needed — the server self-refreshes) |
| `endex_stats` | Index stats: files / blocks / symbols / edges / vectors / coverage |

Tools always operate on the directory the server was started with — one server per repo (start additional `endex mcp` instances for other directories).

## Claude Code

```bash
# project-local scope (default)
claude mcp add endex -- /path/to/endex mcp /path/to/your/repo

# user scope — available in every project; omit DIR to index whatever
# directory Claude Code starts the server in:
claude mcp add endex -s user \
  -e EMBED_PROVIDER=openai \
  -e EMBED_URL=http://localhost:11434/v1 \
  -e EMBED_MODEL=qwen3-embedding \
  -- /path/to/endex mcp
```

Scopes: `local` (this project only), `project` (`.mcp.json`, committed), `user` (`~/.claude.json`, all projects).

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

## OpenCode

OpenCode reads MCP servers from `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "endex": {
      "type": "local",
      "command": ["/path/to/endex", "mcp", "/path/to/your/repo"],
      "environment": {
        "EMBED_PROVIDER": "openai",
        "EMBED_URL": "http://localhost:11434/v1",
        "EMBED_MODEL": "qwen3-embedding"
      },
      "enabled": true
    }
  }
}
```

Restart OpenCode after editing; the `endex_*` tools appear in the agent's tool list.

## pi

Zero-config: the repo ships a native pi extension that spawns and manages the `endex mcp` server for you and registers the seven `endex_*` tools directly (no `.mcp.json` needed).

```bash
pi install git:github.com/effatico/endex
```

Requirements: the `endex` binary on PATH (or set `ENDEX_BIN=/path/to/endex`). Embedding configuration comes from the environment pi runs in (`EMBED_PROVIDER`, `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY` / `COHERE_API_KEY`).

Handy commands inside pi:

- `/endex stats` — files / blocks / symbols / vectors / semantic coverage
- `/endex restart` — kill the server (it respawns on the next tool call)

The extension auto-starts the server on the first tool call and stops it when the pi session ends. Development / quick test without installing:

```bash
pi -e ./extensions/index.ts
```

## Embedding providers

Semantic search needs a provider; without one, `endex_ask` degrades to lexical results with a warning. See [embeddings.md](embeddings.md) for the full matrix (hash / OpenAI-compatible / Cohere / Ollama / LiteLLM) and configuration.
