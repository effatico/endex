# Embedding providers

`endex ask` (and `endex_ask` over MCP) fuses lexical search with embedding similarity via reciprocal rank fusion. The provider is selected by flag or env var; if it's unreachable, `ask` falls back to lexical search with a warning.

| Provider | Flag | Notes |
|---|---|---|
| `hash` (default) | `--embed-provider hash` | Deterministic feature-hashing embedding. Fully offline, instant, zero deps. Fuzzy lexical matching (typos, word variants) — not truly semantic. |
| `openai` | `--embed-provider openai` | Any OpenAI-compatible `/embeddings` endpoint: OpenAI, **Ollama** (`--embed-url http://localhost:11434/v1`), **LiteLLM proxy** (`--embed-url http://localhost:4000/v1`), LM Studio, vLLM, ... |
| `cohere` | `--embed-provider cohere` | Cohere `/embed` API (`embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`). Blocks embed as `search_document`, queries as `search_query` for best retrieval quality. Key via `--embed-key`, `EMBED_API_KEY`, or `COHERE_API_KEY`. |

Env-var equivalents: `EMBED_PROVIDER`, `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY` / `OPENAI_API_KEY` / `COHERE_API_KEY`, `EMBED_DIM` (hash only), `EMBED_BATCH`.

## Examples

```bash
# remote (OpenAI)
endex ask ~/my-repo "how do we handle retries" --embed-provider openai

# remote (Cohere)
endex ask ~/my-repo "how do we handle retries" \
  --embed-provider cohere --embed-model embed-v4.0 --embed-key $COHERE_API_KEY

# local (Ollama)
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

## LiteLLM proxy (gateway to any backend)

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

Switching `EMBED_MODEL` between LiteLLM aliases is safe: endex's manifest detects the new model identity and re-embeds in the background automatically. Caveat: Cohere's `search_document`/`search_query` asymmetry is only available when connecting to Cohere *directly* (`EMBED_PROVIDER=cohere`), not through the proxy — a minor quality nuance, irrelevant for Ollama/OpenAI models.

## Caching behavior

Vectors are keyed by content hash (fnv64 of block text) and persisted in `.endex-index.bin`; `.endex-manifest.json` records the provider identity and corpus fingerprint. Consequences:

- Restarting the server reuses all vectors (zero re-embedding).
- Editing a file only re-embeds the blocks that changed; moved code keeps its vectors.
- Switching provider or model invalidates the vector space — detected via the manifest, rebuilt in the background.
