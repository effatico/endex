import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { EndexMcpClient } from "./client.js";

/**
 * pi extension for endex — the fast code indexer with knowledge graph and
 * hybrid semantic search. Spawns `endex mcp <cwd>` lazily on first tool call
 * and registers the index as native pi tools.
 *
 * Requirements: the `endex` binary on PATH (or ENDEX_BIN).
 *   curl -fsSL https://raw.githubusercontent.com/effatico/endex/main/install.sh | sh
 */

const Limit = Type.Optional(Type.Integer({ description: "Max results (default 20, max 100)." }));

const client = new EndexMcpClient({
  bin: process.env.ENDEX_BIN ?? "endex",
  root: process.cwd(),
});

function textResult(text: string, details: Record<string, unknown> = {}) {
  return { content: [{ type: "text" as const, text }], details };
}

export default function (pi: ExtensionAPI) {
  // Lazily start the MCP server with the session; stop it on shutdown.
  pi.on("session_shutdown", async () => client.dispose());

  pi.registerTool({
    name: "endex_search",
    label: "endex search",
    description:
      "Fast substring search over the whole codebase, ranked by relevance. PREFER THIS over grep when looking for where an identifier, function name, error message, or literal string is used: results are pre-ranked, deduplicated per block, and each hit INCLUDES the full code block text (typically the whole function), so you usually do NOT need a follow-up read of the file. Sub-millisecond even on large repos.",
    promptSnippet: "Ranked substring search over code blocks, full block text included",
    promptGuidelines: [
      "Use endex_search instead of grep/glob when looking for where an identifier, function name, error message, or literal string is used — it returns ranked code blocks with full text.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Case-insensitive substring to find." }),
      limit: Limit,
    }),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_search", params));
    },
  });

  pi.registerTool({
    name: "endex_ask",
    label: "endex ask",
    description:
      "Semantic search over the codebase in NATURAL LANGUAGE — use this whenever you do not know the exact identifier: 'how do we handle retries', 'where is rate limiting enforced', 'authentication middleware'. STRONGLY PREFER this as the FIRST step when exploring an unfamiliar codebase or concept, instead of guessing grep patterns. Each hit includes the full code block text. Results are reranked for relevance when the provider supports it (check endex_stats).",
    promptSnippet: "Natural-language semantic search over the codebase, Cohere-reranked",
    promptGuidelines: [
      "Use endex_ask first when exploring an unfamiliar codebase or concept in natural language, instead of guessing grep patterns.",
      "endex_ask hits include the symbols defined in each block — use those names directly with endex_graph/endex_flow for follow-up navigation.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Natural language question or description of the code." }),
      limit: Limit,
    }),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_ask", params));
    },
  });

  pi.registerTool({
    name: "endex_graph",
    label: "endex graph",
    description:
      "Knowledge-graph neighborhood of a symbol: what it calls (callees), who calls it (callers), and which files import it, with file:line for every entry. USE THIS before editing a shared function to see all its call sites and dependents — it answers 'what breaks if I change X?' far better than grep, because edges are real call/import relationships, not text matches.",
    promptSnippet: "Symbol neighborhood: callers, callees, importers",
    promptGuidelines: [
      "Use endex_graph before editing a shared symbol to see its callers/callees and judge blast radius.",
    ],
    parameters: Type.Object({
      symbol: Type.String({ description: "Symbol name (function, method, class, struct, ...)." }),
    }),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_graph", params));
    },
  });

  pi.registerTool({
    name: "endex_flow",
    label: "endex flow",
    description:
      "Trace EXECUTION FLOWS through the codebase: finds call-graph paths between two symbols (e.g. from 'main' to 'save', from an HTTP handler to the DB write). This is the PRIMARY tool for questions like 'how does X reach Y?', 'what is the code path for feature Z?', 'trace the request lifecycle'. Returns up to 5 shortest paths; every hop has file:line AND the full source block inline (disable with include_blocks=false), so a single call often answers a flow question completely. Tip: use endex_search or endex_ask first to discover the endpoint symbol names if unsure.",
    promptSnippet: "Call-graph paths between two symbols, with source blocks inline",
    promptGuidelines: [
      "Use endex_flow to trace how execution gets from one symbol to another ('how does X reach Y?') instead of manually hopping through files.",
    ],
    parameters: Type.Object({
      from: Type.String({ description: "Source symbol name — an entry point like main, a handler, an exported API." }),
      to: Type.String({ description: "Target symbol name — the downstream function to trace into." }),
      include_blocks: Type.Optional(
        Type.Boolean({ description: "Include source text of each hop's block (default true)." }),
      ),
      max_depth: Type.Optional(Type.Integer({ description: "Max path length (default 8)." })),
    }),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_flow", params));
    },
  });

  pi.registerTool({
    name: "endex_clues",
    label: "endex clues",
    description:
      "Reconnaissance by concept: blocks mentioning a term, each annotated with the symbols DEFINED in that block plus their callers/callees. Use when you have a topic word ('cache', 'auth', 'retry') and want both the matching code AND the key symbols involved — the returned symbol names are ideal follow-up inputs for endex_graph and endex_flow.",
    promptSnippet: "Blocks mentioning a term, annotated with defined symbols",
    parameters: Type.Object({
      term: Type.String({ description: "Concept word to investigate." }),
      limit: Limit,
    }),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_clues", params));
    },
  });

  pi.registerTool({
    name: "endex_index",
    label: "endex index",
    description:
      "Build or refresh the endex code index for a directory. Rarely needed: the endex server watches the filesystem and self-refreshes. Call this only to force a refresh or to warm a fresh checkout.",
    promptSnippet: "Build/refresh the code index (usually automatic)",
    parameters: Type.Object({}),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_index", params));
    },
  });

  pi.registerTool({
    name: "endex_stats",
    label: "endex stats",
    description:
      "Check the endex server/index state: files, blocks, symbols, embedding provider + coverage, cache version/age, last index/embed/save timestamps. Call this to verify setup before relying on endex_ask (coverage < 1 means the background embedder is still warming).",
    promptSnippet: "Server and index stats + semantic coverage",
    parameters: Type.Object({}),
    async execute(_id, params) {
      return textResult(await client.callTool("endex_stats", params));
    },
  });

  pi.registerCommand("endex", {
    description: "endex indexer: /endex stats | /endex restart",
    handler: async (args, ctx) => {
      const sub = (args ?? "").trim();
      if (sub === "restart") {
        client.dispose();
        ctx.ui.notify("endex server stopped; it respawns on next tool call", "info");
        return;
      }
      if (sub && sub !== "stats" && sub !== "status") {
        ctx.ui.notify("usage: /endex stats | /endex restart", "info");
        return;
      }
      try {
        const out = await client.callTool("endex_stats", {});
        const raw = JSON.parse(out);
        // Stats are wrapped in the same meta envelope as every other tool.
        const s = raw.meta?.data ?? raw;
        ctx.ui.notify(
          `endex: ${s.index?.files ?? "?"} files · ${s.index?.blocks ?? "?"} blocks · ${s.index?.symbols ?? "?"} symbols · ${s.embeddings?.vectors ?? "?"} vectors (${Math.round((s.embeddings?.coverage ?? 0) * 100)}% semantic · cache v${s.cache?.version ?? "?"})`,
          "info",
        );
      } catch (e) {
        ctx.ui.notify(`endex unavailable: ${(e as Error).message}`, "error");
      }
    },
  });
}
