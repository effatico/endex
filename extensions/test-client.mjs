// Quick standalone test of the MCP client against the endex binary:
//   ENDEX_BIN=./target/release/endex node extensions/test-client.mjs
import { EndexMcpClient } from "./client.js";

const client = new EndexMcpClient({
  bin: process.env.ENDEX_BIN ?? "endex",
  root: process.cwd(),
  onLog: (l) => console.error("  [server]", l),
});

// Tool payloads arrive wrapped in the meta envelope: { meta: { data, ... } }.
const unwrap = async (name, args) => {
  const raw = JSON.parse(await client.callTool(name, args));
  return raw.meta?.data ?? raw;
};

const stats = await unwrap("endex_stats", {});
console.log("stats:", stats.index.files, "files,", stats.index.symbols, "symbols");

const flow = await unwrap("endex_flow", { from: "main", to: "save", include_blocks: false });
console.log("flow main->save:", flow.path_count, "path(s)");

const search = await unwrap("endex_search", { query: "trigram", limit: 2 });
console.log("search 'trigram':", search.count, "hits, first:", search.hits[0]?.file);

client.dispose();
console.log("client OK");
