// Quick standalone test of the MCP client against the endex binary:
//   ENDEX_BIN=./target/release/endex node extensions/test-client.mjs
import { EndexMcpClient } from "./client.js";

const client = new EndexMcpClient({
  bin: process.env.ENDEX_BIN ?? "endex",
  root: process.cwd(),
  onLog: (l) => console.error("  [server]", l),
});

const status = JSON.parse(await client.callTool("endex_status", {}));
console.log("status:", status.files, "files,", status.symbols, "symbols");

const flow = JSON.parse(
  await client.callTool("endex_flow", { from: "main", to: "save", include_blocks: false }),
);
console.log("flow main->save:", flow.path_count, "path(s)");

const search = JSON.parse(await client.callTool("endex_search", { query: "trigram", limit: 2 }));
console.log("search 'trigram':", search.count, "hits, first:", search.hits[0]?.file);

client.dispose();
console.log("client OK");
