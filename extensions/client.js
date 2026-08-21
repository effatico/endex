// endex MCP stdio client — dependency-free, newline-delimited JSON-RPC.
// Used by the pi extension (extensions/index.ts) and directly testable
// with plain node (see extensions/test-client.mjs).
import { spawn } from "node:child_process";

const DEFAULT_TIMEOUT_MS = 120_000;
const STDERR_TAIL = 20;

export class EndexMcpClient {
  /**
   * @param {object} opts
   * @param {string} opts.bin        endex binary (default "endex" on PATH)
   * @param {string} opts.root       directory the server indexes
   * @param {number} [opts.timeoutMs] per-request timeout
   * @param {(line: string) => void} [opts.onLog] server stderr lines
   */
  constructor({ bin = "endex", root, timeoutMs, onLog } = {}) {
    if (!root) throw new Error("EndexMcpClient: root is required");
    this.bin = bin;
    this.root = root;
    const envTimeout = Number(process.env.ENDEX_TIMEOUT_MS);
    this.timeoutMs = timeoutMs ?? (Number.isFinite(envTimeout) ? envTimeout : DEFAULT_TIMEOUT_MS);
    this.onLog = onLog ?? (() => {});

    this.child = undefined;
    this.readyPromise = undefined;
    this.buf = "";
    this.nextId = 1;
    this.pending = new Map(); // id -> {resolve, reject, timer}
    this.stderrTail = [];
  }

  /** Spawn the server (once) and complete the MCP handshake. */
  ensureReady() {
    if (!this.readyPromise) {
      this.readyPromise = this.#start().catch((err) => {
        // Allow retry on next call.
        this.readyPromise = undefined;
        throw err;
      });
    }
    return this.readyPromise;
  }

  get running() {
    return !!this.child && this.child.exitCode === null;
  }

  async #start() {
    const child = spawn(this.bin, ["mcp", this.root], {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env, // EMBED_* etc. flow through
    });
    this.child = child;

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => this.#onData(chunk));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      for (const line of String(chunk).split("\n")) {
        const t = line.trim();
        if (!t) continue;
        this.stderrTail.push(t);
        if (this.stderrTail.length > STDERR_TAIL) this.stderrTail.shift();
        this.onLog(t);
      }
    });
    child.on("error", (err) => this.#onExit(err));
    child.on("exit", (code) =>
      this.#onExit(code === 0 ? undefined : new Error(`endex exited with code ${code}`)),
    );

    // MCP handshake.
    await this.request("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "pi-endex-extension", version: "0.1.0" },
    });
    this.notify("notifications/initialized");
  }

  #onData(chunk) {
    this.buf += chunk;
    for (;;) {
      const nl = this.buf.indexOf("\n");
      if (nl < 0) return;
      const line = this.buf.slice(0, nl).trim();
      this.buf = this.buf.slice(nl + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // not JSON — ignore
      }
      const id = msg.id;
      if (id === undefined || id === null) continue; // notification
      const p = this.pending.get(id);
      if (!p) continue;
      this.pending.delete(id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(msg.error.message ?? "MCP error"));
      else p.resolve(msg.result);
    }
  }

  #onExit(err) {
    const e =
      err ??
      new Error(
        `endex server stopped.${this.stderrTail.length ? "\n" + this.stderrTail.join("\n") : ""}`,
      );
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(e);
    }
    this.pending.clear();
    this.child = undefined;
    this.readyPromise = undefined;
  }

  /** JSON-RPC request with response. */
  request(method, params) {
    return new Promise((resolve, reject) => {
      if (!this.running) {
        reject(
          new Error(
            `endex server not running (tried to spawn "${this.bin}"). ` +
              `Install: curl -fsSL https://raw.githubusercontent.com/effatico/endex/main/install.sh | sh`,
          ),
        );
        return;
      }
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`endex ${method} timed out after ${this.timeoutMs}ms`));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  /** JSON-RPC notification (no response). */
  notify(method, params) {
    if (this.running) {
      this.child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
    }
  }

  /** Call an endex tool and return its text payload. */
  async callTool(name, args = {}) {
    await this.ensureReady();
    const result = await this.request("tools/call", { name, arguments: args });
    const text = result?.content?.map((c) => c.text ?? "").join("\n") ?? "";
    if (result?.isError) throw new Error(text || `endex tool ${name} failed`);
    return text;
  }

  /** Stop the server. Idempotent. */
  dispose() {
    if (this.child) {
      try {
        this.child.kill();
      } catch {
        /* already gone */
      }
    }
    this.#onExit();
  }
}
