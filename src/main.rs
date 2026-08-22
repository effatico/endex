use endex::{embed, index::Index, mcp, output, search, store, watch};
use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("index") => cmd_index(&args[1..]),
        Some("search") => cmd_search(&args[1..]),
        Some("ask") => cmd_ask(&args[1..]),
        Some("graph") => cmd_graph(&args[1..]),
        Some("flow") => cmd_flow(&args[1..]),
        Some("clues") => cmd_clues(&args[1..]),
        Some("watch") => cmd_watch(&args[1..]),
        Some("mcp") => cmd_mcp(&args[1..]),
        Some("--version") | Some("-V") => println!("endex {}", env!("CARGO_PKG_VERSION")),
        Some("--help") | Some("-h") => usage(0),
        _ => usage(2),
    }
}

fn usage(code: i32) -> ! {
    eprintln!(
        "endex — fast cached code indexer: trigram search + knowledge graph + hybrid semantic search

USAGE:
  endex index  [DIR]              build or refresh the cache for DIR (default: .)
  endex search [DIR] QUERY        fast lexical substring search
  endex ask    [DIR] QUERY        hybrid search: lexical + semantic embeddings
  endex graph  [DIR] SYMBOL       show a symbol's callers / callees / importers
  endex flow   [DIR] FROM TO      find call-graph paths between two symbols
  endex clues  [DIR] TERM         code blocks mentioning TERM + their symbols
  endex watch  [DIR]              watch for changes + interactive REPL
  endex mcp    [DIR]              serve the index as an MCP server over stdio
                                  (Claude Code / Cursor integration)

OPTIONS:
  --limit N            max results (default 50)
  --no-cache           ignore the on-disk cache and rebuild from scratch
  --embed-provider P   cohere (default: Cohere /embed API + /rerank reranking)
                       | openai (any OpenAI-compatible endpoint: OpenAI,
                       Ollama, LM Studio, vLLM, ...) | hash (fully offline)
  --embed-url URL      e.g. http://localhost:11434/v1   (env EMBED_URL)
  --embed-model M      e.g. embed-v4.0, text-embedding-3-small (env EMBED_MODEL)
  --embed-key KEY      API key (env EMBED_API_KEY / COHERE_API_KEY / OPENAI_API_KEY)
  --embed-dim N        hash embedding dimensions (default 256)
  --embed-batch N      remote embedding batch size (default 96 cohere / 64 openai)
  --embed-rerank-model M  Cohere rerank model (default rerank-v3.5,
                       env EMBED_RERANK_MODEL; cohere provider only)

REPL (watch mode):
  QUERY            lexical search          ? QUERY   hybrid semantic search
  :graph NAME      symbol neighborhood     :flow A B call paths
  :clues TERM      blocks + symbols        :embed    build/refresh embeddings
  :limit N  :save  :stats  :quit

The cache is stored under ~/.endex/cache/<repo_name>-<hash> (one 'index.bin'
  plus a 'manifest.json' per project); override the root with ENDEX_CACHE_DIR.
  The watcher and full walks always honor .gitignore; hidden (dot) files and
  the cache files themselves are never indexed."
    );
    std::process::exit(code);
}

// ---------- arg parsing ----------

struct Opts {
    dir: PathBuf,
    terms: Vec<String>,
    limit: usize,
    use_cache: bool,
    embed: embed::ProviderOpts,
}

fn parse_opts(args: &[String]) -> Opts {
    let mut opts = Opts {
        dir: PathBuf::from("."),
        terms: Vec::new(),
        limit: 50,
        use_cache: true,
        embed: embed::ProviderOpts::default(),
    };
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let next = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match args[i].as_str() {
            "--limit" | "-l" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    opts.limit = v;
                }
            }
            "--no-cache" => opts.use_cache = false,
            "--embed-provider" => opts.embed.provider = next(&mut i),
            "--embed-rerank-model" => opts.embed.rerank_model = next(&mut i),
            "--embed-model" => opts.embed.model = next(&mut i),
            "--embed-url" => opts.embed.url = next(&mut i),
            "--embed-key" => opts.embed.key = next(&mut i),
            "--embed-dim" => opts.embed.dim = next(&mut i).and_then(|s| s.parse().ok()),
            "--embed-batch" => opts.embed.batch = next(&mut i).and_then(|s| s.parse().ok()),
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }
    if let Some(dir) = positional.first() {
        opts.dir = PathBuf::from(dir);
    }
    opts.terms = positional.into_iter().skip(1).collect();
    opts
}

fn provider_from(opts: &Opts) -> embed::Provider {
    let (prov, notice) = embed::Provider::resolve_checked(&opts.embed);
    if let Some(n) = notice {
        eprintln!("note: {n}");
    }
    prov
}

// ---------- load-or-build ----------

fn load_or_build(root: &Path, use_cache: bool) -> Index {
    let t0 = Instant::now();
    let mut idx = if use_cache {
        match store::load(root) {
            Some(i) => {
                eprintln!(
                    "  cache loaded: {} files / {} blocks / {} symbols in {:?}",
                    i.file_count(),
                    i.block_count(),
                    i.graph.symbols.len(),
                    t0.elapsed()
                );
                i
            }
            None => {
                eprintln!("  no valid cache found, building full index...");
                let mut i = Index::new(root);
                i.build(root);
                let _ = store::save(&i, root);
                i
            }
        }
    } else {
        eprintln!("  building full index (--no-cache)...");
        let mut i = Index::new(root);
        i.build(root);
        let _ = store::save(&i, root);
        i
    };

    // Cheap incremental refresh (stat-walk + reindex only changed files).
    let t1 = Instant::now();
    let changed = idx.refresh(root);
    if changed > 0 {
        eprintln!(
            "  refresh: {} file(s) changed in {:?}",
            changed,
            t1.elapsed()
        );
        let _ = store::save(&idx, root);
    } else if idx.graph.symbols.is_empty() && !idx.files.is_empty() {
        idx.finalize(); // cache predates the knowledge graph
    }
    idx
}

// ---------- commands ----------

fn cmd_index(args: &[String]) {
    let opts = parse_opts(args);
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    eprintln!("Indexing {} ...", root.display());
    let t0 = Instant::now();
    let idx = load_or_build(&root, opts.use_cache);
    eprintln!(
        "Done: {} files, {} blocks, {} symbols, {} call edges, {} file imports — total {:?}",
        idx.file_count(),
        idx.block_count(),
        idx.graph.symbols.len(),
        idx.graph.call_edge_count(),
        idx.graph.file_imports.len(),
        t0.elapsed()
    );
}

fn cmd_search(args: &[String]) {
    let opts = parse_opts(args);
    let query = opts.terms.join(" ");
    if query.is_empty() {
        eprintln!("error: search requires a QUERY (quote it if it has spaces)");
        std::process::exit(2);
    }
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let idx = load_or_build(&root, opts.use_cache);
    let t = Instant::now();
    let hits = search::search(&idx, &query, opts.limit);
    output::print_hits(&idx, &hits, &query, opts.limit, t.elapsed());
}

fn cmd_ask(args: &[String]) {
    let opts = parse_opts(args);
    let query = opts.terms.join(" ");
    if query.is_empty() {
        eprintln!("error: ask requires a QUERY");
        std::process::exit(2);
    }
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let mut idx = load_or_build(&root, opts.use_cache);
    let prov = provider_from(&opts);
    let t = Instant::now();
    match embed::ask(&mut idx, &prov, &query, opts.limit) {
        Ok(outcome) => {
            let dt = t.elapsed();
            println!(
                "\x1b[1m{}\x1b[0m block(s) matched in \x1b[1m{:.2?}\x1b[0m (hybrid: lexical + semantic{}, {})",
                outcome.hits.len(),
                dt,
                if outcome.reranked { " + rerank" } else { "" },
                prov.id()
            );
            output::print_ask_hits(&idx, &outcome.hits, &query, outcome.reranked);
            if outcome.changed {
                let _ = store::save(&idx, &root);
            }
        }
        Err(e) => {
            eprintln!("warning: semantic search unavailable ({e}); falling back to lexical");
            let hits = search::search(&idx, &query, opts.limit);
            output::print_hits(&idx, &hits, &query, opts.limit, t.elapsed());
        }
    }
}

fn cmd_graph(args: &[String]) {
    let opts = parse_opts(args);
    let name = opts.terms.join(" ");
    if name.is_empty() {
        eprintln!("error: graph requires a SYMBOL name");
        std::process::exit(2);
    }
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let idx = load_or_build(&root, opts.use_cache);
    let g = &idx.graph;
    let syms = g.find_all(&name);
    if syms.is_empty() {
        eprintln!("no symbol named '{name}' found.");
        let sugg = g.suggest(&name);
        if !sugg.is_empty() {
            eprintln!("did you mean: {} ?", sugg.join(", "));
        }
        std::process::exit(1);
    }
    for id in syms {
        print_symbol(&idx, g, id);
    }
}

fn print_symbol(idx: &Index, g: &endex::graph::Graph, id: u32) {
    let s = &g.symbols[id as usize];
    println!(
        "\x1b[1m{}\x1b[0m  [{}]  \x1b[36m{}:{}\x1b[0m",
        s.name,
        s.kind.label(),
        idx.path_of(s.file),
        s.line
    );
    let callees = g.callees(id);
    if !callees.is_empty() {
        println!("  \x1b[2mcalls:\x1b[0m");
        for &c in callees.iter().take(20) {
            let t = &g.symbols[c as usize];
            println!(
                "    → {}  \x1b[2m{}:{}\x1b[0m",
                t.name,
                idx.path_of(t.file),
                t.line
            );
        }
        if callees.len() > 20 {
            println!("    \x1b[2m··· {} total\x1b[0m", callees.len());
        }
    }
    let callers = g.callers(id);
    if !callers.is_empty() {
        println!("  \x1b[2mcalled by:\x1b[0m");
        for &c in callers.iter().take(20) {
            let t = &g.symbols[c as usize];
            println!(
                "    ← {}  \x1b[2m{}:{}\x1b[0m",
                t.name,
                idx.path_of(t.file),
                t.line
            );
        }
        if callers.len() > 20 {
            println!("    \x1b[2m··· {} total\x1b[0m", callers.len());
        }
    }
    let importers: Vec<&str> = g
        .file_imports
        .iter()
        .filter(|(_, to)| *to == s.file)
        .map(|(from, _)| idx.path_of(*from))
        .take(10)
        .collect();
    if !importers.is_empty() {
        println!("  \x1b[2mimported by:\x1b[0m {}", importers.join(", "));
    }
    println!();
}

fn cmd_flow(args: &[String]) {
    let opts = parse_opts(args);
    if opts.terms.len() < 2 {
        eprintln!("error: flow requires two symbol names: endex flow [DIR] FROM TO");
        std::process::exit(2);
    }
    let (from, to) = (opts.terms[0].clone(), opts.terms[1].clone());
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let idx = load_or_build(&root, opts.use_cache);
    let g = &idx.graph;
    let sources = g.find_all(&from);
    let targets: HashSet<u32> = g.find_all(&to).into_iter().collect();
    if sources.is_empty() {
        eprintln!(
            "no symbol named '{from}' found. did you mean: {} ?",
            g.suggest(&from).join(", ")
        );
        std::process::exit(1);
    }
    if targets.is_empty() {
        eprintln!(
            "no symbol named '{to}' found. did you mean: {} ?",
            g.suggest(&to).join(", ")
        );
        std::process::exit(1);
    }
    let paths = g.find_paths(&sources, &targets, 8, 5);
    if paths.is_empty() {
        println!("no call path found from '{from}' to '{to}' (max depth 8)");
        return;
    }
    println!(
        "{} path(s) from \x1b[1m{from}\x1b[0m to \x1b[1m{to}\x1b[0m:",
        paths.len()
    );
    for p in &paths {
        let mut first = true;
        println!();
        for &sid in p {
            let s = &g.symbols[sid as usize];
            if first {
                print!(
                    "\x1b[1m{}\x1b[0m \x1b[2m{}:{}\x1b[0m",
                    s.name,
                    idx.path_of(s.file),
                    s.line
                );
                first = false;
            } else {
                print!(
                    "\n  \x1b[2m--calls-->\x1b[0m \x1b[1m{}\x1b[0m \x1b[2m{}:{}\x1b[0m",
                    s.name,
                    idx.path_of(s.file),
                    s.line
                );
            }
        }
        println!();
    }
}

fn cmd_clues(args: &[String]) {
    let opts = parse_opts(args);
    let term = opts.terms.join(" ");
    if term.is_empty() {
        eprintln!("error: clues requires a TERM");
        std::process::exit(2);
    }
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let idx = load_or_build(&root, opts.use_cache);
    let g = &idx.graph;
    let t = Instant::now();
    let hits = search::search(&idx, &term, 15);
    println!(
        "clues for '{term}' — {} block(s) in \x1b[1m{:.2?}\x1b[0m",
        hits.len(),
        t.elapsed()
    );
    for hit in &hits {
        println!("\x1b[1;36m{}:{}\x1b[0m", idx.path_of(hit.file_id), hit.line);
        if let Some(syms) = g.by_block.get(&hit.block_id) {
            for &sid in syms {
                let s = &g.symbols[sid as usize];
                let callers: Vec<String> = g
                    .callers(sid)
                    .iter()
                    .take(6)
                    .map(|&c| g.symbols[c as usize].name.clone())
                    .collect();
                let callees: Vec<String> = g
                    .callees(sid)
                    .iter()
                    .take(6)
                    .map(|&c| g.symbols[c as usize].name.clone())
                    .collect();
                println!(
                    "  \x1b[1m{}\x1b[0m [{}] — called by: {} · calls: {}",
                    s.name,
                    s.kind.label(),
                    if callers.is_empty() {
                        "-".into()
                    } else {
                        callers.join(", ")
                    },
                    if callees.is_empty() {
                        "-".into()
                    } else {
                        callees.join(", ")
                    }
                );
            }
        }
        output::print_block_matches(&hit.text, hit.line, &term.to_lowercase(), 2);
    }
}

// ---------- mcp ----------

fn cmd_mcp(args: &[String]) {
    let opts = parse_opts(args);
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    let root_str = root.to_string_lossy().to_string();
    // Resolve the embedding provider up front (flags/env) so endex_ask and
    // the background embedder share the same configuration.
    let prov = provider_from(&opts);
    mcp::run(root_str, opts.use_cache, prov);
}

// ---------- watch ----------

fn cmd_watch(args: &[String]) {
    let opts = parse_opts(args);
    let root = opts.dir.canonicalize().unwrap_or_else(|_| opts.dir.clone());
    eprintln!("Indexing {} ...", root.display());
    let mut idx = load_or_build(&root, opts.use_cache);
    eprintln!(
        "Ready: {} files / {} blocks / {} symbols. Watching for changes...",
        idx.file_count(),
        idx.block_count(),
        idx.graph.symbols.len()
    );

    let rx = match watch::watch(&root) {
        Ok(rx) => rx,
        Err(e) => {
            eprintln!("error: could not start watcher: {e}");
            std::process::exit(1);
        }
    };

    enum Msg {
        Changed(Vec<PathBuf>),
        Input(String),
    }
    let (tx, msg_rx) = mpsc::channel();

    let tx_in = tx.clone();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    if tx_in.send(Msg::Input(l)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    std::thread::spawn(move || {
        while let Ok(batch) = rx.recv() {
            if tx.send(Msg::Changed(batch)).is_err() {
                break;
            }
        }
    });

    let mut dirty = false;
    let mut limit = opts.limit;
    let mut prov = provider_from(&opts);
    let mut ignores = watch::Ignores::new(&root);
    println!(
        "Type a query and press Enter. Prefix with ? for hybrid semantic search ({}).
Commands: :graph N  :flow A B  :clues T  :embed [provider]  :limit N  :save  :stats  :quit",
        prov.id()
    );
    print!("search> ");
    let _ = io::stdout().flush();

    loop {
        match msg_rx.recv() {
            Ok(Msg::Input(line)) => {
                let line = line.trim().to_string();
                match line.as_str() {
                    "" => {}
                    ":q" | ":quit" | ":exit" => break,
                    ":save" => {
                        let _ = store::save(&idx, &root);
                        dirty = false;
                    }
                    ":stats" => println!(
                        "{} files / {} blocks / {} symbols / {} call edges / {} embeddings{}",
                        idx.file_count(),
                        idx.block_count(),
                        idx.graph.symbols.len(),
                        idx.graph.call_edge_count(),
                        idx.embeddings.map.len(),
                        if dirty { " (unsaved changes)" } else { "" }
                    ),
                    _ if line == ":embed" || line.starts_with(":embed ") => {
                        if let Some(p) = line.strip_prefix(":embed").map(str::trim) {
                            if !p.is_empty() {
                                prov = embed::Provider::resolve(&embed::ProviderOpts {
                                    provider: Some(p.to_string()),
                                    ..Default::default()
                                });
                            }
                        }
                        match embed::ensure_all(&mut idx, &prov) {
                            Ok(()) => {
                                dirty = true;
                                println!(
                                    "embeddings ready: {} vectors, {} dim ({})",
                                    idx.embeddings.map.len(),
                                    idx.embeddings.dim,
                                    prov.id()
                                );
                            }
                            Err(e) => println!("embed failed: {e}"),
                        }
                    }
                    _ if line.starts_with(':') => {
                        let mut parts = line.split_whitespace();
                        let cmd = parts.next().unwrap_or("");
                        let rest: Vec<&str> = parts.collect();
                        match (cmd, rest.as_slice()) {
                            (":graph", [name]) => {
                                let g = &idx.graph;
                                let syms = g.find_all(name);
                                if syms.is_empty() {
                                    println!("no symbol named '{name}'");
                                } else {
                                    for id in syms {
                                        print_symbol(&idx, g, id);
                                    }
                                }
                            }
                            (":flow", [a, b]) => {
                                let g = &idx.graph;
                                let sources = g.find_all(a);
                                let targets: HashSet<u32> = g.find_all(b).into_iter().collect();
                                let paths = g.find_paths(&sources, &targets, 8, 5);
                                if paths.is_empty() {
                                    println!("no call path from {a} to {b}");
                                }
                                for p in &paths {
                                    let mut first = true;
                                    println!();
                                    for &sid in p {
                                        let s = &g.symbols[sid as usize];
                                        if first {
                                            print!(
                                                "\x1b[1m{}\x1b[0m \x1b[2m{}:{}\x1b[0m",
                                                s.name,
                                                idx.path_of(s.file),
                                                s.line
                                            );
                                            first = false;
                                        } else {
                                            print!("\n  \x1b[2m--calls-->\x1b[0m \x1b[1m{}\x1b[0m \x1b[2m{}:{}\x1b[0m",
                                                s.name, idx.path_of(s.file), s.line);
                                        }
                                    }
                                    println!();
                                }
                            }
                            (":clues", terms) if !terms.is_empty() => {
                                let term = terms.join(" ");
                                let g = &idx.graph;
                                let hits = search::search(&idx, &term, 10);
                                for hit in &hits {
                                    println!(
                                        "\x1b[1;36m{}:{}\x1b[0m",
                                        idx.path_of(hit.file_id),
                                        hit.line
                                    );
                                    if let Some(syms) = g.by_block.get(&hit.block_id) {
                                        for &sid in syms {
                                            let s = &g.symbols[sid as usize];
                                            println!(
                                                "  \x1b[1m{}\x1b[0m [{}]",
                                                s.name,
                                                s.kind.label()
                                            );
                                        }
                                    }
                                }
                            }
                            (":limit", [n]) => {
                                if let Ok(v) = n.parse::<usize>() {
                                    limit = v;
                                    println!("limit set to {v}");
                                }
                            }
                            _ => println!(
                                "unknown command: {line} (try :graph, :flow, :clues, :embed)"
                            ),
                        }
                    }
                    _ if line.starts_with('?') => {
                        let q = line[1..].trim().to_string();
                        if q.is_empty() {
                            println!("usage: ? QUERY");
                        } else {
                            let t = Instant::now();
                            match embed::ask(&mut idx, &prov, &q, limit) {
                                Ok(outcome) => {
                                    dirty |= outcome.changed;
                                    println!(
                                        "\x1b[1m{}\x1b[0m block(s) matched in \x1b[1m{:.2?}\x1b[0m (hybrid{}, {})",
                                        outcome.hits.len(),
                                        t.elapsed(),
                                        if outcome.reranked { " + rerank" } else { "" },
                                        prov.id()
                                    );
                                    output::print_ask_hits(
                                        &idx,
                                        &outcome.hits,
                                        &q,
                                        outcome.reranked,
                                    );
                                }
                                Err(e) => {
                                    eprintln!("semantic search failed: {e}");
                                    let hits = search::search(&idx, &q, limit);
                                    output::print_hits(&idx, &hits, &q, limit, t.elapsed());
                                }
                            }
                        }
                    }
                    _ => {
                        let t = Instant::now();
                        let hits = search::search(&idx, &line, limit);
                        output::print_hits(&idx, &hits, &line, limit, t.elapsed());
                    }
                }
                print!("search> ");
                let _ = io::stdout().flush();
            }
            Ok(Msg::Changed(paths)) => {
                let mut n = 0usize;
                for p in &paths {
                    // Same ignore rules as full walks — otherwise gitignored
                    // secrets (.env) and build output would leak into the
                    // index through watcher events.
                    if ignores.is_ignored(p, p.is_dir()) {
                        continue;
                    }
                    if !p.is_file() {
                        idx.remove_file(p);
                        continue;
                    }
                    if idx.index_file(p) {
                        n += 1;
                    }
                }
                if n > 0 {
                    idx.finalize(); // rebuild graph + GC embeddings in memory
                    dirty = true;
                    eprintln!("\x1b[2m  · reindexed {n} changed file(s), graph refreshed\x1b[0m");
                    print!("search> ");
                    let _ = io::stdout().flush();
                }
            }
            Err(_) => break, // all senders gone
        }
    }

    if dirty {
        eprintln!("Saving updated cache...");
        let _ = store::save(&idx, &root);
    }
    println!("Bye.");
}
