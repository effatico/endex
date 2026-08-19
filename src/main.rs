use endex::{index::Index, search, store, watch};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

const CACHE_FILENAME: &str = ".endex-index.bin";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("index") => cmd_index(&args[1..]),
        Some("search") => cmd_search(&args[1..]),
        Some("watch") => cmd_watch(&args[1..]),
        _ => {
            eprintln!(
                "endex — fast cached code indexer with millisecond search\n\n\
                 USAGE:\n\
                 \x20 endex index  [DIR]          build or refresh the cache for DIR (default: .)\n\
                 \x20 endex search [DIR] QUERY    one-shot search using the cache\n\
                 \x20 endex watch  [DIR]          watch for changes + interactive search REPL\n\n\
                 OPTIONS:\n\
                 \x20 --limit N    max results (default 50)\n\
                 \x20 --no-cache   ignore the on-disk cache and rebuild from scratch\n\n\
                 The cache is stored as {CACHE_FILENAME} inside the indexed directory."
            );
            std::process::exit(2);
        }
    }
}

// ---------- arg helpers ----------

struct Opts {
    dir: PathBuf,
    query: Option<String>,
    limit: usize,
    use_cache: bool,
}

fn parse_opts(args: &[String]) -> Opts {
    let mut opts = Opts {
        dir: PathBuf::from("."),
        query: None,
        limit: 50,
        use_cache: true,
    };
    let mut positional: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" | "-l" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    opts.limit = v;
                    i += 1;
                }
            }
            "--no-cache" => opts.use_cache = false,
            _ => positional.push(&args[i]),
        }
        i += 1;
    }
    if let Some(dir) = positional.first() {
        opts.dir = PathBuf::from(dir);
    }
    if let Some(q) = positional.get(1) {
        opts.query = Some(q.to_string());
    }
    opts
}

// ---------- load-or-build ----------

/// Load the cache if present & valid, else full build. Always refreshes
/// incrementally afterwards so results are never stale.
fn load_or_build(root: &Path, use_cache: bool) -> Index {
    let t0 = Instant::now();
    let mut idx = if use_cache {
        match store::load(root) {
            Some(i) => {
                eprintln!(
                    "  cache loaded: {} files / {} blocks in {:?}",
                    i.file_count(),
                    i.block_count(),
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
    }
    idx
}

// ---------- commands ----------

fn cmd_index(args: &[String]) {
    let opts = parse_opts(args);
    let root = opts.dir.canonicalize().unwrap_or(opts.dir);
    eprintln!("Indexing {} ...", root.display());
    let t0 = Instant::now();
    let idx = load_or_build(&root, opts.use_cache);
    eprintln!(
        "Done: {} files, {} blocks, {} trigram lists — total {:?}",
        idx.file_count(),
        idx.block_count(),
        idx.postings.len(),
        t0.elapsed()
    );
}

fn cmd_search(args: &[String]) {
    let opts = parse_opts(args);
    let Some(query) = opts.query else {
        eprintln!("error: search requires a QUERY (quoted if it has spaces)");
        std::process::exit(2);
    };
    let root = opts.dir.canonicalize().unwrap_or(opts.dir);
    let idx = load_or_build(&root, opts.use_cache);
    let t = Instant::now();
    let hits = search::search(&idx, &query, opts.limit);
    let search_time = t.elapsed();
    print_hits(&idx, &hits, &query, opts.limit, search_time);
}

fn cmd_watch(args: &[String]) {
    let opts = parse_opts(args);
    let root = opts.dir.canonicalize().unwrap_or(opts.dir);
    eprintln!("Indexing {} ...", root.display());
    let mut idx = load_or_build(&root, opts.use_cache);
    eprintln!(
        "Ready: {} files / {} blocks indexed. Watching for changes...",
        idx.file_count(),
        idx.block_count()
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

    // stdin reader thread.
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
    // watcher forwarder thread.
    std::thread::spawn(move || {
        while let Ok(batch) = rx.recv() {
            if tx.send(Msg::Changed(batch)).is_err() {
                break;
            }
        }
    });

    let mut dirty = false;
    let mut limit = opts.limit;
    println!("Type a query and press Enter. Commands: :limit N  :save  :stats  :quit");
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
                        "{} files / {} blocks / {} trigram lists{}",
                        idx.file_count(),
                        idx.block_count(),
                        idx.postings.len(),
                        if dirty { " (unsaved changes)" } else { "" }
                    ),
                    _ if line.starts_with(":limit") => {
                        if let Some(v) = line.split_whitespace().nth(1).and_then(|s| s.parse().ok())
                        {
                            limit = v;
                            println!("limit set to {v}");
                        } else {
                            println!("usage: :limit N");
                        }
                    }
                    _ => {
                        let t = Instant::now();
                        let hits = search::search(&idx, &line, limit);
                        let search_time = t.elapsed();
                        print_hits(&idx, &hits, &line, limit, search_time);
                    }
                }
                print!("search> ");
                let _ = io::stdout().flush();
            }
            Ok(Msg::Changed(paths)) => {
                let mut n = 0usize;
                for p in &paths {
                    if !p.is_file() {
                        idx.remove_file(p);
                        continue;
                    }
                    // Skip our own cache file and anything under .git.
                    if p.file_name().map(|f| f == CACHE_FILENAME).unwrap_or(false)
                        || p.components().any(|c| c.as_os_str() == ".git")
                    {
                        continue;
                    }
                    if idx.index_file(p) {
                        n += 1;
                    }
                }
                if n > 0 {
                    dirty = true;
                    eprintln!("\x1b[2m  · reindexed {n} changed file(s)\x1b[0m");
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

// ---------- output ----------

fn print_hits(
    idx: &Index,
    hits: &[search::Hit],
    query: &str,
    limit: usize,
    search_time: std::time::Duration,
) {
    let q = query.to_lowercase();
    let total = hits.len();
    println!(
        "\x1b[1m{total}\x1b[0m block(s) matched in \x1b[1m{:.2?}\x1b[0m{}",
        search_time,
        if total == limit {
            format!(" (showing top {limit})")
        } else {
            String::new()
        }
    );

    let mut stdout = io::stdout().lock();
    for hit in hits {
        let path = idx.path_of(hit.file_id);
        let _ = write!(stdout, "\x1b[1;36m{}:{}\x1b[0m", path, hit.line);
        // Show the matching lines of the block (with line numbers).
        let mut shown = 0;
        let mut lineno = hit.line;
        for line in hit.text.lines() {
            let l = lineno;
            lineno += 1;
            if !line.to_lowercase().contains(&q) {
                continue;
            }
            if shown == 6 {
                let _ = writeln!(stdout, "\x1b[2m  ··· (more matches in this block)\x1b[0m");
                break;
            }
            let _ = writeln!(stdout);
            let _ = write!(stdout, "  \x1b[2m{l:>5}|\x1b[0m ");
            let lower = line.to_lowercase();
            let mut start = 0usize;
            // highlight every match in the line
            while let Some(pos) = lower[start..].find(&q) {
                let abs = start + pos;
                let _ = write!(stdout, "{}", &line[start..abs]);
                let _ = write!(
                    stdout,
                    "\x1b[1;31m{}\x1b[0m",
                    &line[abs..abs + q.len().min(line.len() - abs)]
                );
                start = abs + q.len().max(1);
            }
            let _ = writeln!(stdout, "{}", &line[start..]);
            shown += 1;
        }
    }
    let _ = stdout.flush();
}
