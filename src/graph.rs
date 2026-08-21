//! Knowledge graph of the codebase: symbol definitions, call edges, and
//! file-level imports.
//!
//! Extraction is heuristic (line-based per language, no full parse) so a full
//! rebuild stays in the tens-of-milliseconds range even for large repos. The
//! graph is a pure function of the in-memory index (defs are stored per file
//! at index time), so incremental updates never re-read files from disk.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::index::{Index, TOMBSTONE_FILE};

// ---------- types ----------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymbolKind {
    Func,
    Method,
    Struct,
    Enum,
    Trait,
    Type,
    Class,
    Interface,
    Impl,
}

impl SymbolKind {
    pub fn label(self) -> &'static str {
        match self {
            SymbolKind::Func => "func",
            SymbolKind::Method => "method",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Type => "type",
            SymbolKind::Class => "class",
            SymbolKind::Interface => "interface",
            SymbolKind::Impl => "impl",
        }
    }
}

/// A definition extracted at index time and stored per file.
#[derive(Serialize, Deserialize, Clone)]
pub struct Def {
    pub name: String,
    pub kind: SymbolKind,
    pub line: u32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file: u32,
    pub line: u32,
    pub block: u32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Graph {
    pub symbols: Vec<Symbol>,
    /// name -> symbol ids
    pub by_name: HashMap<String, Vec<u32>>,
    /// block id -> symbol ids defined in that block
    pub by_block: HashMap<u32, Vec<u32>>,
    /// call edges (from, to), deduped + sorted
    pub edges: Vec<(u32, u32)>,
    /// resolved file -> file import edges
    pub file_imports: Vec<(u32, u32)>,
    #[serde(default)]
    pub out_adj: Vec<Vec<u32>>,
    #[serde(default)]
    pub in_adj: Vec<Vec<u32>>,
}

// ---------- language detection ----------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    TsJs,
    Python,
    Go,
    JavaLike,
    Plain,
}

pub fn lang_of(path: &str) -> Lang {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Lang::Rust,
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => Lang::TsJs,
        "py" => Lang::Python,
        "go" => Lang::Go,
        "java" | "cs" | "kt" | "scala" => Lang::JavaLike,
        _ => Lang::Plain,
    }
}

// ---------- definition extraction ----------

fn is_ident_char(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric()
}

/// Byte position of `kw` occurring as a whole word in `line`.
fn find_word(line: &str, kw: &str) -> Option<usize> {
    let b = line.as_bytes();
    let kb = kw.as_bytes();
    if kb.is_empty() || b.len() < kb.len() {
        return None;
    }
    let mut i = 0;
    while i + kb.len() <= b.len() {
        if &b[i..i + kb.len()] == kb {
            let before_ok = i == 0 || !is_ident_char(b[i - 1]);
            let after_ok = i + kb.len() == b.len() || !is_ident_char(b[i + kb.len()]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Identifier that follows whole-word keyword `kw` (skipping Rust generics
/// `impl<T>` and Go receivers `func (r T) Name`).
fn ident_after_kw(line: &str, kw: &str) -> Option<String> {
    let pos = find_word(line, kw)?;
    let b = line.as_bytes();
    let mut j = pos + kw.len();
    // Skip Go receiver: func (r *T) Name
    if j < b.len() && b[j] == b'(' {
        let mut depth = 0;
        while j < b.len() {
            if b[j] == b'(' {
                depth += 1;
            } else if b[j] == b')' {
                depth -= 1;
                if depth == 0 {
                    j += 1;
                    break;
                }
            }
            j += 1;
        }
    }
    while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
        j += 1;
    }
    // Skip Rust generics: impl<T> Foo
    if j < b.len() && b[j] == b'<' {
        let mut depth = 0;
        while j < b.len() {
            if b[j] == b'<' {
                depth += 1;
            } else if b[j] == b'>' {
                depth -= 1;
                if depth == 0 {
                    j += 1;
                    break;
                }
            }
            j += 1;
        }
        while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
            j += 1;
        }
    }
    if j < b.len() && (b[j] == b'_' || b[j].is_ascii_alphabetic()) {
        let s = j;
        while j < b.len() && is_ident_char(b[j]) {
            j += 1;
        }
        return Some(line[s..j].to_string());
    }
    None
}

const KEYWORDS: &[&str] = &[
    "if", "else", "for", "while", "loop", "match", "case", "return", "new", "typeof", "let", "var",
    "const", "fn", "func", "def", "class", "struct", "enum", "trait", "impl", "use", "mod", "pub",
    "import", "from", "export", "self", "this", "super", "true", "false", "null", "nil", "none",
    "async", "await", "static", "final", "void", "int", "uint", "float", "bool", "string", "error",
    "result", "some", "ok", "err", "throw", "catch", "try", "switch", "break", "continue", "type",
];

fn is_keyword(name: &str) -> bool {
    KEYWORDS.contains(&name)
}

fn rust_def(line: &str) -> Option<(String, SymbolKind)> {
    if let Some(n) = ident_after_kw(line, "fn") {
        return Some((n, SymbolKind::Func));
    }
    if let Some(n) = ident_after_kw(line, "struct") {
        return Some((n, SymbolKind::Struct));
    }
    if let Some(n) = ident_after_kw(line, "enum") {
        return Some((n, SymbolKind::Enum));
    }
    if let Some(n) = ident_after_kw(line, "trait") {
        return Some((n, SymbolKind::Trait));
    }
    if line.contains('=') {
        if let Some(n) = ident_after_kw(line, "type") {
            return Some((n, SymbolKind::Type));
        }
    }
    if let Some(n) = ident_after_kw(line, "impl") {
        return Some((n, SymbolKind::Impl));
    }
    None
}

fn ts_def(line: &str) -> Option<(String, SymbolKind)> {
    if let Some(n) = ident_after_kw(line, "function") {
        return Some((n, SymbolKind::Func));
    }
    if let Some(n) = ident_after_kw(line, "class") {
        return Some((n, SymbolKind::Class));
    }
    if let Some(n) = ident_after_kw(line, "interface") {
        return Some((n, SymbolKind::Interface));
    }
    if let Some(n) = ident_after_kw(line, "enum") {
        return Some((n, SymbolKind::Enum));
    }
    if line.contains('=') {
        if let Some(n) = ident_after_kw(line, "type") {
            return Some((n, SymbolKind::Type));
        }
        if let Some(n) = ident_after_kw(line, "const") {
            if line.contains("=>") || line.contains("function") {
                return Some((n, SymbolKind::Func));
            }
        }
    }
    None
}

fn py_def(line: &str) -> Option<(String, SymbolKind)> {
    if let Some(n) = ident_after_kw(line, "def") {
        return Some((n, SymbolKind::Func));
    }
    if let Some(n) = ident_after_kw(line, "class") {
        return Some((n, SymbolKind::Class));
    }
    None
}

fn go_def(line: &str) -> Option<(String, SymbolKind)> {
    if let Some(n) = ident_after_kw(line, "func") {
        let is_method = {
            let b = line.as_bytes();
            match find_word(line, "func") {
                Some(pos) => {
                    let mut j = pos + 4;
                    while j < b.len() && b[j] == b' ' {
                        j += 1;
                    }
                    j < b.len() && b[j] == b'('
                }
                None => false,
            }
        };
        return Some((
            n,
            if is_method {
                SymbolKind::Method
            } else {
                SymbolKind::Func
            },
        ));
    }
    if let Some(n) = ident_after_kw(line, "type") {
        if line.contains("struct") {
            return Some((n, SymbolKind::Struct));
        }
        if line.contains("interface") {
            return Some((n, SymbolKind::Interface));
        }
    }
    None
}

const JAVA_MODIFIERS: &[&str] = &[
    "public",
    "private",
    "protected",
    "static",
    "final",
    "abstract",
    "override",
    "virtual",
    "async",
    "synchronized",
    "native",
    "sealed",
    "internal",
    "extern",
];

fn java_def(line: &str) -> Option<(String, SymbolKind)> {
    if let Some(n) = ident_after_kw(line, "class") {
        return Some((n, SymbolKind::Class));
    }
    if let Some(n) = ident_after_kw(line, "interface") {
        return Some((n, SymbolKind::Interface));
    }
    if let Some(n) = ident_after_kw(line, "enum") {
        return Some((n, SymbolKind::Enum));
    }
    // Conservative method detection: line must start with an explicit
    // modifier, then a return type, then `name(`.
    if !line.contains('(') {
        return None;
    }
    let mut toks = line.split_whitespace();
    let mut mods = 0;
    while let Some(tok) = toks.next() {
        if JAVA_MODIFIERS.contains(&tok) {
            mods += 1;
        } else {
            if mods == 0 {
                return None;
            }
            let name_tok = toks.next()?;
            let name = name_tok.split('(').next().unwrap_or("");
            if name.len() >= 2
                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                && !is_keyword(name)
            {
                return Some((name.to_string(), SymbolKind::Method));
            }
            return None;
        }
    }
    None
}

/// Extract definitions (with 1-based line numbers) from a file's text.
pub fn extract_defs(path: &str, text: &str) -> Vec<Def> {
    let lang = lang_of(path);
    let mut defs = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_start();
        let d = match lang {
            Lang::Rust => rust_def(line),
            Lang::TsJs => ts_def(line),
            Lang::Python => py_def(line),
            Lang::Go => go_def(line),
            Lang::JavaLike => java_def(line),
            Lang::Plain => None,
        };
        if let Some((name, kind)) = d {
            if name.len() >= 2 && !is_keyword(&name) {
                defs.push(Def {
                    name,
                    kind,
                    line: (i + 1) as u32,
                });
            }
        }
    }
    defs
}

// ---------- import extraction ----------

fn first_quoted(s: &str) -> Option<String> {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'\'' || b[i] == b'"' || b[i] == b'`' {
            let quote = b[i];
            let mut j = i + 1;
            while j < b.len() && b[j] != quote {
                j += 1;
            }
            if j > i + 1 {
                return Some(s[i + 1..j].to_string());
            }
            return None;
        }
    }
    None
}

fn quoted_after(line: &str, kw: &str) -> Option<String> {
    let pos = find_word(line, kw)?;
    first_quoted(&line[pos + kw.len()..])
}

/// Extract module specs referenced by import statements.
pub fn extract_imports(path: &str, text: &str) -> Vec<String> {
    let lang = lang_of(path);
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        match lang {
            Lang::TsJs => {
                if let Some(spec) = quoted_after(line, "from") {
                    out.push(spec);
                } else if line.starts_with("import ") || line.contains("require(") {
                    if let Some(spec) = first_quoted(line) {
                        out.push(spec);
                    }
                }
            }
            Lang::Rust => {
                if line.starts_with("use ") || line.starts_with("pub use ") {
                    let rest = line
                        .split_once("use ")
                        .map(|(_, r)| r)
                        .unwrap_or("")
                        .split('{')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .trim_end_matches(';')
                        .trim();
                    if let Some(p) = rest.strip_prefix("crate::") {
                        out.push(p.trim_end_matches("::").replace("::", "/"));
                    }
                } else if line.starts_with("mod ") || line.starts_with("pub mod ") {
                    let m = line
                        .split_once("mod ")
                        .map(|(_, r)| r)
                        .unwrap_or("")
                        .trim_end_matches(';')
                        .trim();
                    if let Some(first) = m.split(' ').next() {
                        out.push(first.to_string());
                    }
                }
            }
            Lang::Python => {
                if let Some(rest) = line.strip_prefix("from ") {
                    if let Some(spec) = rest.split_whitespace().next() {
                        out.push(spec.to_string());
                    }
                } else if let Some(rest) = line.strip_prefix("import ") {
                    if let Some(spec) = rest.split(',').next() {
                        out.push(spec.trim().to_string());
                    }
                }
            }
            Lang::Go => {
                if line.starts_with("import ") || line.starts_with('"') {
                    if let Some(spec) = first_quoted(line) {
                        out.push(spec);
                    }
                }
            }
            Lang::JavaLike => {
                if let Some(rest) = line.strip_prefix("import ") {
                    out.push(rest.trim_end_matches(';').trim().to_string());
                }
            }
            Lang::Plain => {}
        }
    }
    out
}

// ---------- import resolution ----------

fn normalize_rel(p: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

fn try_exts(base: &str, rel_map: &HashMap<String, u32>) -> Option<u32> {
    const EXTS: &[&str] = &[
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".py", ".go", "",
    ];
    const INDEX: &[&str] = &[
        "/index.ts",
        "/index.tsx",
        "/index.js",
        "/mod.rs",
        "/__init__.py",
    ];
    for e in EXTS {
        if let Some(id) = rel_map.get(&format!("{base}{e}")) {
            return Some(*id);
        }
    }
    for i in INDEX {
        if let Some(id) = rel_map.get(&format!("{base}{i}")) {
            return Some(*id);
        }
    }
    None
}

/// Resolve an import spec to file ids in the repo (may resolve to several
/// files, e.g. a Go package directory).
fn resolve_spec(
    spec: &str,
    from_rel: &str,
    rel_map: &HashMap<String, u32>,
    stem_map: &HashMap<String, Vec<u32>>,
    dir_map: &HashMap<String, Vec<u32>>,
) -> Vec<u32> {
    let mut spec = spec.trim().to_string();
    if spec.is_empty() {
        return Vec::new();
    }
    // Python relative: ".mod" -> "./mod"
    if spec.starts_with('.') && !spec.starts_with("./") && !spec.starts_with("../") {
        spec = format!("./{spec}");
    }
    // TS path aliases: "@/foo" -> "foo"
    for prefix in ["@/", "~/"] {
        if let Some(rest) = spec.strip_prefix(prefix) {
            spec = rest.to_string();
            break;
        }
    }
    // Relative spec: join with the importing file's directory.
    if spec.starts_with("./") || spec.starts_with("../") {
        let dir = from_rel.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
        let norm = normalize_rel(&format!("{dir}/{spec}"));
        if let Some(id) = try_exts(&norm, rel_map) {
            return vec![id];
        }
        return Vec::new();
    }
    // Module-path style: "a/b/c" or "com.x.Foo" or "github.com/o/r/pkg".
    let mut cands: Vec<String> = vec![spec.replace('.', "/")];
    let segs: Vec<&str> = spec.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() >= 2 {
        cands.push(format!("{}/{}", segs[segs.len() - 2], segs[segs.len() - 1]));
    }
    for c in &cands {
        if let Some(id) = try_exts(c, rel_map) {
            return vec![id];
        }
    }
    // Fallback: unique file stem, or a directory of the same name.
    if let Some(last) = segs.last() {
        if let Some(ids) = stem_map.get(*last) {
            if ids.len() == 1 {
                return ids.clone();
            }
        }
        if let Some(ids) = dir_map.get(*last) {
            return ids.iter().copied().take(10).collect();
        }
    }
    Vec::new()
}

// ---------- graph construction ----------

/// Words in a line that are immediately (modulo whitespace) followed by `(`.
fn call_tokens(line: &str) -> Vec<(&str, bool)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'_' || b[i].is_ascii_alphabetic() {
            let s = i;
            while i < b.len() && is_ident_char(b[i]) {
                i += 1;
            }
            let mut j = i;
            while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
                j += 1;
            }
            let paren = j < b.len() && b[j] == b'(';
            out.push((&line[s..i], paren));
        } else {
            i += 1;
        }
    }
    out
}

fn containing_block(index: &Index, blocks: &[u32], line: u32) -> u32 {
    let mut lo = 0usize;
    let mut hi = blocks.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if index.blocks[blocks[mid] as usize].line <= line {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        blocks.first().copied().unwrap_or(u32::MAX)
    } else {
        blocks[lo - 1]
    }
}

fn rel_of(root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

/// Names with more definitions than this and no same-file candidate are not
/// linked at all: global name-matching on ultra-common names (`new`, `get`,
/// `run`, ...) produces pure noise.
const MAX_GLOBAL_CALL_TARGETS: usize = 25;

/// Resolve a bare `name(` call to the likeliest definitions. Edges are
/// name-based (no type info), so disambiguate: a definition in the SAME
/// file always wins; otherwise link all candidates, unless the name is so
/// common the edges would be meaningless.
fn resolve_call_targets(targets: &[u32], from_file: u32, sym_files: &[u32]) -> Vec<u32> {
    if targets.len() <= 1 {
        return targets.to_vec();
    }
    let same_file: Vec<u32> = targets
        .iter()
        .copied()
        .filter(|&t| sym_files[t as usize] == from_file)
        .collect();
    if !same_file.is_empty() {
        return same_file;
    }
    if targets.len() > MAX_GLOBAL_CALL_TARGETS {
        return Vec::new();
    }
    targets.to_vec()
}

/// Rebuild the whole graph from in-memory index state (no disk reads).
pub fn rebuild(index: &Index) -> Graph {
    let mut g = Graph::default();

    // Lookup maps for import resolution.
    let mut rel_map: HashMap<String, u32> = HashMap::new();
    let mut stem_map: HashMap<String, Vec<u32>> = HashMap::new();
    let mut dir_map: HashMap<String, Vec<u32>> = HashMap::new();
    for (path, fe) in &index.files {
        let rel = rel_of(&index.root, path);
        rel_map.insert(rel.clone(), fe.id);
        let stem = rel
            .rsplit('/')
            .next()
            .unwrap_or(&rel)
            .split('.')
            .next()
            .unwrap_or("");
        stem_map.entry(stem.to_string()).or_default().push(fe.id);
        if let Some((dir, _)) = rel.rsplit_once('/') {
            dir_map.entry(dir.to_string()).or_default().push(fe.id);
        }
    }

    // Files in stable id order.
    let mut file_list: Vec<(&std::path::PathBuf, &crate::index::FileEntry)> =
        index.files.iter().collect();
    file_list.sort_by_key(|(_, fe)| fe.id);

    // 1. Symbols.
    let mut file_syms: HashMap<u32, Vec<u32>> = HashMap::new();
    for (path, fe) in &file_list {
        let _ = path;
        let mut syms = Vec::with_capacity(fe.defs.len());
        for def in &fe.defs {
            let block = containing_block(index, &fe.blocks, def.line);
            let id = g.symbols.len() as u32;
            g.symbols.push(Symbol {
                name: def.name.clone(),
                kind: def.kind,
                file: fe.id,
                line: def.line,
                block,
            });
            g.by_name.entry(def.name.clone()).or_default().push(id);
            g.by_block.entry(block).or_default().push(id);
            syms.push(id);
        }
        file_syms.insert(fe.id, syms);
    }

    // 2. Call edges: for each file, scan its blocks; attribute each
    //    `name(` occurrence to the enclosing definition.
    let by_name = &g.by_name;
    let sym_files: Vec<u32> = g.symbols.iter().map(|s| s.file).collect();
    let edge_lists: Vec<Vec<(u32, u32)>> = file_list
        .par_iter()
        .map(|(_, fe)| {
            let defs = &fe.defs;
            let syms = &file_syms[&fe.id];
            let mut local: Vec<(u32, u32)> = Vec::new();
            let mut di = 0usize;
            for &bid in &fe.blocks {
                let blk = &index.blocks[bid as usize];
                if blk.file == TOMBSTONE_FILE {
                    continue;
                }
                for (i, line) in blk.text.lines().enumerate() {
                    let lineno = blk.line + i as u32;
                    while di < defs.len() && defs[di].line <= lineno {
                        di += 1;
                    }
                    if di == 0 {
                        continue;
                    }
                    let cur = syms[di - 1];
                    for (tok, paren) in call_tokens(line) {
                        if !paren || tok.len() < 2 {
                            continue;
                        }
                        if let Some(targets) = by_name.get(tok) {
                            for t in resolve_call_targets(targets, fe.id, &sym_files) {
                                if t != cur {
                                    local.push((cur, t));
                                }
                            }
                        }
                    }
                }
            }
            local
        })
        .collect();
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    for l in edge_lists {
        edge_set.extend(l);
    }
    g.edges = edge_set.into_iter().collect();
    g.edges.sort_unstable();

    // 3. Adjacency.
    g.out_adj = vec![Vec::new(); g.symbols.len()];
    g.in_adj = vec![Vec::new(); g.symbols.len()];
    for &(a, b) in &g.edges {
        g.out_adj[a as usize].push(b);
        g.in_adj[b as usize].push(a);
    }

    // 4. File imports.
    let import_lists: Vec<Vec<u32>> = file_list
        .par_iter()
        .map(|(path, fe)| {
            let mut text = String::new();
            for &bid in &fe.blocks {
                text.push_str(&index.blocks[bid as usize].text);
                text.push('\n');
            }
            let from_rel = rel_of(&index.root, path);
            let mut out = Vec::new();
            for spec in extract_imports(&path.to_string_lossy(), &text) {
                out.extend(resolve_spec(
                    &spec, &from_rel, &rel_map, &stem_map, &dir_map,
                ));
            }
            out.sort_unstable();
            out.dedup();
            out.retain(|&id| id != fe.id);
            out
        })
        .collect();
    let mut fi: HashSet<(u32, u32)> = HashSet::new();
    for ((_, fe), ids) in file_list.iter().zip(import_lists) {
        for id in ids {
            fi.insert((fe.id, id));
        }
    }
    g.file_imports = fi.into_iter().collect();
    g.file_imports.sort_unstable();

    g
}

// ---------- queries ----------

impl Graph {
    pub fn find_all(&self, name: &str) -> Vec<u32> {
        self.by_name.get(name).cloned().unwrap_or_default()
    }

    pub fn callees(&self, id: u32) -> &[u32] {
        self.out_adj
            .get(id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn callers(&self, id: u32) -> &[u32] {
        self.in_adj
            .get(id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn call_edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Symbol names loosely matching `name` (for suggestions).
    pub fn suggest(&self, name: &str) -> Vec<String> {
        let lower = name.to_lowercase();
        let mut cands: Vec<String> = self
            .by_name
            .keys()
            .filter(|k| k.to_lowercase().contains(&lower))
            .take(6)
            .cloned()
            .collect();
        cands.sort();
        cands
    }

    /// Find up to `max_paths` call-graph paths from any source to any target,
    /// shortest first. `max_depth` caps path length.
    pub fn find_paths(
        &self,
        sources: &[u32],
        targets: &HashSet<u32>,
        max_depth: usize,
        max_paths: usize,
    ) -> Vec<Vec<u32>> {
        let mut results: Vec<Vec<u32>> = Vec::new();
        let mut budget: i64 = 200_000;
        for &s in sources {
            if results.len() >= max_paths || budget <= 0 {
                break;
            }
            let mut path = vec![s];
            self.dfs(
                s,
                targets,
                &mut path,
                &mut results,
                &mut budget,
                max_depth,
                max_paths,
            );
        }
        results.sort_by_key(|p| p.len());
        results.dedup();
        results.truncate(max_paths);
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        &self,
        cur: u32,
        targets: &HashSet<u32>,
        path: &mut Vec<u32>,
        results: &mut Vec<Vec<u32>>,
        budget: &mut i64,
        max_depth: usize,
        max_paths: usize,
    ) {
        if *budget <= 0 || results.len() >= max_paths || path.len() > max_depth {
            return;
        }
        *budget -= 1;
        if path.len() > 1 && targets.contains(&cur) {
            results.push(path.clone());
            return;
        }
        if path.len() >= max_depth {
            return;
        }
        let neighbors: Vec<u32> = self.out_adj[cur as usize].clone();
        for n in neighbors {
            if !path.contains(&n) {
                path.push(n);
                self.dfs(n, targets, path, results, budget, max_depth, max_paths);
                path.pop();
            }
        }
    }
}
