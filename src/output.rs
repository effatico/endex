//! CLI output helpers: hit printing with case-insensitive match highlighting.
//!
//! The subtle part is `match_ranges`: lowercasing can CHANGE BYTE LENGTH
//! (e.g. 'İ' U+0130 lowercases to "i̇" — 2 bytes become 3), so byte offsets
//! found in `line.to_lowercase()` must never be used to slice `line` itself
//! (doing so panics on non-char-boundary cuts). We build the lowercased
//! string together with a map back to original byte offsets, and slice only
//! at those mapped positions.

use crate::index::Index;
use crate::search;
use std::io::{self, Write};
use std::time::Duration;

/// Byte ranges (in the ORIGINAL `line`) of case-insensitive matches of `q`
/// (already lowercased). Every returned range lies on char boundaries.
pub fn match_ranges(line: &str, q: &str) -> Vec<(usize, usize)> {
    if q.is_empty() {
        return Vec::new();
    }
    // Lowercased text + map: lowercased byte offset -> original byte offset.
    // One map entry PER BYTE of the lowercased string (a single char may
    // lowercase into a multi-byte sequence, e.g. 'İ' -> "i̇").
    let mut lower = String::with_capacity(line.len());
    let mut map: Vec<usize> = Vec::with_capacity(line.len() + 1);
    let mut buf = [0u8; 4];
    for (i, ch) in line.char_indices() {
        for lc in ch.to_lowercase() {
            lower.push_str(lc.encode_utf8(&mut buf));
            for _ in 0..lc.len_utf8() {
                map.push(i);
            }
        }
    }
    map.push(line.len());
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (pos, _) in lower.match_indices(q) {
        let s = map[pos];
        let e = map[(pos + q.len()).min(map.len() - 1)];
        if e <= s {
            continue;
        }
        if let Some(last) = ranges.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e); // merge adjacent/overlapping
                continue;
            }
        }
        ranges.push((s, e));
    }
    ranges
}

pub fn print_hits(
    idx: &Index,
    hits: &[search::Hit],
    query: &str,
    limit: usize,
    search_time: Duration,
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

    for hit in hits {
        println!("\x1b[1;36m{}:{}\x1b[0m", idx.path_of(hit.file_id), hit.line);
        print_block_matches(&hit.text, hit.line, &q, 6);
    }
}

/// `reranked` labels the score column: after reranking these are provider
/// relevance scores, otherwise RRF fusion weights — different scales, so
/// say which one the reader is looking at.
pub fn print_ask_hits(idx: &Index, hits: &[(f32, search::Hit)], query: &str, reranked: bool) {
    let q = query.to_lowercase();
    let label = if reranked { "relevance" } else { "rrf" };
    for (score, hit) in hits {
        println!(
            "\x1b[1;36m{}:{}\x1b[0m  \x1b[2m{label} {score:.4}\x1b[0m",
            idx.path_of(hit.file_id),
            hit.line
        );
        print_block_matches(&hit.text, hit.line, &q, 4);
    }
}

/// Print the matching lines of a block with line numbers and highlights.
pub fn print_block_matches(text: &str, start_line: u32, q: &str, max_lines: usize) {
    let mut stdout = io::stdout().lock();
    let mut shown = 0;
    for (lineno, line) in (start_line..).zip(text.lines()) {
        let l = lineno;
        let ranges = match_ranges(line, q);
        if ranges.is_empty() {
            continue;
        }
        if shown == max_lines {
            let _ = writeln!(stdout, "\x1b[2m  ··· (more matches in this block)\x1b[0m");
            break;
        }
        let _ = writeln!(stdout);
        let _ = write!(stdout, "  \x1b[2m{l:>5}|\x1b[0m ");
        let mut start = 0usize;
        for (s, e) in ranges {
            let _ = write!(stdout, "{}", &line[start..s]);
            let _ = write!(stdout, "\x1b[1;31m{}\x1b[0m", &line[s..e]);
            start = e;
        }
        let _ = writeln!(stdout, "{}", &line[start..]);
        shown += 1;
    }
    let _ = stdout.flush();
}
