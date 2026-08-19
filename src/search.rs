//! Query execution: trigram intersection for queries >= 3 bytes,
//! parallel full scan fallback for shorter queries.

use crate::index::{Index, TOMBSTONE_FILE};
use rayon::prelude::*;

pub struct Hit {
    pub file_id: u32,
    pub line: u32,
    pub text: String,
    pub occurrences: usize,
}

/// Search for `query` (case-insensitive substring) and return up to `limit`
/// hits, ranked by (occurrences desc, block size asc).
pub fn search(index: &Index, query: &str, limit: usize) -> Vec<Hit> {
    let q = query.to_lowercase();
    let qb = q.as_bytes();
    if qb.is_empty() {
        return Vec::new();
    }

    let mut hits: Vec<Hit> = if qb.len() >= 3 {
        trigram_search(index, &q)
    } else {
        scan_search(index, &q)
    };

    hits.sort_unstable_by(|a, b| {
        b.occurrences
            .cmp(&a.occurrences)
            .then(a.text.len().cmp(&b.text.len()))
    });
    hits.truncate(limit);
    hits
}

fn count_occurrences(lower: &str, q: &str) -> usize {
    lower.match_indices(q).count()
}

/// Trigram-indexed search: intersect posting lists, then verify candidates.
fn trigram_search(index: &Index, q: &str) -> Vec<Hit> {
    let qb = q.as_bytes();

    // Unique trigrams of the query.
    let mut tris: Vec<u32> = Vec::with_capacity(qb.len());
    for w in qb.windows(3) {
        let t = (w[0] as u32) << 16 | (w[1] as u32) << 8 | w[2] as u32;
        if !tris.contains(&t) {
            tris.push(t);
        }
    }

    // Posting lists sorted by length (rarest first).
    let mut lists: Vec<&Vec<u32>> = Vec::with_capacity(tris.len());
    for t in &tris {
        match index.postings.get(t) {
            Some(v) => lists.push(v),
            // A trigram that doesn't exist anywhere => no results at all.
            None => return Vec::new(),
        }
    }
    lists.sort_unstable_by_key(|v| v.len());

    // Intersect, starting from the rarest list.
    let mut candidates: Vec<u32> = lists[0].clone();
    for list in &lists[1..] {
        if candidates.is_empty() {
            break;
        }
        intersect_sorted(&mut candidates, list);
    }

    // Verify candidates in parallel (guards against false positives from
    // trigrams that were split by block boundaries or case folding).
    candidates
        .par_iter()
        .filter_map(|&bid| {
            let blk = &index.blocks[bid as usize];
            if blk.file == TOMBSTONE_FILE {
                return None;
            }
            let lower = blk.text.to_lowercase();
            let occ = count_occurrences(&lower, q);
            if occ == 0 {
                None
            } else {
                Some(Hit {
                    file_id: blk.file,
                    line: blk.line,
                    text: blk.text.clone(),
                    occurrences: occ,
                })
            }
        })
        .collect()
}

/// Fallback for 1-2 byte queries: parallel scan of all live blocks.
fn scan_search(index: &Index, q: &str) -> Vec<Hit> {
    index
        .blocks
        .par_iter()
        .filter_map(|blk| {
            if blk.file == TOMBSTONE_FILE {
                return None;
            }
            let lower = blk.text.to_lowercase();
            let occ = count_occurrences(&lower, q);
            if occ == 0 {
                None
            } else {
                Some(Hit {
                    file_id: blk.file,
                    line: blk.line,
                    text: blk.text.clone(),
                    occurrences: occ,
                })
            }
        })
        .collect()
}

/// In-place intersection of two sorted vecs.
fn intersect_sorted(a: &mut Vec<u32>, b: &[u32]) {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    *a = out;
}
