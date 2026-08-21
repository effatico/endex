//! Recursive file watcher with debounce: bursts of events are collected and
//! forwarded as a single batch of unique paths after a short quiet period.
//!
//! Also provides `Ignores`, the gitignore/hidden-file matcher that watcher
//! consumers must apply before indexing a changed path — the raw watcher
//! reports everything (including `target/`, `node_modules`, `.env`), while
//! `index::walk_files` filters those on full walks. Without this, ignored
//! files (potentially secrets) leak into the index via incremental updates.

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

const QUIET: Duration = Duration::from_millis(300);

/// Spawn a watcher on `root`. Returns a receiver that yields debounced
/// batches of changed paths.
pub fn watch(root: &std::path::Path) -> notify::Result<Receiver<Vec<PathBuf>>> {
    let (out_tx, out_rx) = mpsc::channel();
    let (event_tx, event_rx): (Sender<Event>, Receiver<Event>) = mpsc::channel();

    let mut watcher = RecommendedWatcher::new(
        move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                let _ = event_tx.send(event);
            }
        },
        NotifyConfig::default().with_poll_interval(Duration::from_secs(2)),
    )?;
    watcher.watch(root, RecursiveMode::Recursive)?;

    // Debounce thread: keep the watcher alive and coalesce events.
    std::thread::spawn(move || {
        let _watcher = watcher; // moved here so it lives as long as the thread
        loop {
            let mut dirty: Vec<PathBuf> = Vec::new();
            // Block until the first event of a new burst.
            match event_rx.recv() {
                Ok(ev) => dirty.extend(ev.paths),
                Err(_) => break, // watcher dropped
            }
            // Collect until quiet.
            while let Ok(ev) = event_rx.recv_timeout(QUIET) {
                dirty.extend(ev.paths);
            }
            dirty.sort_unstable();
            dirty.dedup();
            if !dirty.is_empty() && out_tx.send(dirty).is_err() {
                break;
            }
        }
    });

    Ok(out_rx)
}

// ---------- ignore matching for watcher events ----------

/// Mirrors the filtering `index::walk_files` applies on full walks: hidden
/// (dot) files/dirs, `.gitignore` files (root + nested + global) and
/// `.git/info/exclude`. Nested matchers are built lazily and cached.
pub struct Ignores {
    root: PathBuf,
    /// directory -> its .gitignore matcher (None = dir has none)
    matchers: HashMap<PathBuf, Option<Gitignore>>,
    global: Gitignore,
    /// `.gitignore` files are only honored inside a git repository — the
    /// same rule `index::walk_files` follows (`ignore` crate semantics).
    git_repo: bool,
}

impl Ignores {
    pub fn new(root: &Path) -> Self {
        let (global, _) = Gitignore::global();
        Ignores {
            root: root.to_path_buf(),
            matchers: HashMap::new(),
            global,
            git_repo: root.join(".git").exists(),
        }
    }

    fn matcher_for(&mut self, dir: &Path) -> Option<&Gitignore> {
        if !self.git_repo {
            return None;
        }
        if !self.matchers.contains_key(dir) {
            let mut b = GitignoreBuilder::new(dir);
            let mut any = false;
            let gi = dir.join(".gitignore");
            if gi.is_file() {
                b.add(gi);
                any = true;
            }
            // .git/info/exclude behaves like an extra root-level gitignore.
            if dir == self.root.as_path() {
                let excl = dir.join(".git/info/exclude");
                if excl.is_file() {
                    b.add(excl);
                    any = true;
                }
            }
            let built = if any { b.build().ok() } else { None };
            self.matchers.insert(dir.to_path_buf(), built);
        }
        self.matchers.get(dir).and_then(|m| m.as_ref())
    }

    /// True if `path` (under the root) must not be indexed. `is_dir` should
    /// reflect the path's type if known (affects `target/`-style patterns);
    /// pass false when unsure — directory patterns are also matched against
    /// parent components.
    pub fn is_ignored(&mut self, path: &Path, is_dir: bool) -> bool {
        let rel = match path.strip_prefix(&self.root) {
            Ok(r) => r,
            Err(_) => return false,
        };
        // Hidden components (covers .git, .env, .endex-* and any dot-dir).
        for c in rel.components() {
            if let std::path::Component::Normal(os) = c {
                if os.to_string_lossy().starts_with('.') {
                    return true;
                }
            }
        }
        // Global gitignore.
        if self
            .global
            .matched_path_or_any_parents(path, is_dir)
            .is_ignore()
        {
            return true;
        }
        // Root + nested .gitignore files, outermost first.
        let mut dirs = vec![self.root.clone()];
        let mut d = self.root.clone();
        let mut comps = rel.components().peekable();
        while let Some(c) = comps.next() {
            if comps.peek().is_none() {
                break; // last component is the path itself
            }
            if let std::path::Component::Normal(os) = c {
                d.push(os);
                dirs.push(d.clone());
            }
        }
        for dir in dirs {
            let ignored = self
                .matcher_for(&dir)
                .map(|g| g.matched_path_or_any_parents(path, is_dir).is_ignore())
                .unwrap_or(false);
            if ignored {
                return true;
            }
        }
        false
    }
}
