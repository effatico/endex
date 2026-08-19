//! Recursive file watcher with debounce: bursts of events are collected and
//! forwarded as a single batch of unique paths after a short quiet period.

use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
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
