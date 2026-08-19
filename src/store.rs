//! Disk cache: bincode-serialized index with a magic header, written
//! atomically (tmp file + rename) so a crash never corrupts the cache.

use crate::index::Index;
use std::fs;
use std::io;
use std::path::Path;

const MAGIC: &[u8; 10] = b"ENDEXIDX\x01\x00";

pub fn cache_path(root: &Path) -> std::path::PathBuf {
    root.join(".endex-index.bin")
}

pub fn save(index: &Index, root: &Path) -> io::Result<()> {
    let path = cache_path(root);
    let t0 = std::time::Instant::now();
    let mut data =
        bincode::serialize(index).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut buf = Vec::with_capacity(data.len() + MAGIC.len());
    buf.extend_from_slice(MAGIC);
    buf.append(&mut data);

    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &buf)?;
    fs::rename(&tmp, &path)?; // atomic on POSIX
    eprintln!(
        "  cache saved: {:.1} MB in {:?}",
        buf.len() as f64 / (1024.0 * 1024.0),
        t0.elapsed()
    );
    Ok(())
}

pub fn load(root: &Path) -> Option<Index> {
    let path = cache_path(root);
    let buf = fs::read(&path).ok()?;
    if buf.len() < MAGIC.len() || &buf[..MAGIC.len()] != MAGIC {
        return None; // unknown/corrupt cache
    }
    bincode::deserialize(&buf[MAGIC.len()..]).ok()
}
