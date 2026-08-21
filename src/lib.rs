//! endex: fast, cached, self-updating code indexer with millisecond
//! substring search, a code knowledge graph, and hybrid semantic search.
//!
//! The crate is organized as a library (this module) plus a thin CLI binary
//! (`src/main.rs`) so the indexing and search machinery can be reused and
//! tested independently.

pub mod embed;
pub mod graph;
pub mod index;
pub mod mcp;
pub mod output;
pub mod search;
pub mod store;
pub mod watch;

pub use index::Index;
