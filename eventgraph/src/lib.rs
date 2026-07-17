//! eventgraph: feeds -> narrative extraction -> Postgres reference model +
//! DuckLake analytical facts. See README.md for the three-plane architecture.

pub mod model;
pub mod util;
pub mod feed;
pub mod extract;
pub mod normalize;
pub mod pg;
pub mod lake;
pub mod pipeline;
pub mod scrape;

pub use model::*;
