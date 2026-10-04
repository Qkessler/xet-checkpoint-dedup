//! `ckpt-dedup` measures how much of a model checkpoint Xet storage reuses when the next training step's checkpoint
//! is uploaded in full, and explains the result tensor by tensor.
//!
//! - [`layout`] parses safetensors headers into absolute tensor byte ranges.
//! - [`chunking`] runs Xet's content-defined chunker and keeps each chunk's position.
//! - [`overlap`] marks the next file's chunks as reused or new and attributes bytes to tensors.
//! - [`perturb`] writes synthetic perturbations of a bf16 checkpoint at a chosen change density.
//! - [`sweep`] measures chunk overlap against change density for scattered and clustered perturbations.
//! - [`results`] stamps and writes `results/<name>.json`.

pub mod chunking;
pub mod layout;
pub mod overlap;
pub mod perturb;
#[cfg(test)]
mod properties;
pub mod results;
pub mod sweep;
