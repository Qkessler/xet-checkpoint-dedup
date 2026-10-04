//! Content-defined chunking with Xet's own chunker, keeping each chunk's position in the file.
//!
//! The input is fed to [`Chunker`] in 8 MiB blocks and closed with [`Chunker::finish`], the same way xet-core's
//! `chunk` example and upload path do, so boundaries and hashes match what an upload would produce.

use std::collections::HashSet;

use xet_core_structures::merklehash::MerkleHash;
use xet_data::deduplication::Chunker;
use xet_data::deduplication::constants::TARGET_CHUNK_SIZE;

const INGESTION_BLOCK_BYTES: usize = 8 * 1024 * 1024;

/// Largest chunk the chunker emits (twice the 64 KiB target). A chunk this size ended at a forced cut, not at a
/// content-defined boundary.
pub const MAXIMUM_CHUNK_BYTES: u64 = 128 * 1024;

/// One chunk of a file: its absolute byte range and its Xet chunk hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecord {
    pub start_byte: u64,
    pub bytes: u64,
    pub hash: MerkleHash,
}

impl ChunkRecord {
    /// Absolute offset one past the chunk's last byte.
    pub fn end_byte(&self) -> u64 {
        self.start_byte + self.bytes
    }
}

/// Splits `data` into Xet chunks at the default 64 KiB target. The records tile `data` exactly, in order.
pub fn chunk_file(data: &[u8]) -> Vec<ChunkRecord> {
    let mut chunker = Chunker::new(*TARGET_CHUNK_SIZE);
    let mut records = Vec::with_capacity(data.len() / *TARGET_CHUNK_SIZE + 1);
    let mut cursor = 0u64;
    let mut push = |hash: MerkleHash, bytes: usize| {
        records.push(ChunkRecord {
            start_byte: cursor,
            bytes: bytes as u64,
            hash,
        });
        cursor += bytes as u64;
    };
    for block in data.chunks(INGESTION_BLOCK_BYTES) {
        for chunk in chunker.next_block(block, false) {
            push(chunk.hash, chunk.data.len());
        }
    }
    if let Some(chunk) = chunker.finish() {
        push(chunk.hash, chunk.data.len());
    }
    records
}

/// The set of chunk hashes in a file, for membership tests of another file's chunks.
pub fn hash_set(chunks: &[ChunkRecord]) -> HashSet<MerkleHash> {
    chunks.iter().map(|chunk| chunk.hash).collect()
}

/// Formats chunks one per line as `<hash> <bytes>`, the format of xet-core's `chunk` example and the spec
/// reference `.chunks` files.
pub fn format_chunk_lines(chunks: &[ChunkRecord]) -> String {
    chunks.iter().map(|chunk| format!("{} {}\n", chunk.hash, chunk.bytes)).collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::results::sha256_hex;

    const REFERENCE_CSV_SHA256: &str = "f41255b252f776125f2b657136654e4d4d5d2ccf8ef4db0ec186bd1981b69734";
    const REFERENCE_CHUNKS_SHA256: &str = "b20bb4b910bab5d67a4082313cf83afea201747f02e0aedc4158a333ba61c684";

    fn cached_input(key: &str, file: &str) -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".cache/xet-checkpoint-dedup/inputs")
            .join(key)
            .join(file);
        path.exists().then_some(path)
    }

    #[test]
    fn chunks_tile_the_input() {
        let mut data = vec![0u8; 3 * 1024 * 1024 + 17];
        ChaCha8Rng::seed_from_u64(1).fill_bytes(&mut data);
        let chunks = chunk_file(&data);
        let mut cursor = 0;
        for chunk in &chunks {
            assert_eq!(chunk.start_byte, cursor);
            assert!(chunk.bytes <= MAXIMUM_CHUNK_BYTES);
            cursor = chunk.end_byte();
        }
        assert_eq!(cursor, data.len() as u64);
    }

    #[test]
    fn maximum_chunk_bytes_matches_xet_constants() {
        use xet_data::deduplication::constants::MAXIMUM_CHUNK_MULTIPLIER;
        assert_eq!(MAXIMUM_CHUNK_BYTES as usize, *TARGET_CHUNK_SIZE * *MAXIMUM_CHUNK_MULTIPLIER);
    }

    #[test]
    fn empty_input_has_no_chunks() {
        assert!(chunk_file(&[]).is_empty());
    }

    /// Oracle 1: the spec reference `.chunks` file, reproduced byte for byte. Skipped until `just fetch` has cached
    /// the reference files.
    #[test]
    fn reproduces_spec_reference_chunks() {
        let (Some(csv_path), Some(chunks_path)) = (
            cached_input("spec-reference-csv", "Electric_Vehicle_Population_Data_20250917.csv"),
            cached_input("spec-reference-chunks", "Electric_Vehicle_Population_Data_20250917.csv.chunks"),
        ) else {
            eprintln!("skipping golden chunker test: run `just fetch` to cache the spec reference files");
            return;
        };
        let csv = std::fs::read(csv_path).unwrap();
        let expected = std::fs::read(chunks_path).unwrap();
        assert_eq!(sha256_hex(&csv), REFERENCE_CSV_SHA256);
        assert_eq!(sha256_hex(&expected), REFERENCE_CHUNKS_SHA256);
        assert_eq!(format_chunk_lines(&chunk_file(&csv)).as_bytes(), expected.as_slice());
    }
}
