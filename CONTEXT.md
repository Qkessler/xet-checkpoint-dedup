# Checkpoint dedup probe

Measures how much of a model checkpoint Xet storage actually reuses when the next training step's checkpoint is uploaded in full, and explains the result tensor by tensor.

## Language

### Checkpoints

**Checkpoint**:
The complete set of model weights saved at one training step, stored as a safetensors file.
_Avoid_: snapshot, model file

**Checkpoint pair**:
Two checkpoints from the same run, a **base** (uploaded first) and a **next** (uploaded second, measured against the base).
_Avoid_: before/after, old/new

**Step gap**:
The number of optimizer steps between the base and the next checkpoint of a pair.

**Anchor**:
A full checkpoint uploaded as-is, in the sense the delta-weight-sync blog uses the word.
_Avoid_: full sync, keyframe

**Sparse delta**:
An encoding of only the elements that changed between two checkpoints (indices plus new values).
_Avoid_: patch, diff

### Model structure

**Tensor**:
One named n-dimensional array of values inside a checkpoint, with a dtype and a shape.
_Avoid_: weight (ambiguous with a single value), parameter

**Element**:
A single value inside a tensor.
_Avoid_: weight, parameter, value

**Layer**:
The group of tensors sharing one `model.layers.N.` prefix.

**Tensor kind**:
The role a tensor plays regardless of layer: embedding, attention projection (q, k, v, o), MLP projection (gate, up, down), norm, or output head.

### Measurements

**Unchanged element fraction**:
The share of a tensor's elements whose bytes are bit-identical between the base and the next checkpoint.
_Avoid_: sparsity (the blog's word, which also means other things)

**Chunk overlap**:
The share of the next checkpoint's bytes that fall in chunks whose hash also appears in the base checkpoint, before any upload policy is applied.
_Avoid_: dedup ratio, savings

**Transferred bytes**:
The bytes Xet's upload path reports as new for the next checkpoint after uploading the base to the same store.
_Avoid_: upload size, network bytes

**Defrag gap**:
Bytes counted in chunk overlap that the upload path still re-sends because defrag prevention refused the match.

**Attributed bytes**:
A tensor's share of a chunk, measured as the bytes of the chunk that lie inside that tensor's byte range.

### Perturbation sweep

**Change density**:
The fraction of a checkpoint's elements deliberately altered in a synthetic perturbation.

**Scattered perturbation**:
A perturbation whose altered elements are chosen uniformly at random across the whole checkpoint.

**Clustered perturbation**:
A perturbation with the same change density, but with the altered elements concentrated in contiguous rows or whole tensors.

### Xet storage

**Chunk**:
A content-defined slice of a file, about 64 KiB on average (8 to 128 KiB), identified by its hash.

**Xorb**:
A container object holding roughly a thousand individually compressed chunks, up to 64 MiB.
_Avoid_: shorb, block, blob

**Shard**:
A metadata object holding file recipes for many files plus an index of which chunk lives in which xorb.
_Avoid_: manifest

**Term**:
One entry in a file recipe: a xorb hash plus a contiguous chunk range inside it.
_Avoid_: segment, range

**Defrag prevention**:
The upload policy that refuses short dedup matches, and re-uploads those chunks instead, to keep the number of terms per file low so downloads stay fast.
