# cuda-ghost-tiles

**Learned sparse attention patterns for efficient agent cognition.**

> In a sea of possible connections, most don't matter.
> Ghost tiles learn which positions in an attention matrix are worth computing.

## The Problem

Full attention is O(n^2). For a fleet of agents processing long contexts, this becomes the bottleneck. Most attention positions contribute nothing.

## The Solution

Ghost tiles learn which positions *actually matter*. Unimportant tiles become "ghosts" - present in the logical pattern but computationally absent.

### How It Works

1. **Tile Grid**: Divide attention matrix into 8x8 tiles
2. **Weight Learning**: Track which tiles get used, strengthen active ones
3. **Pruning**: Deactivate lowest-weight tiles to meet sparsity budget
4. **Decay**: Unused tiles lose weight over time
5. **Merge**: Combine complementary patterns for multi-task agents
6. **CUDA Mapping**: Each tile = one GPU thread block

### Available In

- **Rust** (`cuda-ghost-tiles`) - Primary implementation
- **C** (`ghost-tiles-c`) - Zero deps, ARM64/WASM portable, 12/12 tests pass
- **C++** (`ghost-tiles-cpp`) - Modern C++17 with RAII and ranges
- **C#** (`ghost-tiles-csharp`) - .NET 8 with LINQ
- **CUDA** (`ghost-tiles-cuda`) - GPU kernels for sparse attention

## Ecosystem Integration

- `cuda-attention` - Uses ghost tiles for saliency scoring
- `cuda-memory-fabric` - Tiles stored as procedural memory
- `cuda-emergence` - Sparse patterns detected as emergent behavior
- `cuda-voxel-logic` - 2D attention tiles generalize to 3D spatial tiles

## See Also

- [cuda-attention](https://github.com/Lucineer/cuda-attention) - Attention mechanisms
- [cuda-emergence](https://github.com/Lucineer/cuda-emergence) - Emergent pattern detection
- [ghost-tiles-c](https://github.com/Lucineer/ghost-tiles-c) - C implementation
- [ghost-tiles-cuda](https://github.com/Lucineer/ghost-tiles-cuda) - CUDA kernels

## License

MIT OR Apache-2.0