# cuda-ghost-tiles

Learned sparse attention — ghost tiles prune unnecessary positions while preserving relevance (Rust)

Part of the Cocapn fleet — a Lucineer vessel component.

## What It Does

### Key Types

- `GhostTile` — core data structure
- `GhostPattern` — core data structure
- `GhostTileManager` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-ghost-tiles.git
cd cuda-ghost-tiles

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_ghost_tiles::*;

// See src/lib.rs for full API
// 20 unit tests included
```

### Available Implementations

- `GhostTile` — see source for methods
- `GhostPattern` — see source for methods
- `GhostTileManager` — see source for methods

## Testing

```bash
cargo test
```

20 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: other
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates


## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
