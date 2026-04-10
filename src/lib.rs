/*!
# cuda-ghost-tiles

Ghost Tiles — learned sparse attention patterns.

In a sea of possible connections, most don't matter. Ghost tiles learn
which positions in an attention matrix are worth computing and which are
ghosts — present in the pattern but computationally absent.

This is how a fleet of agents decides what to pay attention to:
not everything, not randomly, but *learned* relevance.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tile in the attention grid
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tile {
    pub row: usize,
    pub col: usize,
    pub weight: f64,        // attention weight [0,1]
    pub is_active: bool,    // computed or ghosted
    pub usage_count: u32,
    pub last_used: u64,
}

/// Ghost tile pattern — sparse attention mask
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostPattern {
    pub id: String,
    pub grid_size: usize,
    pub tile_size: usize,
    pub tiles: Vec<Tile>,
    pub active_ratio: f64,  // fraction of tiles that are active
    pub sparsity_budget: f64, // max fraction that can be active
}

impl GhostPattern {
    pub fn new(id: &str, grid_size: usize, tile_size: usize, sparsity_budget: f64) -> Self {
        let tiles_per_dim = grid_size / tile_size;
        let mut tiles = vec![];
        for r in 0..tiles_per_dim {
            for c in 0..tiles_per_dim {
                tiles.push(Tile { row: r, col: c, weight: 0.5, is_active: true, usage_count: 0, last_used: 0 });
            }
        }
        GhostPattern { id: id.to_string(), grid_size, tile_size, tiles, active_ratio: 1.0, sparsity_budget: sparsity_budget.clamp(0.01, 1.0) }
    }

    /// Get active tiles
    pub fn active_tiles(&self) -> Vec<&Tile> {
        self.tiles.iter().filter(|t| t.is_active).collect()
    }

    /// Prune to sparsity budget — deactivate lowest-weight tiles
    pub fn prune(&mut self) {
        let max_active = (self.tiles.len() as f64 * self.sparsity_budget) as usize;
        if max_active >= self.tiles.len() { return; }

        // Sort by weight ascending, deactivate the lowest
        let mut indexed: Vec<(usize, f64)> = self.tiles.iter().enumerate().map(|(i, t)| (i, t.weight)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (idx, _) in indexed.iter().take(self.tiles.len() - max_active) {
            self.tiles[*idx].is_active = false;
        }

        self.active_ratio = self.tiles.iter().filter(|t| t.is_active).count() as f64 / self.tiles.len() as f64;
    }

    /// Record that a tile was used (increase its weight)
    pub fn use_tile(&mut self, row: usize, col: usize, reward: f64) {
        if let Some(tile) = self.tiles.iter_mut().find(|t| t.row == row && t.col == col) {
            tile.usage_count += 1;
            tile.last_used = now();
            tile.weight = (tile.weight + reward * 0.1).clamp(0.0, 1.0);
            tile.is_active = true;
        }
    }

    /// Decay unused tiles
    pub fn decay(&mut self, rate: f64) {
        let current = now();
        for tile in &mut self.tiles {
            let age = current.saturating_sub(tile.last_used);
            if age > 1000 {
                tile.weight *= 1.0 - rate;
            }
        }
    }

    /// Rebalance — prune and reactivate
    pub fn rebalance(&mut self) {
        self.decay(0.05);
        self.prune();
    }

    /// Efficiency score — attention computed / total possible
    pub fn efficiency(&self) -> f64 {
        1.0 - self.active_ratio
    }

    /// Coverage — what fraction of positions can reach an active tile
    pub fn coverage(&self) -> f64 {
        self.tiles.iter().filter(|t| t.is_active).count() as f64 / self.tiles.len() as f64
    }
}

/// Ghost tile manager for multiple patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostTileManager {
    pub patterns: HashMap<String, GhostPattern>,
    pub global_sparsity: f64,
}

impl GhostTileManager {
    pub fn new(sparsity: f64) -> Self { GhostTileManager { patterns: HashMap::new(), global_sparsity: sparsity } }

    pub fn add_pattern(&mut self, pattern: GhostPattern) {
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    /// Find most efficient pattern for a given attention task
    pub fn best_pattern(&self) -> Option<&GhostPattern> {
        self.patterns.values().max_by(|a, b| a.efficiency().partial_cmp(&b.efficiency()).unwrap())
    }

    /// Merge two patterns (union of active tiles)
    pub fn merge(&mut self, id_a: &str, id_b: &str, new_id: &str) -> Option<String> {
        let a = self.patterns.get(id_a)?;
        let b = self.patterns.get(id_b)?;
        let mut merged = GhostPattern::new(new_id, a.grid_size, a.tile_size, self.global_sparsity);

        // Union of active tiles from both
        for tile in &mut merged.tiles {
            let in_a = a.tiles.iter().find(|t| t.row == tile.row && t.col == tile.col).map(|t| t.is_active).unwrap_or(false);
            let in_b = b.tiles.iter().find(|t| t.row == tile.row && t.col == tile.col).map(|t| t.is_active).unwrap_or(false);
            let wa = a.tiles.iter().find(|t| t.row == tile.row && t.col == tile.col).map(|t| t.weight).unwrap_or(0.0);
            let wb = b.tiles.iter().find(|t| t.row == tile.row && t.col == tile.col).map(|t| t.weight).unwrap_or(0.0);
            tile.is_active = in_a || in_b;
            tile.weight = (wa + wb) / 2.0;
        }

        merged.prune();
        let name = new_id.to_string();
        self.patterns.insert(name.clone(), merged);
        Some(name)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_pattern() {
        let p = GhostPattern::new("p1", 64, 8, 0.5);
        assert_eq!(p.tiles.len(), 64); // 8x8 grid of 8x8 tiles
    }

    #[test]
    fn test_prune() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.25);
        // Set some weights low
        for (i, tile) in p.tiles.iter_mut().enumerate() {
            tile.weight = if i < 16 { 0.9 } else { 0.1 };
        }
        p.prune();
        let active = p.active_tiles().len();
        assert!(active <= 16); // 25% of 64 = 16
    }

    #[test]
    fn test_use_tile() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.5);
        p.use_tile(0, 0, 1.0);
        let tile = p.tiles.iter().find(|t| t.row == 0 && t.col == 0).unwrap();
        assert_eq!(tile.usage_count, 1);
        assert!(tile.weight > 0.5);
    }

    #[test]
    fn test_efficiency() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.5);
        assert_eq!(p.efficiency(), 0.0); // all active
        p.sparsity_budget = 0.5;
        p.prune();
        assert!(p.efficiency() > 0.0);
    }

    #[test]
    fn test_decay() {
        let mut p = GhostPattern::new("p1", 64, 8, 1.0);
        p.use_tile(0, 0, 1.0);
        let before = p.tiles[0].weight;
        // Set last_used to past
        p.tiles[0].last_used = 0;
        p.decay(0.5);
        assert!(p.tiles[0].weight < before);
    }

    #[test]
    fn test_merge() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut a = GhostPattern::new("a", 64, 8, 0.5);
        let mut b = GhostPattern::new("b", 64, 8, 0.5);
        // A activates top-left, B activates bottom-right
        for t in &mut a.tiles { if t.row < 4 { t.weight = 0.9; } else { t.weight = 0.1; } }
        for t in &mut b.tiles { if t.row >= 4 { t.weight = 0.9; } else { t.weight = 0.1; } }
        a.prune(); b.prune();
        mgr.add_pattern(a); mgr.add_pattern(b);
        let merged = mgr.merge("a", "b", "merged");
        assert!(merged.is_some());
    }

    #[test]
    fn test_best_pattern() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut p1 = GhostPattern::new("dense", 64, 8, 0.8);
        let mut p2 = GhostPattern::new("sparse", 64, 8, 0.2);
        p1.prune(); p2.prune();
        mgr.add_pattern(p1); mgr.add_pattern(p2);
        let best = mgr.best_pattern().unwrap();
        assert_eq!(best.id, "sparse"); // more efficient
    }

    #[test]
    fn test_rebalance() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.3);
        for t in &mut p.tiles { t.weight = 0.5; }
        p.rebalance(); // should prune + decay
        let active = p.active_tiles().len();
        assert!(active < p.tiles.len());
    }

    #[test]
    fn test_coverage() {
        let p = GhostPattern::new("p1", 64, 8, 1.0);
        assert!((p.coverage() - 1.0).abs() < 0.01);
    }
}
