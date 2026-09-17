"""Island-based sampler with optional quality-diversity feature bins.

The sampler rotates selection across multiple islands, keeps a lightweight
archive for exploitation, and can maintain MAP-Elites-style feature maps to
encourage broader coverage during search.
"""
import ast
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class IslandSampler(BaseSampler):
    """Blend island evolution, archive reuse, and optional feature-map niches."""
    
    def __init__(
        self,
        num_islands: int = 5,
        migration_interval: int = 10,
        migration_rate: float = 0.1,
        exploration_ratio: float = 0.2,
        exploitation_ratio: float = 0.3,
        c: float = 1.414,
        feature_dimensions: Optional[List[str]] = None,
        feature_bins: int = 10,
    ):
        """
        Args:
            num_islands: Number of islands maintained by the sampler.
            migration_interval: Generations between migration events.
            migration_rate: Fraction of top nodes migrated to neighbors.
            exploration_ratio: Probability of uniform random sampling.
            exploitation_ratio: Probability of archive-based exploitation.
            c: Compatibility parameter kept for config parity with other samplers.
            feature_dimensions: Feature names used to build MAP-Elites bins.
            feature_bins: Number of bins per feature dimension.
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.c = c
        
        self.islands: List[Set[int]] = [set() for _ in range(num_islands)]
        self.island_generations: List[int] = [0] * num_islands
        self.island_best_nodes: List[Optional[int]] = [None] * num_islands
        
        self.feature_dimensions = feature_dimensions or []
        self.feature_bins = feature_bins
        self.island_feature_maps: List[Dict[Tuple, int]] = [{} for _ in range(num_islands)]
        
        self.feature_stats: Dict[str, Dict[str, Any]] = {}
        
        self.last_migration_generation = 0
        
        self.current_island = 0
        
        self.archive: Set[int] = set()
        self.archive_size = 100
        
        self.diversity_cache: Dict[int, Dict[str, Any]] = {}  # hash -> {"value": float, "timestamp": float}
        self.diversity_cache_size: int = 1000  # LRU cache size
        self.diversity_reference_set: List[str] = []  # Reference program codes for consistent diversity
        self.diversity_reference_size: int = 20  # Size of reference set
        
        self.all_nodes: Dict[int, "Node"] = {}  # node_id -> Node
    
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        """Sample nodes from the current island using mixed exploration modes."""
        if not nodes:
            return []
        
        for node in nodes:
            if node.id is not None:
                self.all_nodes[node.id] = node
        
        n = min(n, len(nodes))
        
        if self._should_migrate():
            self._migrate(nodes)
        
        if self.current_island >= self.num_islands:
            self.current_island = 0
        
        if len(self.island_generations) != self.num_islands:
            old_len = len(self.island_generations)
            if old_len < self.num_islands:
                self.island_generations.extend([0] * (self.num_islands - old_len))
                self.island_best_nodes.extend([None] * (self.num_islands - old_len))
                self.islands.extend([set() for _ in range(self.num_islands - old_len)])
            else:
                self.island_generations = self.island_generations[:self.num_islands]
                self.island_best_nodes = self.island_best_nodes[:self.num_islands]
                self.islands = self.islands[:self.num_islands]
        
        island_nodes = self._get_island_nodes(self.current_island, nodes)
        
        if not island_nodes:
            island_nodes = nodes
        
        self.island_generations[self.current_island] += 1
        
        self.current_island = (self.current_island + 1) % self.num_islands
        
        selected = []
        for _ in range(n):
            rand_val = random.random()
            
            if rand_val < self.exploration_ratio:
                node = self._sample_random(island_nodes)
            elif rand_val < self.exploration_ratio + self.exploitation_ratio:
                node = self._sample_from_archive(nodes)
                if node is None:
                    node = self._sample_weighted(island_nodes)
            else:
                node = self._sample_weighted(island_nodes)
            
            if node and node not in selected:
                selected.append(node)
                node.visit_count += 1
        
        return selected
    
    def sample_from_island(
        self, 
        island_id: int, 
        nodes: List["Node"], 
        n: int
    ) -> List["Node"]:
        """Sample nodes from a specific island without advancing the rotation."""
        island_id = island_id % self.num_islands
        island_nodes = self._get_island_nodes(island_id, nodes)
        
        if not island_nodes:
            island_nodes = nodes
        
        n = min(n, len(island_nodes))
        selected = []
        
        for _ in range(n):
            rand_val = random.random()
            
            if rand_val < self.exploration_ratio:
                node = self._sample_random(island_nodes)
            elif rand_val < self.exploration_ratio + self.exploitation_ratio:
                node = self._sample_from_archive(nodes)
                if node is None:
                    node = self._sample_weighted(island_nodes)
            else:
                node = self._sample_weighted(island_nodes)
            
            if node and node not in selected:
                selected.append(node)
                node.visit_count += 1
        
        return selected
    
    def on_node_added(self, node: "Node") -> None:
        """Register a new node with its island, archive, and feature map."""
        if node.id is None:
            return
        
        self.all_nodes[node.id] = node
        
        island_id = node.meta_info.get("island")
        
        if island_id is None:
            if node.parent and len(node.parent) > 0:
                island_id = self.current_island
            else:
                island_id = self.current_island
        
        island_id = island_id % self.num_islands
        node.meta_info["island"] = island_id
        
        if self.feature_dimensions:
            feature_coords = self._calculate_feature_coords(node)
            if feature_coords is not None:
                feature_key = tuple(feature_coords)
                feature_map = self.island_feature_maps[island_id]
                
                should_replace = feature_key not in feature_map
                
                if not should_replace:
                    existing_node_id = feature_map[feature_key]
                    if existing_node_id in self.all_nodes:
                        existing_node = self.all_nodes[existing_node_id]
                        if node.score > existing_node.score:
                            should_replace = True
                
                if should_replace:
                    feature_map[feature_key] = node.id
        
        self.islands[island_id].add(node.id)
        
        if self.island_best_nodes[island_id] is None:
            self.island_best_nodes[island_id] = node.id
        
        self._update_archive(node)
        
        if "diversity" in self.feature_dimensions:
            if len(self.diversity_reference_set) < self.diversity_reference_size:
                self.diversity_reference_set = []
    
    def on_node_removed(self, node: "Node") -> None:
        """Remove a node from island membership, caches, and archive state."""
        if node.id is None:
            return
        
        self.all_nodes.pop(node.id, None)
        
        island_id = node.meta_info.get("island")
        if island_id is not None and island_id < len(self.islands):
            self.islands[island_id].discard(node.id)
            
            if self.feature_dimensions:
                feature_coords = self._calculate_feature_coords(node)
                if feature_coords is not None:
                    feature_key = tuple(feature_coords)
                    feature_map = self.island_feature_maps[island_id]
                    if feature_key in feature_map and feature_map[feature_key] == node.id:
                        del feature_map[feature_key]
        
        self.archive.discard(node.id)
        
        if "diversity" in self.feature_dimensions:
            if node.code and node.code in self.diversity_reference_set:
                self.diversity_reference_set = []
    
    def _get_island_nodes(self, island_id: int, all_nodes: List["Node"]) -> List["Node"]:
        """Return nodes that currently belong to the requested island."""
        island_node_ids = self.islands[island_id]
        return [n for n in all_nodes if n.id in island_node_ids]
    
    def _sample_random(self, nodes: List["Node"]) -> Optional["Node"]:
        """Uniformly sample one node from the candidate list."""
        if not nodes:
            return None
        return random.choice(nodes)
    
    def _sample_weighted(self, nodes: List["Node"]) -> Optional["Node"]:
        """Sample one node proportionally to score with a small positive floor."""
        if not nodes:
            return None
        
        weights = []
        for node in nodes:
            weights.append(max(node.score, 0.001))
        
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(nodes)] * len(nodes)
        
        return random.choices(nodes, weights=weights, k=1)[0]
    
    def _sample_from_archive(self, all_nodes: List["Node"]) -> Optional["Node"]:
        """Sample one node from the archive when archived candidates exist."""
        if not self.archive:
            return None
        
        archive_nodes = [n for n in all_nodes if n.id in self.archive]
        if not archive_nodes:
            return None
        
        return random.choice(archive_nodes)
    
    def _update_archive(self, node: "Node") -> None:
        """Add a node to the archive set."""
        if node.id is None:
            return
        
        self.archive.add(node.id)
        
        if len(self.archive) > self.archive_size:
            pass
    
    def _calculate_feature_coords(self, node: "Node") -> Optional[List[int]]:
        """Project a node into discrete feature bins for the feature map."""
        if not self.feature_dimensions:
            return None
        
        coords = []
        for dim in self.feature_dimensions:
            if dim == "complexity":
                feature_value = len(node.code) if node.code else 0
            elif dim == "diversity":
                if len(self.all_nodes) < 2:
                    feature_value = 0.0
                else:
                    feature_value = self._get_cached_diversity(node)
            else:
                if not node.results or dim not in node.results:
                    return None
                
                feature_value = node.results[dim]
            
            if not isinstance(feature_value, (int, float)):
                return None
            
            self._update_feature_stats(dim, feature_value)
            
            scaled_value = self._scale_feature_value(dim, feature_value)
            
            bin_idx = int(scaled_value * self.feature_bins)
            bin_idx = max(0, min(self.feature_bins - 1, bin_idx))
            coords.append(bin_idx)
        
        return coords
    
    def _update_feature_stats(self, feature: str, value: float) -> None:
        """Track min/max statistics used to normalize feature values."""
        if feature not in self.feature_stats:
            self.feature_stats[feature] = {
                "min": value,
                "max": value,
                "values": []
            }
        
        stats = self.feature_stats[feature]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["values"].append(value)
        
        if len(stats["values"]) > 1000:
            stats["values"] = stats["values"][-1000:]
    
    def _scale_feature_value(self, feature: str, value: float) -> float:
        """Scale a feature value into the ``[0, 1]`` range via min-max stats."""
        if feature not in self.feature_stats:
            return 0.5
        
        stats = self.feature_stats[feature]
        min_val = stats["min"]
        max_val = stats["max"]
        
        if max_val - min_val < 1e-10:
            return 0.5
        
        scaled = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, scaled))
    
    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """Approximate code diversity using lightweight structural signals."""
        if code1 == code2:
            return 0.0
        
        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)
        
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)
        
        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))
        
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5
        
        return diversity
    
    def _get_cached_diversity(self, node: "Node") -> float:
        """Return a cached or freshly computed diversity score for a node."""
        if not node.code:
            return 0.0
        
        code_hash = hash(node.code)
        
        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]
        
        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()
        
        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != node.code:
                diversity_scores.append(self._fast_code_diversity(node.code, ref_code))
        
        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )
        
        self._cache_diversity_value(code_hash, diversity)
        
        return diversity
    
    def _update_diversity_reference_set(self) -> None:
        """Refresh the reference set used for stable diversity estimates."""
        if len(self.all_nodes) == 0:
            return
        
        all_programs = list(self.all_nodes.values())
        
        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs if p.code]
        else:
            selected = []
            remaining = all_programs.copy()
            
            if remaining:
                first_idx = random.randint(0, len(remaining) - 1)
                selected.append(remaining.pop(first_idx))
            
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1
                
                for i, candidate in enumerate(remaining):
                    if not candidate.code:
                        continue
                    min_div = float("inf")
                    for selected_prog in selected:
                        if not selected_prog.code:
                            continue
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)
                    
                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i
                
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))
            
            self.diversity_reference_set = [p.code for p in selected if p.code]
    
    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:
        """Store a diversity score in the LRU-style diversity cache."""
        if len(self.diversity_cache) >= self.diversity_cache_size:
            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]
        
        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}
    
    def _invalidate_diversity_cache(self) -> None:
        """Clear cached diversity scores and their reference set."""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
    
    def _should_migrate(self) -> bool:
        """Return whether the next migration event should be triggered."""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval
    
    def _migrate(self, all_nodes: List["Node"]) -> None:
        """Send top-performing migrants to neighboring islands."""
        if self.num_islands < 2:
            return
        
        self.last_migration_generation = max(self.island_generations)
        
        for island_id in range(self.num_islands):
            island_nodes = self._get_island_nodes(island_id, all_nodes)
            if not island_nodes:
                continue
            
            island_nodes.sort(key=lambda n: n.score, reverse=True)
            num_to_migrate = max(1, int(len(island_nodes) * self.migration_rate))
            migrants = island_nodes[:num_to_migrate]
            
            target_islands = [
                (island_id + 1) % self.num_islands,
                (island_id - 1) % self.num_islands,
            ]
            
            for migrant in migrants:
                if migrant.meta_info.get("migrant", False):
                    continue
                
                for target_island in target_islands:
                    target_nodes = self._get_island_nodes(target_island, all_nodes)
                    has_duplicate = any(n.code == migrant.code for n in target_nodes)
                    
                    if has_duplicate:
                        continue
                    
                    self.islands[target_island].add(migrant.id)
                    migrant.meta_info["migrant"] = True
    
    def get_island_stats(self, all_nodes: List["Node"]) -> Dict[str, Any]:
        """Summarize island populations, scores, and feature-map coverage."""
        stats = {
            "num_islands": self.num_islands,
            "island_populations": [],
            "island_generations": self.island_generations.copy(),
            "archive_size": len(self.archive),
            "current_island": self.current_island,
            "last_migration_generation": self.last_migration_generation,
            "feature_dimensions": self.feature_dimensions,
            "feature_stats": self.feature_stats.copy(),
        }
        
        for island_id in range(self.num_islands):
            island_nodes = self._get_island_nodes(island_id, all_nodes)
            feature_map_size = len(self.island_feature_maps[island_id]) if self.feature_dimensions else 0
            
            stats["island_populations"].append({
                "island_id": island_id,
                "size": len(island_nodes),
                "best_score": max((n.score for n in island_nodes), default=0.0),
                "avg_score": sum(n.score for n in island_nodes) / len(island_nodes) if island_nodes else 0.0,
                "feature_map_coverage": feature_map_size,
            })
        
        return stats
    
    def reset(self) -> None:
        """Reset transient sampler state while keeping configuration intact."""
        self.islands = [set() for _ in range(self.num_islands)]
        self.island_generations = [0] * self.num_islands
        self.island_best_nodes = [None] * self.num_islands
        self.island_feature_maps = [{} for _ in range(self.num_islands)]
        self.feature_stats.clear()
        self.last_migration_generation = 0
        self.current_island = 0
        self.archive.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Serialize sampler state for checkpointing."""
        return {
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "current_island": self.current_island,
            "archive": list(self.archive),
            "island_best_nodes": self.island_best_nodes,
            "island_feature_maps": [
                {str(k): v for k, v in feature_map.items()}
                for feature_map in self.island_feature_maps
            ],
            "feature_stats": self.feature_stats,
            "diversity_cache": {str(k): v for k, v in self.diversity_cache.items()},
            "diversity_reference_set": self.diversity_reference_set,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore sampler state from a serialized checkpoint."""
        loaded_generations = state.get("island_generations", [])
        loaded_best_nodes = state.get("island_best_nodes", [])
        
        if len(loaded_generations) != self.num_islands:
            if len(loaded_generations) < self.num_islands:
                loaded_generations.extend([0] * (self.num_islands - len(loaded_generations)))
                if len(loaded_best_nodes) < self.num_islands:
                    loaded_best_nodes.extend([None] * (self.num_islands - len(loaded_best_nodes)))
            else:
                loaded_generations = loaded_generations[:self.num_islands]
                loaded_best_nodes = loaded_best_nodes[:self.num_islands]
        
        self.island_generations = loaded_generations
        self.last_migration_generation = state.get("last_migration_generation", 0)
        self.current_island = state.get("current_island", 0) % self.num_islands
        self.archive = set(state.get("archive", []))
        self.island_best_nodes = loaded_best_nodes if len(loaded_best_nodes) == self.num_islands else [None] * self.num_islands
        
        loaded_feature_maps = state.get("island_feature_maps", [])
        if loaded_feature_maps and len(loaded_feature_maps) == self.num_islands:
            self.island_feature_maps = [
                {ast.literal_eval(k): v for k, v in feature_map.items()}
                for feature_map in loaded_feature_maps
            ]
        else:
            self.island_feature_maps = [{} for _ in range(self.num_islands)]
        
        self.feature_stats = state.get("feature_stats", {})
        
        loaded_diversity_cache = state.get("diversity_cache", {})
        self.diversity_cache = {int(k): v for k, v in loaded_diversity_cache.items()}
        self.diversity_reference_set = state.get("diversity_reference_set", [])
    
    def rebuild_from_nodes(self, nodes: List["Node"]) -> None:
        """Reconstruct island membership and feature maps from stored nodes."""
        self.islands = [set() for _ in range(self.num_islands)]
        self.island_feature_maps = [{} for _ in range(self.num_islands)]
        
        self.all_nodes = {node.id: node for node in nodes if node.id is not None}
        
        if "diversity" in self.feature_dimensions:
            self._invalidate_diversity_cache()
        
        for node in nodes:
            if node.id is None:
                continue
            
            island_id = node.meta_info.get("island")
            if island_id is not None:
                island_id = island_id % self.num_islands
                self.islands[island_id].add(node.id)
                
                if self.feature_dimensions:
                    feature_coords = self._calculate_feature_coords(node)
                    if feature_coords is not None:
                        feature_key = tuple(feature_coords)
                        self.island_feature_maps[island_id][feature_key] = node.id
                
                if node.id in self.archive or node.score > 0:
                    self._update_archive(node)
