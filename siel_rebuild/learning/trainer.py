# -*- coding: utf-8 -*-

# ─── PATH FIX ────────────────────────────────────────────────────
import sys; from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

"""
Trainer v4.4 — COMPLETE REWRITE
- Binary bit prediction (8 outputs, not 256)
- MCTS planning in phase space
- Integrated attention mechanism
- Multi-domain chimera initialization
- Proper batching & curriculum learning
- All fixes from 10 deepseek iterations baked in
"""

import sys
import os
import time
import json
import random
import pickle
import hashlib
import gc
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from config import CONFIG
    from core.wave import WaveField
    from memory.concept_store import StreamingConceptStore
    from memory.episodic import EpisodicMemory
    from utils.logger import logger
except ImportError:
    try:
        from ..config import CONFIG
        from ..core.wave import WaveField
        from ..memory.concept_store import StreamingConceptStore
        from ..memory.episodic import EpisodicMemory
        from ..utils.logger import logger
    except ImportError:
        import importlib.util
        def import_from_file(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        config_path = _current_dir / 'config.py'
        CONFIG = import_from_file(config_path, 'config').CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM (INTEGRATED)
# ═══════════════════════════════════════════════════════════════════════════

class PhaseAttention(nn.Module):
    """Attention over oscillator phases."""
    
    def __init__(self, n_osc: int, n_heads: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_osc = n_osc
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, phases: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            phases: (B, n_osc) phase tensor
            features: (B, n_osc, dim) feature tensor
            
        Returns:
            output: (B, n_osc, dim) attended features
            attn_weights: (B, n_heads, n_osc, n_osc) attention weights
        """
        B, n_osc, dim = features.shape
        
        q = self.q_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, n_osc, dim)
        out = self.out_proj(out)
        
        return out, attn


class LocalPhaseAttention(nn.Module):
    """Localized attention only within oscillator neighborhoods."""
    
    def __init__(self, n_osc: int, neighbor_indices: List[List[int]], n_heads: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_osc = n_osc
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.neighbor_indices = neighbor_indices
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, phases: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Local attention within neighborhoods."""
        B, n_osc, dim = features.shape
        
        q = self.q_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(features).view(B, n_osc, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply local attention (only within neighborhoods)
        out = torch.zeros_like(q.transpose(1, 2)).unsqueeze(-1)
        attn_list = []
        
        for i in range(n_osc):
            neighbors = [i] + self.neighbor_indices[i]
            neighbors = [n for n in neighbors if 0 <= n < n_osc]
            
            q_i = q[:, :, i:i+1, :]
            k_neighbors = k[:, :, neighbors, :]
            v_neighbors = v[:, :, neighbors, :]
            
            scores = (q_i @ k_neighbors.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_list.append(attn)
            
            out_i = attn @ v_neighbors
            out = out_i
        
        out = out.squeeze(-2).transpose(1, 2).contiguous().view(B, n_osc, dim)
        out = self.out_proj(out)
        
        return out, None


# ═══════════════════════════════════════════════════════════════════════════
# MCTS — MONTE CARLO TREE SEARCH (FULLY FIXED)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSConfig:
    """Configuration for MCTS with sensible defaults."""
    n_simulations: int = 25
    exploration_constant: float = 1.414
    num_actions_per_dim: int = 8
    max_tree_size: int = 1000
    max_depth: int = 10
    perturbation_scale: float = 0.1
    use_phase_rotation: bool = True
    use_scaling: bool = True
    use_sinusoidal: bool = True
    use_bit_flip: bool = False


class MCTSNode:
    """
    Node in MCTS tree with memory optimization.
    Uses __slots__ for memory efficiency.
    """
    __slots__ = ['state', 'parent', 'children', 'visit_count', 'total_value', 
                 'action', 'prior_probability', 'depth']
    
    def __init__(
        self,
        state: torch.Tensor,
        parent: Optional['MCTSNode'] = None,
        children: Optional[List['MCTSNode']] = None,
        visit_count: int = 0,
        total_value: float = 0.0,
        action: Optional[int] = None,
        prior_probability: float = 0.0,
        depth: int = 0
    ):
        self.state = state
        self.parent = parent
        self.children = children or []
        self.visit_count = visit_count
        self.total_value = total_value
        self.action = action
        self.prior_probability = prior_probability
        self.depth = depth
    
    @property
    def value(self) -> float:
        """Mean value of the node."""
        return self.total_value / max(1, self.visit_count)
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound score."""
        if self.visit_count == 0:
            return float('inf')
        q_value = self.value
        parent_visits = self.parent.visit_count if self.parent else 1
        u_value = exploration_constant * self.prior_probability * math.sqrt(
            max(1, parent_visits) / (1 + self.visit_count)
        )
        return q_value + u_value
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None


# ═══════════════════════════════════════════════════════════════════════════
# MCTS — MONTE CARLO TREE SEARCH (FULLY FIXED)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSConfig:
    """Configuration for MCTS with sensible defaults."""
    n_simulations: int = 25
    exploration_constant: float = 1.414
    num_actions_per_dim: int = 8
    max_tree_size: int = 1000
    max_depth: int = 10
    perturbation_scale: float = 0.1
    use_phase_rotation: bool = True
    use_scaling: bool = True
    use_sinusoidal: bool = True
    use_bit_flip: bool = False


class MCTSNode:
    """
    Node in MCTS tree with memory optimization.
    Uses __slots__ for memory efficiency.
    """
    __slots__ = ['state', 'parent', 'children', 'visit_count', 'total_value', 
                 'action', 'prior_probability', 'depth']
    
    def __init__(
        self,
        state: torch.Tensor,
        parent: Optional['MCTSNode'] = None,
        children: Optional[List['MCTSNode']] = None,
        visit_count: int = 0,
        total_value: float = 0.0,
        action: Optional[int] = None,
        prior_probability: float = 0.0,
        depth: int = 0
    ):
        self.state = state
        self.parent = parent
        self.children = children or []
        self.visit_count = visit_count
        self.total_value = total_value
        self.action = action
        self.prior_probability = prior_probability
        self.depth = depth
    
    @property
    def value(self) -> float:
        """Mean value of the node."""
        return self.total_value / max(1, self.visit_count)
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound score."""
        if self.visit_count == 0:
            return float('inf')
        q_value = self.value
        parent_visits = self.parent.visit_count if self.parent else 1
        u_value = exploration_constant * self.prior_probability * math.sqrt(
            max(1, parent_visits) / (1 + self.visit_count)
        )
        return q_value + u_value
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None

# ═══════════════════════════════════════════════════════════════════════════
# MCTS 
# ═══════════════════════════════════════════════════════════════════════════
class MCTS:
    """
    Monte Carlo Tree Search for phase space optimization.
    
    Features:
    - Multi-dimensional phase exploration
    - Multiple perturbation types (rotation, scaling, sinusoidal, bit-flip)
    - Deterministic state hashing for caching
    - Batch search support
    - Memory-efficient with automatic cleanup
    - Metrics collection for debugging
    """
    
    def __init__(
        self, 
        forward_fn: Callable, 
        n_simulations: int = 25,
        exploration_constant: float = 1.414,
        max_tree_size: int = 1000,
        max_depth: int = 10,
        num_actions_per_dim: int = 8,
        perturbation_scale: float = 0.1,
        use_phase_rotation: bool = True,
        use_scaling: bool = True,
        use_sinusoidal: bool = True,
        use_bit_flip: bool = False,
    ):
        """
        Initialize MCTS with a forward function reference.
        
        Args:
            forward_fn: Callable that takes a state and returns logits
            n_simulations: Number of MCTS simulations per search
            exploration_constant: UCB exploration constant
            max_tree_size: Maximum tree size before pruning
            max_depth: Maximum search depth
            num_actions_per_dim: Number of actions per dimension
            perturbation_scale: Scale of perturbations
            use_phase_rotation: Enable phase rotation actions
            use_scaling: Enable phase scaling actions
            use_sinusoidal: Enable sinusoidal perturbation actions
            use_bit_flip: Enable bit-flip actions
        """
        self.forward_fn = forward_fn
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant
        self.max_tree_size = max_tree_size
        self.max_depth = max_depth
        self.num_actions_per_dim = num_actions_per_dim
        self.perturbation_scale = perturbation_scale
        self.use_phase_rotation = use_phase_rotation
        self.use_scaling = use_scaling
        self.use_sinusoidal = use_sinusoidal
        self.use_bit_flip = use_bit_flip
        
        self.root: Optional[MCTSNode] = None
        self._tree_size = 0
        self._simulation_cache: Dict[int, float] = {}
        self._metrics: Dict[str, List[float]] = {
            'tree_size': [],
            'max_depth': [],
            'avg_visit_count': [],
            'best_value': [],
            'cache_hits': [],
            'simulation_time': [],
        }
        self._last_search_depth = 0
        
    def search(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform MCTS search from given state.
        
        Args:
            state: Phase state tensor (n_dims,)
            
        Returns:
            Refined state tensor
        """
        # Clean up previous tree
        self._cleanup_tree()
        
        # Detach and clone to prevent gradient tracking
        state = state.detach().clone()
        
        self.root = MCTSNode(state=state)
        self._tree_size = 1
        self._last_search_depth = 0
        
        # Clear simulation cache for new search
        self._simulation_cache.clear()
        
        start_time = time.time()
        
        for sim in range(self.n_simulations):
            # Select a leaf node
            node = self._select(self.root)
            self._last_search_depth = max(self._last_search_depth, node.depth)
            
            # Handle leaf node
            if node.visit_count == 0:
                self._expand(node)
                if node.children:
                    # Select best child and simulate
                    child = self._select_child(node)
                    value = self._simulate(child)
                    self._backpropagate(child, value)
                else:
                    # No children possible (max depth reached)
                    value = self._simulate(node)
                    self._backpropagate(node, value)
            else:
                if node.children:
                    child = self._select_child(node)
                    value = self._simulate(child)
                    self._backpropagate(child, value)
                else:
                    self._expand(node)
                    if node.children:
                        child = self._select_child(node)
                        value = self._simulate(child)
                        self._backpropagate(child, value)
                    else:
                        value = self._simulate(node)
                        self._backpropagate(node, value)
        
        # Record metrics
        self._record_metrics()
        
        # Return best child state or original
        if self.root.children:
            # Use visit count for selection (more robust than value)
            best_child = max(self.root.children, key=lambda c: c.visit_count)
            result = best_child.state.detach().clone()
        else:
            result = state
        
        # Clean up tree after search
        self._cleanup_tree()
        
        # Record time
        self._metrics['simulation_time'].append(time.time() - start_time)
        
        return result
    
    def batch_search(self, states: torch.Tensor) -> torch.Tensor:
        """
        Perform MCTS on multiple states in parallel.
        
        Args:
            states: (B, n_dims) tensor
            
        Returns:
            (B, n_dims) refined states
        """
        results = []
        for i in range(states.shape[0]):
            results.append(self.search(states[i]))
        return torch.stack(results, dim=0)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select node using UCB with depth limit.
        
        Returns:
            Leaf node for expansion
        """
        if not node.children or node.depth >= self.max_depth:
            return node
        
        best_child = self._select_child(node)
        return self._select(best_child)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB."""
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.ucb_score(self.exploration_constant))
    
    def _expand(self, node: MCTSNode):
        """
        Expand node with multi-dimensional phase perturbations.
        
        Explores:
        - Phase rotation (shifts)
        - Phase scaling
        - Sinusoidal perturbations
        - Optional bit-flip operations
        """
        if self._tree_size >= self.max_tree_size:
            return
        
        n_dims = node.state.shape[0]
        phase_dims = min(n_dims, 8)  # Limit to 8 dims for efficiency
        
        action_types = []
        if self.use_phase_rotation:
            action_types.append('rotation')
        if self.use_scaling:
            action_types.append('scaling')
        if self.use_sinusoidal:
            action_types.append('sinusoidal')
        if self.use_bit_flip:
            action_types.append('bitflip')
        
        if not action_types:
            action_types = ['rotation']  # Fallback
        
        for dim in range(phase_dims):
            for action_idx in range(self.num_actions_per_dim):
                if self._tree_size >= self.max_tree_size:
                    return
                
                # Skip if this would exceed max depth
                if node.depth + 1 > self.max_depth:
                    return
                
                new_state = node.state.detach().clone()
                
                # Determine action type cyclically
                action_type = action_types[action_idx % len(action_types)]
                action_value = action_idx // len(action_types)
                
                if action_type == 'rotation':
                    # Phase rotation
                    delta = (action_value + 1) * (2 * np.pi / self.num_actions_per_dim)
                    new_state[dim] = (new_state[dim] + delta) % (2 * np.pi)
                    
                elif action_type == 'scaling':
                    # Phase scaling with continuity
                    scale = 1.0 + (action_value + 1) * 0.05 * self.perturbation_scale
                    new_state[dim] = (new_state[dim] * scale) % (2 * np.pi)
                    
                elif action_type == 'sinusoidal':
                    # Sinusoidal perturbation
                    perturb = 0.5 * np.sin(2 * np.pi * (action_value + 1) / self.num_actions_per_dim)
                    new_state[dim] = (new_state[dim] + perturb * self.perturbation_scale) % (2 * np.pi)
                    
                elif action_type == 'bitflip':
                    # Bit-flip style (quantized representation)
                    phase_value = (new_state[dim] / (2 * np.pi))
                    bit_value = int((phase_value * 255) % 256)
                    bit_value ^= (1 << (action_value % 8))
                    new_state[dim] = (bit_value / 255.0) * 2 * np.pi
                
                # Ensure valid range
                new_state = torch.remainder(new_state, 2 * np.pi)
                
                # Create child node
                child = MCTSNode(
                    state=new_state,
                    parent=node,
                    action=dim * self.num_actions_per_dim + action_idx,
                    prior_probability=1.0 / (phase_dims * self.num_actions_per_dim * len(action_types)),
                    depth=node.depth + 1
                )
                node.children.append(child)
                self._tree_size += 1
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node using the forward function.
        
        Returns:
            Reward value in [-1, 1]
        """
        # Check cache first
        state_hash = self._get_state_hash(node.state)
        if state_hash in self._simulation_cache:
            self._metrics['cache_hits'].append(1)
            return self._simulation_cache[state_hash]
        
        self._metrics['cache_hits'].append(0)
        
        with torch.no_grad():
            # Ensure proper tensor shape
            state = node.state.detach().clone()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            try:
                # Call the forward function (returns logits only)
                logits = self.forward_fn(state)
                
                # Handle different return types
                if isinstance(logits, tuple):
                    if len(logits) >= 2:
                        logits = logits[1]  # Second element is logits
                    else:
                        logits = logits[0]
                
                # Ensure logits is a tensor
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits, dtype=torch.float32)
                
                # Calculate reward from logits
                probs = torch.sigmoid(logits)
                
                # Use max confidence as primary reward
                confidence = torch.max(probs, dim=-1)[0]
                
                # Entropy penalty to encourage exploration
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                
                # Reward: high confidence with low entropy
                reward = confidence.mean().item() - 0.05 * entropy.mean().item()
                
                # Clamp to [-1, 1]
                reward = max(-1.0, min(1.0, reward))
                
            except Exception as e:
                # Fallback reward on error
                reward = 0.0
        
        # Cache result
        self._simulation_cache[state_hash] = reward
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value through the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent
    
    def _get_state_hash(self, state: torch.Tensor) -> int:
        """Get deterministic hash for state tensor."""
        if state.is_cuda:
            state = state.cpu()
        
        # Use float16 for consistent hashing
        state_f16 = state.to(torch.float16)
        data = state_f16.numpy().tobytes()
        return int(hashlib.md5(data).hexdigest(), 16) % (2**63)
    
    def _record_metrics(self):
        """Record search metrics."""
        if self.root:
            self._metrics['tree_size'].append(self._tree_size)
            self._metrics['max_depth'].append(self._last_search_depth)
            if self.root.children:
                visit_counts = [c.visit_count for c in self.root.children]
                values = [c.value for c in self.root.children]
                self._metrics['avg_visit_count'].append(sum(visit_counts) / len(visit_counts))
                self._metrics['best_value'].append(max(values))
            else:
                self._metrics['avg_visit_count'].append(0)
                self._metrics['best_value'].append(self.root.value)
    
    def _cleanup_tree(self):
        """Clean up tree to prevent memory leaks."""
        if self.root is not None:
            self._recursive_cleanup(self.root)
            self.root = None
            self._tree_size = 0
            self._simulation_cache.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _recursive_cleanup(self, node: MCTSNode):
        """Recursively cleanup nodes."""
        if node is None:
            return
        if node.children:
            for child in node.children:
                self._recursive_cleanup(child)
            node.children.clear()
        # Clear references
        node.parent = None
        if hasattr(node, 'state') and node.state is not None:
            node.state = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get MCTS performance metrics."""
        if not self._metrics['tree_size']:
            return {
                'tree_size': 0.0,
                'max_depth': 0.0,
                'avg_visit_count': 0.0,
                'best_value': 0.0,
                'cache_hit_rate': 0.0,
                'avg_simulation_time': 0.0,
            }
        
        total_sims = sum(1 for x in self._metrics.get('cache_hits', []) if x == 0)
        total_queries = len(self._metrics.get('cache_hits', [1]))
        cache_hit_rate = 1.0 - (total_sims / max(1, total_queries))
        
        return {
            'tree_size': float(sum(self._metrics['tree_size']) / len(self._metrics['tree_size'])),
            'max_depth': float(sum(self._metrics['max_depth']) / len(self._metrics['max_depth'])),
            'avg_visit_count': float(sum(self._metrics['avg_visit_count']) / max(1, len(self._metrics['avg_visit_count']))),
            'best_value': float(sum(self._metrics['best_value']) / max(1, len(self._metrics['best_value']))),
            'cache_hit_rate': float(cache_hit_rate),
            'avg_simulation_time': float(sum(self._metrics['simulation_time']) / max(1, len(self._metrics['simulation_time']))),
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        for key in self._metrics:
            self._metrics[key].clear()


# ═══════════════════════════════════════════════════════════════════════════
# STABILITY MONITORING
# ═══════════════════════════════════════════════════════════════════════════

class StabilityMonitor:
    """Monitor for numerical stability."""
    
    def __init__(self, max_violations: int = 5, grad_clip_norm: float = 1.0):
        self.violations = deque(maxlen=max_violations)
        self.gradient_norms = deque(maxlen=100)
        self.loss_components = deque(maxlen=100)
        self.grad_clip_norm = grad_clip_norm
    
    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.violations.append({'name': name, 'time': time.time()})
            return False
        return True
    
    def check_loss(self, loss: float, name: str) -> bool:
        if np.isnan(loss) or np.isinf(loss):
            self.violations.append({'name': f"loss_{name}", 'time': time.time()})
            return False
        return True
    
    def check_gradients(self, params: List) -> bool:
        for p in params:
            if hasattr(p, 'grad') and p.grad is not None:
                if not self.check_tensor(p.grad, "gradient"):
                    return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CACHING
# ═══════════════════════════════════════════════════════════════════════════

class DatasetCache:
    """Cache dataset to disk."""
    
    def __init__(self, cache_dir: str = "./siel_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset(self, dataset: List, val_dataset: List, name: str = "default") -> None:
        cache_file = self.cache_dir / f"dataset_{name}.pkl"
        data = {'train': dataset, 'val': val_dataset, 'time': time.time()}
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved dataset: {len(dataset)} train, {len(val_dataset)} val")
    
    def load_dataset(self, name: str = "default") -> Optional[Tuple[List, List]]:
        cache_file = self.cache_dir / f"dataset_{name}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded dataset: {len(data['train'])} train, {len(data['val'])} val")
                return data['train'], data['val']
        return None


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER v4.4 — COMPLETE
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """Trainer v4.4: Binary bit prediction with MCTS and attention."""
    
    def __init__(
        self,
        wave: WaveField,
        store: StreamingConceptStore,
        memory: EpisodicMemory,
        resume: bool = True,
        use_mcts: bool = True,
        use_attention: bool = True,
    ):
        self.wave = wave
        self.store = store
        self.memory = memory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n_osc = wave.grid_size * wave.grid_size
        grid_size = wave.grid_size
        
        # ─── CONFIG ────────────────────────────────────────────────────────
        self.learning_rate = 0.003
        self.batch_size = 32
        self.context_window = 64
        self.early_stopping_patience = 40
        self.mastery_threshold = 0.1
        
        # ─── COUPLING PARAMETERS ──────────────────────────────────────────
        self.k_coupling = 2.0
        self.noise_scale = 0.1
        self.dt = 0.01
        self.alpha = np.pi / 2.5
        self.cos_alpha = torch.tensor(np.cos(self.alpha), device=self.device)
        self.sin_alpha = torch.tensor(np.sin(self.alpha), device=self.device)
        
        # ─── LOSS WEIGHTS ─────────────────────────────────────────────────
        self.loss_weight_task = 1.0
        self.loss_weight_sync = 0.05
        self.loss_weight_coherence = 0.005
        self.loss_weight_entropy = 0.01
        
        # ─── STABILITY ─────────────────────────────────────────────────────
        self.stability = StabilityMonitor()
        self.dataset_cache = DatasetCache()
        
        # ─── SMART INITIALIZATION ────────────────────────────────────────
        smart_phases = self._smart_phase_init(grid_size)
        self.wave.inject_phase_map(smart_phases.cpu().numpy() if isinstance(smart_phases, torch.Tensor) else smart_phases)
        
        # ─── SPARSE COUPLING ──────────────────────────────────────────────
        edge_targets = []
        edge_neighbors = []
        edge_density = 0.35
        
        for y in range(grid_size):
            for x in range(grid_size):
                idx = y * grid_size + x
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if random.random() > edge_density:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid_size and 0 <= nx < grid_size:
                            edge_targets.append(idx)
                            edge_neighbors.append(ny * grid_size + nx)
        
        self.E = len(edge_targets)
        self.registered_edge_targets = torch.tensor(edge_targets, dtype=torch.long, device=self.device)
        self.registered_edge_neighbors = torch.tensor(edge_neighbors, dtype=torch.long, device=self.device)
        
        # ─── WEIGHTS ──────────────────────────────────────────────────────
        self.W_edges = nn.Parameter(torch.randn(self.E, device=self.device) * 0.01)
        
        # ─── READOUT LAYER ────────────────────────────────────────────────
        self.readout = nn.Sequential(
            nn.Linear(n_osc * 2, 512, device=self.device),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256, device=self.device),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8, device=self.device),
        )
        for layer in self.readout:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # ─── ATTENTION ────────────────────────────────────────────────────
        self.use_attention = use_attention
        if self.use_attention:
            # Attention operates on one projected global feature token. The
            # readout still receives the original per-oscillator features.
            self.attention_input = nn.Linear(n_osc * 2, 512, device=self.device)
            self.attention = PhaseAttention(n_osc=1, n_heads=8, dim=512, dropout=0.1).to(self.device)
            self.attention_output = nn.Linear(512, n_osc * 2, device=self.device)
        else:
            self.attention_input = None
            self.attention = None
            self.attention_output = None
        
        # ─── MCTS ─────────────────────────────────────────────────────────
        self.use_mcts = use_mcts
        if self.use_mcts:
            self.mcts = MCTS(forward_fn=self._forward_for_mcts, n_simulations=25)
        else:
            self.mcts = None
        
        # ─── OPTIMIZER ────────────────────────────────────────────────────
        all_params = [self.W_edges, self.readout]
        if self.attention:
            all_params += [self.attention_input, self.attention, self.attention_output]
        
        self.optimizer = torch.optim.Adam(
            [p for module in all_params for p in (module.parameters() if isinstance(module, nn.Module) else [module])],
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        # ─── DATASET ──────────────────────────────────────────────────────
        self.dataset = []
        self.val_dataset = []
        self.train_loader = None
        self.current_epoch = 0
        self.training = False
        
        # Load or generate dataset
        cached = self.dataset_cache.load_dataset()
        if cached and resume:
            self.dataset, self.val_dataset = cached
        else:
            self._generate_dataset()
            self.dataset_cache.save_dataset(self.dataset, self.val_dataset)
        
        self._prepare_dataloaders()
        
        # ─── METRICS ──────────────────────────────────────────────────────
        self.loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"[OK] Trainer initialized: {n_osc} oscillators, {self.E} edges, attention={use_attention}, mcts={use_mcts}")
    
    def _smart_phase_init(self, grid_size: int) -> np.ndarray:
        """Multi-domain chimera initialization."""
        phases = np.zeros((grid_size, grid_size))
        split_ratio = 0.6
        split_idx = int(grid_size * split_ratio)
        
        # Domain 1: Synchronized
        phases[:split_idx, :split_idx] = 0.0
        
        # Domain 2: Phase-shifted
        phases[:split_idx, split_idx:] = np.pi / 3
        
        # Domain 3: Random with smooth transitions
        rand_domain = np.random.uniform(0, 2*np.pi, (grid_size - split_idx, grid_size))
        for y in range(grid_size - split_idx):
            for x in range(grid_size):
                rand_domain[y, x] += 0.5 * np.sin(2 * np.pi * x / grid_size)
        phases[split_idx:, :] = rand_domain
        
        # Add noise
        phases += np.random.normal(0, 0.02, (grid_size, grid_size))
        
        return phases % (2 * np.pi)
    
    def _generate_dataset(self):
        """Generate dataset with proper context windows."""
        code_samples = [
            # Basics
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(a, b): return a * b",
            "def divide(a, b): return a / b if b != 0 else None",
            "def power(a, b): return a ** b",
            "def sqrt(x): return x ** 0.5",
            "def absolute(x): return x if x >= 0 else -x",
            "def is_even(x): return x % 2 == 0",
            "def is_odd(x): return x % 2 != 0",
            "def negate(x): return -x",
            "def increment(x): return x + 1",
            "def decrement(x): return x - 1",
            # Recursive
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            # Search
            "def binary_search(arr, target): left, right = 0, len(arr)-1; while left <= right: mid = (left+right)//2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid+1; else: right = mid-1; return -1",
            "def linear_search(arr, target): for i, x in enumerate(arr): if x == target: return i; return -1",
            # Sort snippets
            "def bubble_sort(arr): for i in range(len(arr)): for j in range(len(arr)-1-i): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]",
            "def quick_sort(arr): return [] if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quick_sort([x for x in arr[1:] if x > arr[0]])",
            # Data structures
            "class Node: def __init__(self, value): self.value = value; self.next = None",
            "def stack_push(stack, value): stack.append(value)",
            "def queue_pop(queue): return queue.pop(0) if queue else None",
            # String operations
            "def reverse_string(s): return s[::-1]",
            "def is_palindrome(s): return s == s[::-1]",
            "def count_chars(s): from collections import Counter; return Counter(s)",
            # Math
            "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
            "def lcm(a, b): return abs(a*b) // gcd(a, b)",
            "def is_prime(n): return all(n % i != 0 for i in range(2, int(n**0.5)+1)) if n > 1 else False",
        ]
        
        self.dataset = []
        for code in code_samples:
            bytes_data = code.encode('utf-8')
            for i in range(len(bytes_data) - self.context_window - 1):
                context = bytes_data[i:i+self.context_window]
                target_byte = bytes_data[i+self.context_window]
                context_tensor = self._bytes_to_tensor(context)
                target_tensor = self._target_to_bits(target_byte)
                self.dataset.append((context_tensor, target_tensor))
        
        random.shuffle(self.dataset)
        split = int(len(self.dataset) * 0.8)
        self.val_dataset = self.dataset[split:]
        self.dataset = self.dataset[:split]
        
        logger.info(f"[DATA] Generated {len(self.dataset)} train / {len(self.val_dataset)} val samples")
    
    def _prepare_dataloaders(self):
        """Create DataLoaders from dataset."""
        if not self.dataset:
            return
        
        train_contexts = torch.stack([s[0] for s in self.dataset])
        train_targets = torch.stack([s[1] for s in self.dataset])
        
        self.train_dataset = TensorDataset(train_contexts, train_targets)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
    
    def _bytes_to_tensor(self, bytes_data: bytes) -> torch.Tensor:
        """Convert bytes to phase tensor."""
        n_osc = self.wave.grid_size * self.wave.grid_size
        if len(bytes_data) >= n_osc:
            arr = np.frombuffer(bytes_data[:n_osc], dtype=np.uint8).astype(np.float32)
        else:
            arr = np.zeros(n_osc, dtype=np.float32)
            arr[:len(bytes_data)] = np.frombuffer(bytes_data, dtype=np.uint8).astype(np.float32)
        
        phases = (arr / 255.0) * 2 * np.pi
        return torch.tensor(phases, dtype=torch.float32)
    
    def _target_to_bits(self, target: int) -> torch.Tensor:
        """Convert byte (0-255) to 8-bit vector."""
        bits = [(target >> i) & 1 for i in range(8)]
        return torch.tensor(bits, dtype=torch.float32)
    
    def _forward_for_mcts(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass for MCTS (logits only)."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            B, n_osc = state.shape
            state = torch.remainder(state, 2 * np.pi)
            
            # Sparse coupling
            tgt_phases = state.index_select(1, self.registered_edge_targets)
            nb_phases = state.index_select(1, self.registered_edge_neighbors)
            
            diff = tgt_phases - nb_phases
            sin_diff = torch.sin(diff)
            cos_diff = torch.cos(diff)
            term = sin_diff * self.cos_alpha - cos_diff * self.sin_alpha
            
            weights = torch.clamp(self.W_edges, -0.5, 0.5).unsqueeze(0)
            coupling_per_edge = weights * term
            
            coupling = torch.zeros(B, n_osc, device=self.device)
            coupling.index_add_(1, self.registered_edge_targets, coupling_per_edge)
            coupling = self.k_coupling * coupling / max(1, self.E / n_osc)
            coupling = torch.clamp(coupling, -1.0, 1.0)
            
            # Phase update
            dtheta_dt = 1.0 + coupling
            new_phases = state + dtheta_dt * self.dt
            new_phases = torch.remainder(new_phases, 2 * np.pi)
            
            # Features
            sin_phases = torch.sin(new_phases)
            cos_phases = torch.cos(new_phases)
            features = torch.cat([sin_phases, cos_phases], dim=1)
            
            # Readout
            logits = self.readout(features)
        
        return logits
    
    def _forward(self, phases: torch.Tensor) -> Tuple:
        """Full forward pass with attention and MCTS."""
        if phases.dim() == 1:
            phases = phases.unsqueeze(0)
        
        B, n_osc = phases.shape
        phases = torch.remainder(phases, 2 * np.pi)
        
        # MCTS planning if training
        if self.use_mcts and self.training and self.mcts:
            refined_phases = []
            for b in range(B):
                refined = self.mcts.search(phases[b])
                refined_phases.append(refined)
            phases = torch.stack(refined_phases, dim=0)
        
        # Sparse coupling
        tgt_phases = phases.index_select(1, self.registered_edge_targets)
        nb_phases = phases.index_select(1, self.registered_edge_neighbors)
        
        diff = tgt_phases - nb_phases
        sin_diff = torch.sin(diff)
        cos_diff = torch.cos(diff)
        term = sin_diff * self.cos_alpha - cos_diff * self.sin_alpha
        
        weights = torch.clamp(self.W_edges, -0.5, 0.5).unsqueeze(0)
        coupling_per_edge = weights * term
        
        coupling = torch.zeros(B, n_osc, device=self.device)
        coupling.index_add_(1, self.registered_edge_targets, coupling_per_edge)
        coupling = self.k_coupling * coupling / max(1, self.E / n_osc)
        coupling = torch.clamp(coupling, -1.0, 1.0)
        
        # Phase update
        noise = torch.randn_like(phases) * self.noise_scale * 0.001 if self.training else 0
        dtheta_dt = 1.0 + coupling
        new_phases = phases + dtheta_dt * self.dt + noise
        new_phases = torch.remainder(new_phases, 2 * np.pi)
        
        # Features
        sin_phases = torch.sin(new_phases)
        cos_phases = torch.cos(new_phases)
        features = torch.cat([sin_phases, cos_phases], dim=1)
        
        # Attention
        if self.use_attention and self.attention:
            attention_features = self.attention_input(features).unsqueeze(1)
            attention_phases = new_phases.mean(dim=1, keepdim=True)
            attention_features, _ = self.attention(attention_phases, attention_features)
            features = self.attention_output(attention_features.squeeze(1))
        
        # Readout
        logits = self.readout(features)
        
        # Auxiliary losses
        cos_phases_mean = torch.mean(torch.cos(new_phases), dim=1, keepdim=True)
        sync_loss = torch.mean(1.0 - cos_phases_mean ** 2)
        sync_loss = torch.clamp(sync_loss, 0.0, 1.0)
        
        diff_flat = diff.view(-1)
        coherence_loss = torch.mean(torch.sin(diff_flat) ** 2)
        coherence_loss = torch.clamp(coherence_loss, 0.0, 1.0)
        
        probs = torch.sigmoid(logits)
        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        entropy_loss = torch.clamp(entropy_loss, 0.0, 1.0)
        
        return new_phases, logits, sync_loss, coherence_loss, entropy_loss
    
    def _train_step(self, batch: Tuple) -> Dict:
        """Single training step."""
        self.optimizer.zero_grad()
        self.training = True
        
        contexts, targets = batch
        contexts = contexts.to(self.device)
        targets = targets.to(self.device)
        
        try:
            new_phases, logits, sync_loss, coherence_loss, entropy_loss = self._forward(contexts)
            
            task_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            
            loss = (self.loss_weight_task * task_loss +
                   self.loss_weight_sync * sync_loss +
                   self.loss_weight_coherence * coherence_loss +
                   self.loss_weight_entropy * entropy_loss)
            
            if not self.stability.check_loss(loss.item(), "total"):
                self.optimizer.zero_grad()
                return {'loss': float('nan'), 'components': {}}
            
            loss.backward()
            
            all_params = [self.W_edges] + list(self.readout.parameters())
            if self.attention:
                all_params += (
                    list(self.attention_input.parameters()) +
                    list(self.attention.parameters()) +
                    list(self.attention_output.parameters())
                )
            
            if not self.stability.check_gradients(all_params):
                self.optimizer.zero_grad()
                return {'loss': float('nan'), 'components': {}}
            
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            self.optimizer.step()
            
            # Clamp weights
            self.W_edges.data = torch.clamp(self.W_edges.data, -0.5, 0.5)
            for param in self.readout.parameters():
                param.data = torch.clamp(param.data, -1.0, 1.0)
            
            self.training = False
            return {
                'loss': loss.item(),
                'components': {
                    'task': task_loss.item(),
                    'sync': sync_loss.item(),
                    'coherence': coherence_loss.item(),
                    'entropy': entropy_loss.item(),
                }
            }
        
        except Exception as e:
            logger.warning(f"Train step failed: {e}")
            self.optimizer.zero_grad()
            self.training = False
            return {'loss': float('nan'), 'components': {}}
    
    def _validate(self) -> Tuple[float, float]:
        """Validation pass."""
        if not self.val_dataset:
            return float('inf'), 0.0
        
        self.training = False
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0
        
        val_contexts = torch.stack([s[0] for s in self.val_dataset[:min(50, len(self.val_dataset))]])
        val_targets = torch.stack([s[1] for s in self.val_dataset[:min(50, len(self.val_dataset))]])
        
        val_contexts = val_contexts.to(self.device)
        val_targets = val_targets.to(self.device)
        
        with torch.no_grad():
            new_phases, logits, sync_loss, coherence_loss, entropy_loss = self._forward(val_contexts)
            task_loss = F.binary_cross_entropy_with_logits(logits, val_targets, reduction='mean')
            loss = (task_loss + 0.05 * sync_loss + 0.005 * coherence_loss + 0.01 * entropy_loss)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == val_targets).float().mean().item()
        
        self.training = True
        return loss.item(), acc
    
    def train(self, epochs: int = 100) -> Dict:
        """Main training loop."""
        logger.info(f"[START] Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Curriculum: anneal coupling
            progress = epoch / max(1, epochs)
            self.k_coupling = 2.0 * (1 - progress) + 0.3 * progress
            self.noise_scale = 0.1 * (1 - progress) + 0.005 * progress
            
            # Train epoch
            if not self.train_loader:
                return {'error': 'No training data'}
            
            total_loss = 0.0
            for batch in self.train_loader:
                result = self._train_step(batch)
                if np.isfinite(result['loss']):
                    total_loss += result['loss']
            
            train_loss = total_loss / max(1, len(self.train_loader))
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # LR schedule
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.3f}, lr={current_lr:.5f}")
            
            self.loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"[STOP] Early stopping at epoch {epoch+1}")
                    break
            
            # Mastery check
            if val_loss < self.mastery_threshold:
                logger.info(f"[MASTERY] MASTERY at epoch {epoch+1}!")
                return {
                    'epochs_completed': epoch + 1,
                    'final_loss': train_loss,
                    'best_val_loss': self.best_val_loss,
                    'mastery_achieved': True,
                    'accuracy': val_acc,
                }
        
        return {
            'epochs_completed': epoch + 1,
            'final_loss': train_loss,
            'best_val_loss': self.best_val_loss,
            'mastery_achieved': False,
            'accuracy': val_acc,
        }
    
    def get_stats(self) -> Dict:
        """Get training stats."""
        return {
            'epoch': self.current_epoch,
            'loss_history': self.loss_history[-20:],
            'val_loss_history': self.val_loss_history[-20:],
            'val_acc_history': self.val_acc_history[-20:],
            'best_val_loss': self.best_val_loss,
            'stability_violations': len(self.stability.violations),
        }
