"""
WaveField — Vectorized wave propagation with attention.
"""

import random
import math
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

from .oscillator import PhaseOscillator
from .coupling import TrainableCouplingMatrix

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG

try:
    from ..utils.cpu import cpu_saver
except ImportError:
    from utils.cpu import cpu_saver


class WaveField:
    """
    Wave field with vectorized propagation.
    Uses flat NumPy arrays for performance.
    """
    
    def __init__(self, grid_size: int = None, pin_count: int = None):
        self.grid_size = grid_size or CONFIG.GRID_SIZE
        self.pin_count = pin_count or CONFIG.PIN_COUNT
        
        # ─── VECTORIZED ARRAYS ────────────────────────────────────────────
        # Phase array (2D grid) — ALL vectorized operations
        self.phase_array = np.random.uniform(0, 2 * np.pi, (self.grid_size, self.grid_size))
        self.freq_array = np.random.uniform(
            *CONFIG.NATURAL_FREQ_RANGE, 
            (self.grid_size, self.grid_size)
        )
        self.amp_array = np.ones((self.grid_size, self.grid_size))
        self.resonance_array = np.zeros((self.grid_size, self.grid_size))
        
        # ─── OSCILLATOR OBJECTS ───────────────────────────────────────────
        # Keep for backward compatibility but use arrays for computation
        self.grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                osc = PhaseOscillator()
                osc.phase = self.phase_array[y, x]
                osc.natural_freq = self.freq_array[y, x]
                osc.amplitude = self.amp_array[y, x]
                self.grid[y, x] = osc
        
        # ─── COUPLING MATRIX ──────────────────────────────────────────────
        self.coupling_matrix = TrainableCouplingMatrix(self.grid_size)
        
        # ─── PINS ──────────────────────────────────────────────────────────
        self.pin_coords = self._generate_pin_coords()
        self.pin_index = {coord: i for i, coord in enumerate(self.pin_coords)}
        
        # ─── GOALS ─────────────────────────────────────────────────────────
        self.goal_attractors: Dict[Tuple[int, int], float] = {}
        
        # ─── ATTENTION ─────────────────────────────────────────────────────
        self.use_attention = CONFIG.ATTENTION_HEADS > 0
        self.attention = None
        self.hidden_states = None
        self.attention_mix = CONFIG.ATTENTION_MIX
        
        if self.use_attention:
            self._init_attention()
        
        # ─── CACHE ─────────────────────────────────────────────────────────
        self._cache_phases = None
        self._cache_step = -1
    
    def _init_attention(self):
        """Initialize attention module."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            class SimpleAttention(nn.Module):
                def __init__(self, num_oscillators: int, phase_dim: int, n_heads: int = 4):
                    super().__init__()
                    self.n_heads = n_heads
                    self.head_dim = phase_dim // n_heads
                    self.q_proj = nn.Linear(phase_dim, phase_dim)
                    self.k_proj = nn.Linear(phase_dim, phase_dim)
                    self.v_proj = nn.Linear(phase_dim, phase_dim)
                    self.out_proj = nn.Linear(phase_dim, phase_dim)
                    self.coupling_scale = nn.Parameter(torch.ones(1) * 0.1)
                
                def forward(self, phases: torch.Tensor, hidden: torch.Tensor):
                    B, N, D = hidden.shape
                    
                    q = self.q_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
                    k = self.k_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
                    v = self.v_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
                    
                    attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    attn = F.softmax(attn, dim=-1)
                    K_ij = attn.mean(dim=1)
                    
                    theta_i = phases.unsqueeze(2)
                    theta_j = phases.unsqueeze(1)
                    phase_diffs = torch.sin(theta_j - theta_i)
                    d_theta = self.coupling_scale * (K_ij * phase_diffs).sum(dim=-1)
                    
                    v_aggregated = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
                    updated_hidden = self.out_proj(v_aggregated)
                    
                    return d_theta, updated_hidden, K_ij
            
            self.attention = SimpleAttention(
                num_oscillators=self.grid_size * self.grid_size,
                phase_dim=CONFIG.PHASE_DIM,
                n_heads=CONFIG.ATTENTION_HEADS
            )
            self.hidden_states = np.random.randn(
                self.grid_size * self.grid_size, CONFIG.PHASE_DIM
            ).astype(np.float32) * 0.01
            
            print(f"🧠 Attention initialized: {CONFIG.ATTENTION_HEADS} heads")
            
        except ImportError:
            self.use_attention = False
            print("⚠️ PyTorch not available, attention disabled.")
    
    def _generate_pin_coords(self) -> List[Tuple[int, int]]:
        """Generate pin coordinates (evenly distributed)."""
        coords = []
        step = max(1, self.grid_size // int(math.sqrt(self.pin_count)))
        for y in range(0, self.grid_size, step):
            for x in range(0, self.grid_size, step):
                if len(coords) < self.pin_count:
                    y_offset = random.randint(-1, 1)
                    x_offset = random.randint(-1, 1)
                    coords.append((
                        min(max(y + y_offset, 0), self.grid_size - 1),
                        min(max(x + x_offset, 0), self.grid_size - 1)
                    ))
        return coords[:self.pin_count]
    
    def _sync_arrays_to_grid(self):
        """Sync phase_array back to oscillator objects (when needed)."""
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                osc = self.grid[y, x]
                osc.phase = self.phase_array[y, x]
                osc.amplitude = self.amp_array[y, x]
                osc.resonance_score = self.resonance_array[y, x]
    
    def _sync_grid_to_arrays(self):
        """Sync oscillator objects to arrays (when needed)."""
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                osc = self.grid[y, x]
                self.phase_array[y, x] = osc.phase
                self.freq_array[y, x] = osc.natural_freq
                self.amp_array[y, x] = osc.amplitude
                self.resonance_array[y, x] = osc.resonance_score
    
    def propagate(self, dt: float = 0.01, steps: int = 10):
        """VECTORIZED propagation using NumPy."""
        grid_size = self.grid_size
        
        for step in range(steps):
            cpu_saver.throttle("propagation")
            
            # ─── 1. Get coupling matrix ──────────────────────────────────
            # This is the expensive part — we need the full matrix
            W = self.coupling_matrix.U @ self.coupling_matrix.V.T
            W += self.coupling_matrix.local_bias
            
            # ─── 2. Compute phase differences (vectorized) ──────────────
            # phase_array: (grid_size, grid_size) → flatten to (N,)
            phases_flat = self.phase_array.flatten()
            N = len(phases_flat)
            
            # Broadcast: (N, 1) - (1, N) = (N, N)
            phase_diffs = phases_flat[:, None] - phases_flat[None, :]
            
            # ─── 3. Coupling effect (vectorized) ─────────────────────────
            # W: (N, N), sin(phase_diffs): (N, N)
            coupling_effect = np.sum(W * np.sin(phase_diffs), axis=1)
            
            # ─── 4. Update phases ──────────────────────────────────────────
            natural_freq_flat = self.freq_array.flatten()
            dphase = dt * (
                natural_freq_flat + 
                CONFIG.COUPLING_STRENGTH * coupling_effect / max(1, N)
            )
            
            # Update flat array
            phases_flat = (phases_flat + dphase) % (2 * np.pi)
            
            # Reshape back to 2D
            self.phase_array = phases_flat.reshape(grid_size, grid_size)
            
            # ─── 5. Goal attractors ──────────────────────────────────────
            if self.goal_attractors:
                for (y, x), target_phase in self.goal_attractors.items():
                    idx = y * grid_size + x
                    self.phase_array[y, x] += 0.1 * np.sin(target_phase - self.phase_array[y, x]) * dt
                    self.phase_array[y, x] %= (2 * np.pi)
            
            # ─── 6. Update resonance ──────────────────────────────────────
            self.resonance_array = self.amp_array * np.abs(np.sin(self.phase_array))
            
            # ─── 7. Sync to oscillators (for compatibility) ──────────────
            self._sync_arrays_to_grid()
    
    def propagate_hybrid(self, dt: float = 0.01, steps: int = 10):
        """Hybrid propagation (local + attention)."""
        if self.use_attention and self.attention is not None:
            try:
                import torch
                
                for step in range(steps):
                    cpu_saver.throttle("propagation")
                    
                    # ─── Local propagation ──────────────────────────────
                    self.propagate(dt * (1 - self.attention_mix), steps=1)
                    
                    # ─── Attention ───────────────────────────────────────
                    phases_flat = self.phase_array.flatten()
                    hidden_t = torch.tensor(self.hidden_states, dtype=torch.float32).unsqueeze(0)
                    phases_t = torch.tensor(phases_flat, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        d_theta, updated_hidden, _ = self.attention(phases_t, hidden_t)
                    
                    d_theta_np = d_theta.squeeze(0).numpy()
                    
                    # Apply attention update
                    phases_flat += dt * self.attention_mix * d_theta_np
                    phases_flat %= (2 * np.pi)
                    
                    # Reshape back
                    self.phase_array = phases_flat.reshape(self.grid_size, self.grid_size)
                    self.hidden_states = updated_hidden.squeeze(0).numpy()
                    
                    # Update resonance
                    self.resonance_array = self.amp_array * np.abs(np.sin(self.phase_array))
                    
                    # Sync
                    self._sync_arrays_to_grid()
                    
            except ImportError:
                self.propagate(dt, steps)
        else:
            self.propagate(dt, steps)
    
    def inject_phase(self, position: Tuple[int, int], phase: float):
        """Inject a phase at a specific position."""
        y, x = position
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
            self.phase_array[y, x] = phase % (2 * np.pi)
            self.resonance_array[y, x] = min(1.0, self.resonance_array[y, x] + 0.3)
            self.grid[y, x].phase = self.phase_array[y, x]
            self.grid[y, x].resonance_score = self.resonance_array[y, x]
    
    def inject_phase_map(self, phase_map: np.ndarray):
        """Inject a full phase map."""
        if phase_map.shape == (self.grid_size, self.grid_size):
            self.phase_array = phase_map % (2 * np.pi)
            self.resonance_array = self.amp_array * np.abs(np.sin(self.phase_array))
            self._sync_arrays_to_grid()
    
    def get_resonating_pins(self) -> List[Tuple[int, int]]:
        """Get all resonating pin positions."""
        return [
            (py, px) for py, px in self.pin_coords
            if self.resonance_array[py, px] > CONFIG.PIN_RESONANCE_THRESHOLD
        ]
    
    def set_goal(self, position: Tuple[int, int], target_phase: float):
        """Set a goal attractor."""
        self.goal_attractors[position] = target_phase % (2 * np.pi)
    
    def clear_goals(self):
        """Clear all goal attractors."""
        self.goal_attractors.clear()
    
    def get_wave_snapshot(self) -> np.ndarray:
        """Get a snapshot of the wave field."""
        return self.amp_array * np.sin(self.phase_array)
    
    def get_phases(self) -> np.ndarray:
        """Get current phase array."""
        return self.phase_array.copy()

    def compute_loss(self, target_phases: np.ndarray) -> float:
        """Compute circular loss between current and target phases."""
        current = self.phase_array.flatten()
        target = np.asarray(target_phases).flatten()

        n = min(len(current), len(target))
        if n == 0:
            return 0.0

        current = current[:n]
        target = target[:n]

        diff = np.abs(current - target) % (2 * np.pi)
        diff = np.minimum(diff, 2 * np.pi - diff)
        return float(np.mean(diff) / np.pi)
    
    def save_state(self) -> Dict:
        """Save the current wave state (for checkpointing)."""
        return {
            'phases': self.phase_array.copy(),
            'freqs': self.freq_array.copy(),
            'amps': self.amp_array.copy(),
            'resonance': self.resonance_array.copy(),
            'hidden': self.hidden_states.copy() if self.hidden_states is not None else None,
            'attractors': self.goal_attractors.copy(),
        }
    
    def restore_state(self, state: Dict):
        """Restore a saved wave state."""
        self.phase_array = state['phases'].copy()
        self.freq_array = state.get('freqs', self.freq_array).copy()
        self.amp_array = state.get('amps', self.amp_array).copy()
        self.resonance_array = state.get('resonance', self.resonance_array).copy()
        
        if state.get('hidden') is not None and self.hidden_states is not None:
            self.hidden_states = state['hidden'].copy()
        
        self.goal_attractors = state.get('attractors', {}).copy()
        self._sync_arrays_to_grid()
    
    def get_stats(self) -> Dict:
        """Get wave field statistics."""
        return {
            'phase_mean': float(np.mean(self.phase_array)),
            'phase_std': float(np.std(self.phase_array)),
            'resonance_mean': float(np.mean(self.resonance_array)),
            'resonance_max': float(np.max(self.resonance_array)),
            'active_pins': len(self.get_resonating_pins()),
            'total_oscillators': self.grid_size * self.grid_size,
        }