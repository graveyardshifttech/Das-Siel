"""
TrainableCouplingMatrix — Adaptive rank coupling matrix.
"""

import numpy as np
from typing import Optional, List, Dict
from collections import deque

# Import CONFIG robustly in different runtime contexts.
try:
    # Preferred when running from the project root (parent dir on sys.path)
    from config import CONFIG
except Exception:
    try:
        # Preferred when running as a subpackage (e.g., siel.core)
        from ..config import CONFIG
    except Exception:
        # Fallback to absolute package import if package name is 'siel'
        from siel.config import CONFIG


class TrainableCouplingMatrix:
    """
    Trainable coupling matrix W ≈ U @ V^T.
    Adaptive rank for complex patterns.
    """
    
    def __init__(self, grid_size: int, base_rank: int = 16, max_rank: int = 64):
        self.grid_size = grid_size
        self.n_oscillators = grid_size * grid_size
        self.base_rank = base_rank
        self.max_rank = max_rank
        self.current_rank = base_rank
        
        # Low-rank factorization
        self.U = np.random.randn(self.n_oscillators, base_rank).astype(np.float32) * 0.1
        self.V = np.random.randn(self.n_oscillators, base_rank).astype(np.float32) * 0.1
        
        # Local bias (preserves spatial locality)
        self.local_bias = self._init_local_bias()
        
        # Momentum for training
        self.U_momentum = np.zeros_like(self.U)
        self.V_momentum = np.zeros_like(self.V)
        
        # Adaptive rank tracking
        self.complexity_trace = deque(maxlen=100)
        self.rank_increase_count = 0
        
        # Stats
        self.weight_decay = CONFIG.WEIGHT_DECAY
    
    def _init_local_bias(self) -> np.ndarray:
        """Initialize local bias with spatial decay."""
        bias = np.zeros((self.n_oscillators, self.n_oscillators), dtype=np.float32)
        radius = CONFIG.COUPLING_RADIUS
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                i = y * self.grid_size + x
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            j = ny * self.grid_size + nx
                            dist = np.sqrt(dx*dx + dy*dy)
                            bias[i, j] = 1.0 / (1.0 + dist)
        return bias
    
    def get_coupling_vector(self, i: int) -> np.ndarray:
        """Get coupling weights for oscillator i."""
        low_rank = np.dot(self.U[i:i+1], self.V.T).flatten()
        return low_rank + self.local_bias[i]
    
    def get_weight(self, i: int, j: int) -> float:
        """Get coupling weight between oscillators i and j."""
        return float(np.dot(self.U[i], self.V[j])) + self.local_bias[i, j]
    
    def update_weights(self, phases: np.ndarray, target_phases: np.ndarray,
    learning_rate: float = 0.001,
                    indices: Optional[List[int]] = None) -> float:
        """
        Update coupling weights using anti-Hebbian learning.
        
        Returns:
            loss: The phase error after update.
        """
        n = len(phases)
        
        # Compute phase differences
        phase_diffs = phases[:, None] - phases[None, :]
        current_coupling = np.sin(phase_diffs)
        
        target_diffs = target_phases[:, None] - target_phases[None, :]
        target_coupling = np.sin(target_diffs)
        
        # Error signal
        error_signal = target_coupling - current_coupling
        
        # Gradient clipping
        max_grad_norm = CONFIG.GRADIENT_CLIP
        
        if indices is not None and len(indices) > 0:
            # Only update specific oscillators
            idx_array = np.array(indices)
            error_signal_slice = error_signal[np.ix_(idx_array, idx_array)]
            
            U_slice = self.U[idx_array]
            V_slice = self.V[idx_array]
            
            grad_U = np.dot(error_signal_slice, V_slice)
            grad_V = np.dot(error_signal_slice.T, U_slice)
            
            # Clip gradients
            grad_U = np.clip(grad_U, -max_grad_norm, max_grad_norm)
            grad_V = np.clip(grad_V, -max_grad_norm, max_grad_norm)
            
            # Update with momentum
            momentum = CONFIG.MOMENTUM_FACTOR
            self.U_momentum[idx_array] = (
                momentum * self.U_momentum[idx_array] + 
                learning_rate * grad_U / n
            )
            self.V_momentum[idx_array] = (
                momentum * self.V_momentum[idx_array] + 
                learning_rate * grad_V / n
            )
            
            self.U[idx_array] += self.U_momentum[idx_array]
            self.V[idx_array] += self.V_momentum[idx_array]
            
            # Weight decay
            self.U[idx_array] -= self.weight_decay * self.U[idx_array]
            self.V[idx_array] -= self.weight_decay * self.V[idx_array]
            
        else:
            # Full update
            grad_U = np.dot(error_signal, self.V)
            grad_V = np.dot(error_signal.T, self.U)
            
            grad_U = np.clip(grad_U, -max_grad_norm, max_grad_norm)
            grad_V = np.clip(grad_V, -max_grad_norm, max_grad_norm)
            
            momentum = CONFIG.MOMENTUM_FACTOR
            self.U_momentum = momentum * self.U_momentum + learning_rate * grad_U / n
            self.V_momentum = momentum * self.V_momentum + learning_rate * grad_V / n
            
            self.U += self.U_momentum
            self.V += self.V_momentum
            
            self.U -= self.weight_decay * self.U
            self.V -= self.weight_decay * self.V
        
        # Clamp weights
        self.U = np.clip(self.U, -1.0, 1.0)
        self.V = np.clip(self.V, -1.0, 1.0)
        
        # Calculate loss
        new_phases = self._simulate_update(phases)
        loss = np.mean(np.abs(new_phases - target_phases))

        # Track complexity for adaptive rank
        self.complexity_trace.append(loss)

        return float(loss)

    def _simulate_update(self, phases: np.ndarray) -> np.ndarray:
        """
        Simulate phase update for loss calculation.
        Uses the current coupling matrix to predict next phases.
        """
        # Get coupling matrix
        W = self.U @ self.V.T

        # Simulate one step of Kuramoto dynamics
        N = len(phases)
        phases_copy = phases.copy()

        # Phase differences
        phase_diffs = phases_copy[:, None] - phases_copy[None, :]

        # Coupling effect
        coupling_effect = np.sum(W * np.sin(phase_diffs), axis=1)

        # Natural frequency + coupling
        dt = 0.01
        natural_freq = 1.0  # Average natural frequency

        # Update
        new_phases = phases_copy + dt * (natural_freq + CONFIG.COUPLING_STRENGTH * coupling_effect / N)

        return new_phases % (2 * np.pi)

    def get_stats(self) -> Dict:
        """Get coupling matrix statistics."""
        weight_matrix = self.U @ self.V.T
        return {
            'mean_abs_weight': float(np.mean(np.abs(weight_matrix))),
            'local_bias_mean': float(np.mean(self.local_bias)),
            'current_rank': self.current_rank,
            'max_rank': self.max_rank,
            'rank_ratio': self.current_rank / self.max_rank,
            'avg_complexity': np.mean(list(self.complexity_trace)) if self.complexity_trace else 0.0,
            'rank_increases': self.rank_increase_count,
        }