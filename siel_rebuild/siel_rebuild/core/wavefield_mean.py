"""
Mean-Field WaveField — O(N) propagation using FFT/order parameter.
Replaces the full O(N²) coupling with O(N) mean-field approximation.
"""

import numpy as np
from typing import Optional, Tuple, Dict

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG

try:
    from ..utils.trig import SIN_LUT
except ImportError:
    from utils.trig import SIN_LUT

try:
    from .wave_simd import SIMDWavePropagator
except ImportError:
    from core.wave_simd import SIMDWavePropagator


class MeanFieldWaveField:
    """
    WaveField using mean-field approximation.
    O(N) propagation instead of O(N²).
    """
    
    def __init__(self, grid_size: int = None):
        self.grid_size = grid_size or CONFIG.GRID_SIZE
        self.N = self.grid_size * self.grid_size
        
        # SIMD propagator for fast updates
        self.propagator = SIMDWavePropagator(self.grid_size)
        
        # Initialize phases
        self.propagator.phases = np.random.uniform(0, 2 * np.pi, self.N)
        self.propagator.freqs = np.random.uniform(*CONFIG.NATURAL_FREQ_RANGE, self.N)
        
        # Coupling matrix (stored but used for mean-field only)
        self.coupling_matrix = None
        self.coupling_strength = CONFIG.COUPLING_STRENGTH
        
        # Pins
        self.pin_count = CONFIG.PIN_COUNT
        self.pin_coords = self._generate_pin_coords()
        
        # Goals
        self.goal_attractors: Dict[int, float] = {}
        
        # Stats
        self.stats = {
            'phase_mean': 0.0,
            'phase_std': 0.0,
            'order_parameter': 0.0,
            'resonance': 0.0
        }
    
    def _generate_pin_coords(self) -> np.ndarray:
        """Generate pin indices."""
        step = max(1, self.grid_size // int(np.sqrt(self.pin_count)))
        coords = []
        for y in range(0, self.grid_size, step):
            for x in range(0, self.grid_size, step):
                if len(coords) < self.pin_count:
                    coords.append(y * self.grid_size + x)
        return np.array(coords[:self.pin_count])
    
    def propagate(self, dt: float = 0.01, steps: int = 10):
        """Propagate using mean-field approximation (O(N))."""
        for _ in range(steps):
            # Use mean-field propagator (O(N))
            self.propagator.propagate_mean_field(dt)
            
            # Apply goal attractors
            if self.goal_attractors:
                for idx, target_phase in self.goal_attractors.items():
                    self.propagator.phases[idx] += 0.1 * np.sin(
                        target_phase - self.propagator.phases[idx]
                    ) * dt
                    self.propagator.phases[idx] %= (2 * np.pi)
            
            # Update stats
            self._update_stats()
    
    def _update_stats(self):
        """Update statistics."""
        phases = self.propagator.phases
        self.stats['phase_mean'] = float(np.mean(phases))
        self.stats['phase_std'] = float(np.std(phases))
        
        # Order parameter
        R = np.mean(np.exp(1j * phases))
        self.stats['order_parameter'] = float(np.abs(R))
        
        # Resonance (simplified)
        self.stats['resonance'] = float(np.mean(np.abs(np.sin(phases))))
    
    def inject_phase(self, idx: int, phase: float):
        """Inject phase at index."""
        if 0 <= idx < self.N:
            self.propagator.phases[idx] = phase % (2 * np.pi)
    
    def get_phase_map(self) -> np.ndarray:
        """Get 2D phase map."""
        return self.propagator.phases.reshape(self.grid_size, self.grid_size)
    
    def get_wave_snapshot(self) -> np.ndarray:
        """Get wave snapshot."""
        phases = self.propagator.phases
        return np.sin(phases).reshape(self.grid_size, self.grid_size)
    
    def get_resonating_pins(self) -> np.ndarray:
        """Get resonating pin indices."""
        threshold = CONFIG.PIN_RESONANCE_THRESHOLD
        resonance = np.abs(np.sin(self.propagator.phases))
        return self.pin_coords[resonance[self.pin_coords] > threshold]
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return self.stats
    
    def save_state(self) -> Dict:
        """Save state."""
        return {
            'phases': self.propagator.phases.copy(),
            'freqs': self.propagator.freqs.copy(),
            'goals': self.goal_attractors.copy(),
        }
    
    def restore_state(self, state: Dict):
        """Restore state."""
        self.propagator.phases = state['phases'].copy()
        self.propagator.freqs = state['freqs'].copy()
        self.goal_attractors = state.get('goals', {}).copy()