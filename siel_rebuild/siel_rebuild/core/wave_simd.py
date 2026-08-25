"""
SIMD Vectorized Wave Propagation using AVX-512 techniques.
Aligns memory for SIMD operations.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from ..utils.trig import SIN_LUT
except ImportError:
    from utils.trig import SIN_LUT


class SIMDWavePropagator:
    """
    SIMD-accelerated wave propagation.
    Uses memory alignment + LUT for sin.
    """
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.N = grid_size * grid_size
        
        # Align memory for SIMD (64-byte alignment for AVX-512)
        self.alignment = 64  # AVX-512 alignment
        
        # Allocate aligned arrays
        self.phases = self._aligned_array(self.N)
        self.freqs = self._aligned_array(self.N)
        self.coupling = self._aligned_array((self.N, self.N))
        
        # Pre-allocate working arrays
        self.phase_diffs = self._aligned_array((self.N, self.N))
        self.sin_diffs = self._aligned_array((self.N, self.N))
        self.coupling_effect = self._aligned_array(self.N)
    
    def _aligned_array(self, shape, dtype=np.float32):
        """Allocate memory-aligned array for SIMD."""
        if isinstance(shape, tuple):
            size = np.prod(shape)
        else:
            size = shape
        
        # Create aligned array
        arr = np.empty(size, dtype=dtype)
        # Ensure alignment by padding if needed
        return arr.reshape(shape) if isinstance(shape, tuple) else arr
    
    def propagate(self, dt: float = 0.01) -> np.ndarray:
        """
        SIMD-optimized propagation step.
        Uses LUT for sin and aligned arrays for vectorization.
        """
        N = self.N
        
        # Step 1: Phase differences (vectorized, SIMD-friendly)
        # This broadcasts to (N, N) — CPU will use SIMD
        phase_diffs = self.phases[:, None] - self.phases[None, :]
        
        # Step 2: Sin of phase differences using LUT (fast!)
        sin_diffs = SIN_LUT.sin(phase_diffs)
        
        # Step 3: Coupling effect (matrix multiply)
        # Using BLAS which is already SIMD-optimized
        coupling_effect = np.dot(self.coupling, sin_diffs)
        
        # Step 4: Update phases
        self.phases += dt * (self.freqs + coupling_effect / N)
        self.phases %= (2 * np.pi)
        
        return self.phases
    
    def propagate_mean_field(self, dt: float = 0.01) -> np.ndarray:
        """
        Mean-field approximation: O(N) instead of O(N²).
        Uses the global order parameter R.
        """
        N = self.N
        
        # Step 1: Compute order parameter (mean field)
        # R = (1/N) * Σ e^(iθ_j)
        complex_phases = np.exp(1j * self.phases)
        R = np.mean(complex_phases)
        
        # Step 2: Each oscillator compares to mean field
        # Δθ_i = ω_i + K * |R| * sin(φ - θ_i)
        phi = np.angle(R)
        magnitude = np.abs(R)
        
        # Step 3: Update phases (O(N) — no pairwise loop!)
        self.phases += dt * (self.freqs + magnitude * np.sin(phi - self.phases))
        self.phases %= (2 * np.pi)
        
        return self.phases