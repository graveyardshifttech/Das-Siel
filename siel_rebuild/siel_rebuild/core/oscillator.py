"""
PhaseOscillator — Pure oscillator physics.
No training logic, no memory, just phase dynamics.
"""

import math
import numpy as np
from typing import Optional
from collections import deque

# Import CONFIG robustly in different runtime contexts
try:
    # Preferred when running from package root (parent dir on sys.path)
    from config import CONFIG
except Exception:
    try:
        # Preferred when running as a subpackage (e.g., siel.core)
        from ..config import CONFIG
    except Exception:
        # Fallback to absolute package import if package name is 'siel'
        from siel.config import CONFIG


class PhaseOscillator:
    """
    A single phase oscillator with Kuramoto dynamics.
    """
    
    def __init__(self, dim: int = None, natural_freq: float = None):
        self.dim = dim or CONFIG.PHASE_DIM
        self.natural_freq = natural_freq or np.random.uniform(*CONFIG.NATURAL_FREQ_RANGE)
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.amplitude = 1.0 + 0.1 * np.random.randn()
        
        self.phase_vector = np.exp(1j * np.linspace(0, 2 * np.pi, self.dim))
        self.phase_vector *= np.exp(1j * self.phase)
        
        self.phase_history = deque(maxlen=64)
        self.phase_history.append(self.phase)
        
        # Memory pinning
        self.pinned_concept_id: Optional[int] = None
        self.pin_resonance_count: int = 0
        self.resonance_score: float = 0.0
    
    def evolve(self, dt: float = 0.01, coupling_weights: np.ndarray = None,
        phases_field: np.ndarray = None):
        """
        Evolve oscillator by one step.
        
        Args:
            dt: Time step
            coupling_weights: Coupling to other oscillators
            phases_field: All oscillator phases (for local coupling)
        """
        dphase = self.natural_freq * dt
        
        if coupling_weights is not None and phases_field is not None:
            phase_diffs = phases_field - self.phase
            coupling_effect = np.sum(coupling_weights * np.sin(phase_diffs))
            n = max(1, len(coupling_weights))
            dphase += (CONFIG.COUPLING_STRENGTH / n) * coupling_effect * dt
        
        self.phase = (self.phase + dphase) % (2 * np.pi)
        self.phase_vector = np.exp(1j * np.linspace(0, 2 * np.pi, self.dim))
        self.phase_vector *= np.exp(1j * self.phase)
        self.amplitude = 1.0 + 0.1 * np.sin(self.phase)
        self.phase_history.append(self.phase)
        self.resonance_score = self.amplitude * abs(np.sin(self.phase))
    
    def inject_phase(self, phase: float):
        """Inject a phase value."""
        self.phase = phase % (2 * np.pi)
        self.phase_vector = np.exp(1j * np.linspace(0, 2 * np.pi, self.dim))
        self.phase_vector *= np.exp(1j * self.phase)
    
    def get_state(self) -> dict:
        """Get oscillator state."""
        return {
            'phase': self.phase,
            'amplitude': self.amplitude,
            'resonance_score': self.resonance_score,
            'natural_freq': self.natural_freq,
        }