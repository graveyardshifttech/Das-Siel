"""
Trigonometric Lookup Tables for fast computation.
"""

import numpy as np


class TrigLUT:
    """Lookup table for sine and cosine."""
    
    def __init__(self, resolution: int = 10000):
        """
        Initialize trig LUT.
        
        Args:
            resolution: Number of points in lookup table (10000 = ~0.00063 rad precision)
        """
        self.resolution = resolution
        self.x_vals = np.linspace(0, 2 * np.pi, resolution)
        self.sin_vals = np.sin(self.x_vals)
        self.cos_vals = np.cos(self.x_vals)
    
    def sin(self, x):
        """Fast sine using LUT."""
        # Normalize to [0, 2π]
        x_norm = np.remainder(x, 2 * np.pi)
        # Map to LUT indices
        indices = np.round((x_norm / (2 * np.pi)) * (self.resolution - 1)).astype(np.int32)
        indices = np.clip(indices, 0, self.resolution - 1)
        # Lookup
        return self.sin_vals[indices]
    
    def cos(self, x):
        """Fast cosine using LUT."""
        x_norm = np.remainder(x, 2 * np.pi)
        indices = np.round((x_norm / (2 * np.pi)) * (self.resolution - 1)).astype(np.int32)
        indices = np.clip(indices, 0, self.resolution - 1)
        return self.cos_vals[indices]


# Global LUT instance
SIN_LUT = TrigLUT()
