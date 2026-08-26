"""
CPU Saver — Throttle computation to avoid overheating on resource-constrained systems.
"""

import time
import psutil
from typing import Optional


class CPUSaver:
    """Throttle CPU usage based on system load."""
    
    def __init__(self, mode: str = "moderate"):
        """
        Initialize CPU saver.
        
        Modes:
        - "aggressive": Maximum throttling (sleep 100ms between batches)
        - "moderate": Balanced (sleep 10ms)
        - "light": Minimal throttling (sleep 1ms)
        - "none": No throttling
        """
        self.mode = mode
        self.sleep_times = {
            "aggressive": 0.1,
            "moderate": 0.01,
            "light": 0.001,
            "none": 0.0,
        }
        self.last_throttle = {}
    
    def throttle(self, task: str = "batch"):
        """Throttle based on mode."""
        sleep_time = self.sleep_times.get(self.mode, 0.01)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0


# Global CPU saver instance
cpu_saver = CPUSaver(mode="moderate")
