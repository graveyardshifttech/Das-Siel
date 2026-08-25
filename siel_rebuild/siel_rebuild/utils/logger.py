"""
Logger — Unified logging for Siel.
"""

import time
import os
from pathlib import Path
from datetime import datetime


class Logger:
    """Simple but powerful logging."""
    
    def __init__(self, log_dir: str = "./siel_logs", name: str = "siel"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.start_time = time.time()
    
    def _format_message(self, msg: str, level: str = "INFO") -> str:
        """Format log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{level}]  {msg}"
    
    def log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        formatted = self._format_message(msg, level)
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def info(self, msg: str):
        """Log info level."""
        self.log(msg, "INFO")
    
    def warning(self, msg: str):
        """Log warning level."""
        self.log(msg, "WARN")
    
    def debug(self, msg: str):
        """Log debug level."""
        self.log(msg, "DEBUG")
    
    def error(self, msg: str):
        """Log error level."""
        self.log(msg, "ERROR")
    
    def log_training_epoch(self, name: str, epoch: int, loss: float, affinity: float):
        """Log training epoch."""
        self.info(f"[{name}] Epoch {epoch}: loss={loss:.6f}, affinity={affinity:.3f}")
    
    def flush(self):
        """Flush logs (no-op for file-based logger)."""
        pass
    
    def get_stats(self) -> dict:
        """Get logger stats."""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_seconds': elapsed,
            'log_file': str(self.log_file),
        }


# Global logger instance
logger = Logger()
