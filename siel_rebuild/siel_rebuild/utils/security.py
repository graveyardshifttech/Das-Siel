"""
Security helpers for input validation, rate limiting, and resource management.
"""

import time
import re
import os
from typing import Any, Optional, Dict, List
from collections import deque
from dataclasses import dataclass, field

from ..config import CONFIG


@dataclass
class ResourceLimits:
    """Resource limits to prevent exhaustion."""
    
    max_generations: int = 1000
    max_vision_history: int = 200
    max_executions: int = 100
    max_memory_mb: int = 1024  # 1GB
    max_requests_per_minute: int = 60
    
    generations: int = 0
    executions: int = 0
    request_timestamps: deque = field(default_factory=deque)
    
    def check_generations(self) -> bool:
        """Check if generation limit is reached."""
        if self.generations >= self.max_generations:
            return False
        self.generations += 1
        return True
    
    def check_executions(self) -> bool:
        """Check if execution limit is reached."""
        if self.executions >= self.max_executions:
            return False
        self.executions += 1
        return True
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit is reached."""
        now = time.time()
        # Clean old timestamps
        self.request_timestamps = deque(
            [t for t in self.request_timestamps if t > now - 60],
            maxlen=self.max_requests_per_minute
        )
        
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            return False
        
        self.request_timestamps.append(now)
        return True
    
    def reset(self):
        """Reset limits."""
        self.generations = 0
        self.executions = 0
        self.request_timestamps.clear()
    
    def get_stats(self) -> Dict:
        """Get current limits status."""
        return {
            'generations': f"{self.generations}/{self.max_generations}",
            'executions': f"{self.executions}/{self.max_executions}",
            'rate_limit': f"{len(self.request_timestamps)}/{self.max_requests_per_minute}",
            'memory_mb': self.max_memory_mb
        }


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Safely convert to str."""
    if value is None:
        return default
    try:
        return str(value)
    except:
        return default


def sanitize_code(code: str) -> str:
    """Sanitize code by removing dangerous patterns."""
    # Remove null bytes
    code = code.replace('\x00', '')
    
    # Remove control characters (except newline, tab)
    code = ''.join(c for c in code if ord(c) >= 32 or c in '\n\t')
    
    return code


def detect_secrets(text: str) -> List[str]:
    """Detect potential secrets in text."""
    secrets = []
    
    # API keys
    api_patterns = [
        r'sk-[a-zA-Z0-9]{48}',  # OpenAI
        r'hf_[a-zA-Z0-9]{40}',  # Hugging Face
        r'ghp_[a-zA-Z0-9]{36}',  # GitHub
        r'token:[a-zA-Z0-9]+',  # Generic token
        r'Bearer\s+[a-zA-Z0-9_-]+',  # Bearer token
    ]
    
    for pattern in api_patterns:
        matches = re.findall(pattern, text)
        secrets.extend(matches)
    
    # Email addresses
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    secrets.extend(emails[:10])  # Limit to 10
    
    # IP addresses (private only)
    ips = re.findall(r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b', text)
    secrets.extend(ips[:5])
    
    return secrets[:20]  # Limit to 20


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    # Check for path traversal
    if '../' in path or '..\\' in path:
        return False
    
    # Check for dangerous extensions
    dangerous_extensions = ['.exe', '.dll', '.so', '.dylib', '.bin']
    ext = os.path.splitext(path)[1].lower()
    if ext in dangerous_extensions:
        return False
    
    # Check for dangerous names
    dangerous_names = ['config', 'secret', 'key', 'password', 'credential']
    basename = os.path.basename(path).lower()
    for name in dangerous_names:
        if name in basename:
            return False
    
    return True


# Global resource limits
resource_limits = ResourceLimits()