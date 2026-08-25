"""
State management with crash recovery and rollback.
"""

import os
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..config import CONFIG
from ..utils.logger import logger


@dataclass
class StateSnapshot:
    """A snapshot of system state."""
    
    timestamp: float
    version: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute checksum of data."""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class StateManager:
    """
    Manages state with automatic snapshots and recovery.
    """
    
    def __init__(self, state_dir: str = "./siel_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_state: Dict[str, Any] = {}
        
        # Snapshots
        self.snapshots: List[StateSnapshot] = []
        self.max_snapshots = 10
        
        # Recovery
        self.last_good_state: Optional[Dict[str, Any]] = None
        self.recovery_count = 0
    
    def save(self, key: str, value: Any):
        """Save a value to state."""
        self.current_state[key] = value
        self._auto_snapshot()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state."""
        return self.current_state.get(key, default)
    
    def snapshot(self) -> StateSnapshot:
        """Create a new snapshot."""
        snapshot = StateSnapshot(
            timestamp=time.time(),
            version=1,
            data=self.current_state.copy()
        )
        
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Save to disk
        self._save_snapshot(snapshot)
        
        return snapshot
    
    def _auto_snapshot(self):
        """Auto-snapshot if significant changes."""
        # Only snapshot every 10 changes or if important
        if len(self.current_state) % 10 == 0:
            self.snapshot()
    
    def _save_snapshot(self, snapshot: StateSnapshot):
        """Save snapshot to disk."""
        path = self.state_dir / f"snapshot_{int(snapshot.timestamp)}.json"
        try:
            with open(path, 'w') as f:
                json.dump({
                    'timestamp': snapshot.timestamp,
                    'version': snapshot.version,
                    'checksum': snapshot.checksum,
                    'data': snapshot.data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.log(f"Failed to save snapshot: {e}", "ERROR")
    
    def load_snapshot(self, timestamp: float = None) -> Optional[Dict[str, Any]]:
        """Load a snapshot from disk."""
        if timestamp is None:
            # Load most recent
            snapshots = sorted(self.state_dir.glob("snapshot_*.json"))
            if not snapshots:
                return None
            path = snapshots[-1]
        else:
            path = self.state_dir / f"snapshot_{int(timestamp)}.json"
            if not path.exists():
                return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Verify checksum
            snapshot = StateSnapshot(
                timestamp=data['timestamp'],
                version=data['version'],
                data=data['data']
            )
            
            if snapshot.checksum != data['checksum']:
                logger.log(f"Checksum mismatch for snapshot {path}", "ERROR")
                return None
            
            return data['data']
        except Exception as e:
            logger.log(f"Failed to load snapshot: {e}", "ERROR")
            return None
    
    def recover(self) -> bool:
        """Recover from last good state."""
        # Try to load last good state
        if self.last_good_state is not None:
            self.current_state = self.last_good_state.copy()
            self.recovery_count += 1
            logger.log(f"Recovered from last good state (recovery #{self.recovery_count})")
            return True
        
        # Try to load from snapshot
        data = self.load_snapshot()
        if data is not None:
            self.current_state = data.copy()
            self.last_good_state = data.copy()
            self.recovery_count += 1
            logger.log(f"Recovered from snapshot (recovery #{self.recovery_count})")
            return True
        
        logger.log("Recovery failed: no good state available", "ERROR")
        return False
    
    def mark_good(self):
        """Mark current state as good."""
        self.last_good_state = self.current_state.copy()
        self.snapshot()
    
    def get_state_size(self) -> int:
        """Get size of current state in memory."""
        try:
            data_str = json.dumps(self.current_state, default=str)
            return len(data_str.encode('utf-8'))
        except:
            return 0
    
    def get_stats(self) -> Dict:
        """Get state manager statistics."""
        return {
            'state_size_bytes': self.get_state_size(),
            'snapshot_count': len(self.snapshots),
            'recovery_count': self.recovery_count,
            'state_keys': list(self.current_state.keys()),
            'has_last_good': self.last_good_state is not None
        }


# Global state manager
state_manager = StateManager()