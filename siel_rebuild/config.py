"""
DNA — Configuration for Siel
All settings in one place.
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class Config:
    """Global configuration for Siel."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # GRID & OSCILLATORS
    # ═══════════════════════════════════════════════════════════════════════
    GRID_SIZE: int = 64
    BOTTLENECK_SIZE: int = 8
    PHASE_DIM: int = 512
    PIN_COUNT: int = 48
    PIN_RESONANCE_THRESHOLD: float = 0.12
    NATURAL_FREQ_RANGE: Tuple[float, float] = (0.2, 3.5)
    COUPLING_STRENGTH: float = 0.5
    COUPLING_RADIUS: int = 6
    
    # ═══════════════════════════════════════════════════════════════════════
    # MEMORY
    # ═══════════════════════════════════════════════════════════════════════
    MAX_DISK_CONCEPTS: int = 50000  # Reduced for performance
    MMAP_FILENAME: str = "concept_space.siel"
    EPISODIC_MEMORY_SIZE: int = 2000
    IMPORTANCE_THRESHOLD: float = 0.2
    FORGETTING_RATE: float = 0.01
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    WEIGHT_LEARNING_RATE: float = 0.1
    EPOCHS: int = 100
    MAX_FILE_SIZE: int = 100000
    BATCH_SIZE: int = 3
    MOMENTUM_FACTOR: float = 0.95
    GRADIENT_CLIP: float = 0.5
    WEIGHT_DECAY: float = 0.0001
    MASTERY_THRESHOLD: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    MAX_EPOCHS_PER_FILE: int = 50
    
    # ═══════════════════════════════════════════════════════════════════════
    # REPLAY (Catastrophic Forgetting Prevention)
    # ═══════════════════════════════════════════════════════════════════════
    REPLAY_BUFFER_SIZE: int = 100
    REPLAY_FREQUENCY: int = 3
    REPLAY_BATCH_SIZE: int = 5
    PERFECT_AFFINITY_THRESHOLD: float = 0.85
    
    # ═══════════════════════════════════════════════════════════════════════
    # ATTENTION
    # ═══════════════════════════════════════════════════════════════════════
    ATTENTION_HEADS: int = 4
    ATTENTION_MIX: float = 0.5
    ATTENTION_DROPOUT: float = 0.1
    
    # ═══════════════════════════════════════════════════════════════════════
    # MCTS
    # ═══════════════════════════════════════════════════════════════════════
    MCTS_SIMULATIONS: int = 10  # Reduced for speed
    MCTS_EXPLORATION: float = 1.414
    MCTS_BRANCHING: int = 3
    MCTS_MAX_DEPTH: int = 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # AUTONOMY & CURIOSITY
    # ═══════════════════════════════════════════════════════════════════════
    CURIOSITY_WEIGHT: float = 0.5
    UNCERTAINTY_THRESHOLD: float = 0.3
    AUTONOMY_INTERVAL: int = 60
    DESIRE_STRENGTH_THRESHOLD: float = 0.6
    
    # ═══════════════════════════════════════════════════════════════════════
    # AUTO-LEARNING
    # ═══════════════════════════════════════════════════════════════════════
    WATCH_POLL_INTERVAL: int = 2
    
    # ═══════════════════════════════════════════════════════════════════════
    # CPU
    # ═══════════════════════════════════════════════════════════════════════
    CPU_MODE: str = "moderate"
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIRECTORIES
    # ═══════════════════════════════════════════════════════════════════════
    WATCH_DIR: str = "./watch"
    SANDBOX_DIR: str = "./sandbox"
    LOG_DIR: str = "./siel_logs"
    
    # ═══════════════════════════════════════════════════════════════════════
    # VERBOSE
    # ═══════════════════════════════════════════════════════════════════════
    VERBOSE: bool = True


# Singleton config
CONFIG = Config()