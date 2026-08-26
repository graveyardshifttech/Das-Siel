"""
DualStreamTrainer — Separates code and language into different frequency bands.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque

try:
    from config import CONFIG
except ImportError:
    from config import CONFIG

try:
    from core.wave import WaveField
except ImportError:
    from core.wave import WaveField


class DualStreamTrainer:
    """
    Trains oscillators on BOTH code and language.
    Different modalities get different frequency bands.
    """
    
    def __init__(self, wave_field: WaveField):
        self.wave = wave_field
        
        # Frequency bands
        self.code_band = (0.5, 1.5)   # Low frequencies for code
        self.lang_band = (2.0, 4.0)   # High frequencies for language
        
        # Confidence tracking
        self.modality_resonance = {'code': 0.5, 'language': 0.5}
        self.training_count = {'code': 0, 'language': 0}
        self.loss_history = {'code': deque(maxlen=100), 'language': deque(maxlen=100)}
    
    def train_code(self, code: str) -> float:
        """Train on code with code frequency band."""
        if len(code) < 10:
            return 0.0
        
        phases = self._text_to_phases(code, self.code_band)
        loss = self._apply_training(phases, 'code')
        self.training_count['code'] += 1
        self.loss_history['code'].append(loss)
        self.modality_resonance['code'] = min(1.0, self.modality_resonance['code'] + 0.01)
        return loss
    
    def train_language(self, text: str) -> float:
        """Train on language with language frequency band."""
        if len(text) < 10:
            return 0.0
        
        phases = self._text_to_phases(text, self.lang_band)
        loss = self._apply_training(phases, 'language')
        self.training_count['language'] += 1
        self.loss_history['language'].append(loss)
        self.modality_resonance['language'] = min(1.0, self.modality_resonance['language'] + 0.01)
        return loss
    
    def _text_to_phases(self, text: str, freq_band: Tuple[float, float]) -> np.ndarray:
        """Convert text to phases at a specific frequency band."""
        # Encode text to bytes
        bytes_data = text.encode('utf-8', errors='ignore')[:CONFIG.GRID_SIZE * CONFIG.GRID_SIZE]
        phases = np.array([(b / 255.0) * 2 * np.pi for b in bytes_data])
        
        # Modulate by frequency band
        freq_multiplier = (freq_band[0] + freq_band[1]) / 2
        phases = phases * freq_multiplier
        
        # Pad to grid size
        grid_size = CONFIG.GRID_SIZE * CONFIG.GRID_SIZE
        if len(phases) < grid_size:
            phases = np.pad(phases, (0, grid_size - len(phases)))
        elif len(phases) > grid_size:
            phases = phases[:grid_size]
        
        return phases
    
    def _apply_training(self, target_phases: np.ndarray, modality: str) -> float:
        """Apply training to the wave field."""
        current_phases = self.wave.get_phases().flatten()
        target_phases = np.asarray(target_phases).flatten()

        lr = CONFIG.WEIGHT_LEARNING_RATE
        if modality == 'code' and self.modality_resonance['code'] < 0.3:
            lr *= 1.5
        elif modality == 'language' and self.modality_resonance['language'] < 0.3:
            lr *= 1.5

        self.wave.coupling_matrix.update_weights(
            current_phases,
            target_phases,
            learning_rate=lr
        )

        self.wave.propagate_hybrid(dt=0.01, steps=3)

        new_phases = self.wave.get_phases().flatten()
        min_len = min(len(new_phases), len(target_phases))
        diff = np.abs(new_phases[:min_len] - target_phases[:min_len]) % (2 * np.pi)
        diff = np.minimum(diff, 2 * np.pi - diff)
        loss = float(np.mean(diff) / np.pi)

        return float(loss)
    
    def identify_modality(self, text: str) -> str:
        """Identify if text is code or language."""
        if len(text) < 10:
            return 'unknown'
        
        # Quick heuristic
        code_score = 0
        lang_score = 0
        
        code_patterns = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ']
        lang_patterns = [' the ', ' and ', ' to ', ' of ', ' for ', ' with ', ' on ']
        
        for pattern in code_patterns:
            if pattern in text:
                code_score += 1
        
        for pattern in lang_patterns:
            if pattern in text:
                lang_score += 1
        
        if code_score > lang_score * 2:
            return 'code'
        elif lang_score > code_score * 2:
            return 'language'
        
        return 'mixed'
    
    def get_confidence(self) -> Dict:
        """Get modality confidence scores."""
        return {
            'code': self.modality_resonance['code'],
            'language': self.modality_resonance['language'],
            'code_samples': self.training_count['code'],
            'language_samples': self.training_count['language'],
            'code_loss': np.mean(list(self.loss_history['code'])) if self.loss_history['code'] else None,
            'lang_loss': np.mean(list(self.loss_history['language'])) if self.loss_history['language'] else None,
        }