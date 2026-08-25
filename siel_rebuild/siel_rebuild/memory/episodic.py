"""
Episodic Memory — Human-like memory with importance weighting.
Stores events, experiences, and learns what matters.
"""

import time
import json
from typing import Dict, List, Any, Optional
from collections import deque, Counter
from dataclasses import dataclass, field

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class Episode:
    """A single memory episode."""
    timestamp: float
    type: str  # 'learning', 'execution', 'discovery', 'success', 'failure', 'interaction'
    content: Dict[str, Any]
    importance: float = 0.5
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    context: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Human-like episodic memory with:
    - Importance weighting (some memories matter more)
    - Emotional valence (positive/negative experiences)
    - Semantic extraction (knowledge from events)
    - Procedural memory (skills from repeated actions)
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or CONFIG.EPISODIC_MEMORY_SIZE
        self.episodes: deque = deque(maxlen=self.max_size)
        
        # Semantic memory: extracted knowledge
        self.semantic_memory: Dict[str, List[Dict]] = {}
        
        # Procedural memory: learned skills
        self.procedural_memory: Dict[str, Dict] = {}
        
        # Memory statistics
        self.importance_threshold = CONFIG.IMPORTANCE_THRESHOLD
        self.forgetting_rate = CONFIG.FORGETTING_RATE
        self.total_episodes = 0
        self.consolidation_count = 0
    
    def remember(self, episode_type: str, content: Dict[str, Any],
                 importance: float = 0.5, emotional_valence: float = 0.0,
                 context: Optional[Dict] = None) -> Episode:
        """
        Store a memory episode.
        
        Args:
            episode_type: 'learning', 'execution', 'discovery', 'success', 'failure', 'interaction'
            content: The memory content
            importance: 0-1, how important this memory is
            emotional_valence: -1 to 1, emotional tone
            context: Additional context
        """
        episode = Episode(
            timestamp=time.time(),
            type=episode_type,
            content=content,
            importance=importance,
            emotional_valence=emotional_valence,
            context=context or {}
        )
        
        self.episodes.append(episode)
        self.total_episodes += 1
        
        # Important episodes update semantic memory
        if importance > self.importance_threshold:
            self._update_semantic_memory(episode)
        
        # Successful actions update procedural memory
        if episode_type == 'success' and content.get('procedure'):
            self.procedural_memory[content['procedure']] = {
                'timestamp': episode.timestamp,
                'content': content,
                'importance': importance,
                'repetitions': self.procedural_memory.get(content['procedure'], {}).get('repetitions', 0) + 1
            }
        
        # Periodic consolidation
        if self.total_episodes % 100 == 0:
            self._consolidate()
        
        return episode
    
    def _update_semantic_memory(self, episode: Episode):
        """Update semantic memory from important episodes."""
        key = episode.type
        
        if key not in self.semantic_memory:
            self.semantic_memory[key] = []
        
        # Store compressed version
        entry = {
            'timestamp': episode.timestamp,
            'content': episode.content,
            'importance': episode.importance,
            'emotional_valence': episode.emotional_valence,
            'summary': self._extract_summary(episode)
        }
        
        self.semantic_memory[key].append(entry)
        
        # Keep only top 10 most important
        self.semantic_memory[key] = sorted(
            self.semantic_memory[key],
            key=lambda x: x['importance'],
            reverse=True
        )[:10]
    
    def _extract_summary(self, episode: Episode) -> str:
        """Extract a summary from an episode."""
        content = episode.content
        if 'code' in content:
            return f"Code: {content['code'][:50]}..."
        elif 'file' in content:
            return f"File: {content['file']}"
        elif 'error' in content:
            return f"Error: {content['error'][:50]}"
        elif 'discovery' in content:
            return f"Discovery: {content.get('what', 'unknown')}"
        return str(content)[:50]
    
    def _consolidate(self):
        """Consolidate memories (reduce redundancy, strengthen important ones)."""
        self.consolidation_count += 1
        
        # Reduce importance of old memories over time
        current_time = time.time()
        for episode in list(self.episodes):
            age = current_time - episode.timestamp
            if age > 86400:  # > 1 day old
                # Reduce importance slightly
                episode.importance *= (1 - self.forgetting_rate)
        
        # Remove low-importance old memories if we're full
        if len(self.episodes) >= self.max_size:
            # Remove lowest importance episodes
            sorted_episodes = sorted(
                [e for e in self.episodes if e.importance < 0.3],
                key=lambda x: x.importance
            )
            for e in sorted_episodes[:10]:  # Remove up to 10
                self.episodes.remove(e)
    
    def recall(self, episode_type: Optional[str] = None,
               limit: int = 10,
               min_importance: float = 0.0) -> List[Episode]:
        """Recall episodes matching criteria."""
        filtered = []
        
        for e in self.episodes:
            if episode_type and e.type != episode_type:
                continue
            if e.importance < min_importance:
                continue
            filtered.append(e)
        
        # Sort by importance (most important first)
        filtered.sort(key=lambda x: x.importance, reverse=True)
        return filtered[:limit]
    
    def recall_semantic(self, key: str) -> List[Dict]:
        """Recall semantic memory."""
        return self.semantic_memory.get(key, [])
    
    def recall_procedural(self, procedure: str) -> Optional[Dict]:
        """Recall procedural memory."""
        return self.procedural_memory.get(procedure)
    
    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        return list(self.episodes)[-limit:]
    
    def get_summary(self) -> Dict:
        """Get memory summary statistics."""
        return {
            'total_episodes': self.total_episodes,
            'active_episodes': len(self.episodes),
            'memory_usage': len(self.episodes) / self.max_size,
            'types': Counter(e.type for e in self.episodes),
            'semantic_keys': list(self.semantic_memory.keys()),
            'procedural_keys': list(self.procedural_memory.keys()),
            'avg_importance': sum(e.importance for e in self.episodes) / max(1, len(self.episodes)),
            'consolidations': self.consolidation_count
        }
    
    def save(self, filepath: str):
        """Save memory to disk."""
        data = {
            'total_episodes': self.total_episodes,
            'consolidation_count': self.consolidation_count,
            'semantic_memory': self.semantic_memory,
            'procedural_memory': self.procedural_memory,
            'episodes': [
                {
                    'timestamp': e.timestamp,
                    'type': e.type,
                    'content': e.content,
                    'importance': e.importance,
                    'emotional_valence': e.emotional_valence,
                    'context': e.context
                }
                for e in self.episodes
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: str):
        """Load memory from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.total_episodes = data.get('total_episodes', 0)
        self.consolidation_count = data.get('consolidation_count', 0)
        self.semantic_memory = data.get('semantic_memory', {})
        self.procedural_memory = data.get('procedural_memory', {})
        
        # Reconstruct episodes
        self.episodes.clear()
        for e_data in data.get('episodes', []):
            episode = Episode(
                timestamp=e_data['timestamp'],
                type=e_data['type'],
                content=e_data['content'],
                importance=e_data.get('importance', 0.5),
                emotional_valence=e_data.get('emotional_valence', 0.0),
                context=e_data.get('context', {})
            )
            self.episodes.append(episode)