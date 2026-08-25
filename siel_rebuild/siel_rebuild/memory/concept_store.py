"""
ConceptStore — Streaming .siel file format.
Loads ONLY metadata, streams concepts on demand.
No more 400MB RAM usage!
"""

import os
import json
import mmap
import struct
import time
import numpy as np
from typing import Dict, Optional, List, Any, Set
from collections import deque
from pathlib import Path
try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG


class StreamingConceptStore:
    """
    .siel file format — streams concepts, doesn't load all into RAM.
    
    File format:
    ┌─────────────────────────────────────┐
    │ Header: 4096 bytes                  │
    │  - Magic: "SIEL" (4 bytes)          │
    │  - Version: 3 (4 bytes)             │
    │  - Max Concepts: 4 bytes            │
    │  - Embed Dim: 4 bytes               │
    │  - Concept Count: 4 bytes           │
    │  - Free List Head: 8 bytes          │
    │  - Reserved: 4064 bytes             │
    ├─────────────────────────────────────┤
    │ Concept 0 (512 complex floats)      │
    │ Concept 1 (512 complex floats)      │
    │ ...                                 │
    │ Concept N (512 complex floats)      │
    ├─────────────────────────────────────┤
    │ Metadata (JSON, variable size)      │
    └─────────────────────────────────────┘
    """
    
    MAGIC = b"SIEL"
    VERSION = 3
    HEADER_SIZE = 4096
    HEADER_FORMAT = '<4sIIIIq'
    HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FORMAT)
    RESERVED_SIZE = HEADER_SIZE - HEADER_STRUCT_SIZE
    FULL_HEADER_FORMAT = f'<4sIIIIq{RESERVED_SIZE}s'
    
    def __init__(self, filename: str = None, max_concepts: int = None, embed_dim: int = None):
        self.filename = filename or CONFIG.MMAP_FILENAME
        self.max_concepts = min(
            max_concepts or CONFIG.MAX_DISK_CONCEPTS,
            50000  # Cap for performance
        )
        self.embed_dim = embed_dim or CONFIG.PHASE_DIM
        
        self.concept_floats = self.embed_dim * 2  # Complex → 2 floats
        self.concept_bytes = self.concept_floats * 4
        self.data_size = self.max_concepts * self.concept_bytes
        self.total_size = self.HEADER_SIZE + self.data_size
        
        # ═══ Cache (LRU) — Only 100 concepts in RAM ═══
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_size = 100
        self._dirty: Set[int] = set()  # Concepts needing save
        self._access_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # ═══ Metadata ═══
        self.pin_mapping: Dict[int, int] = {}
        self.concept_metadata: Dict[int, Dict] = {}
        self.replay_buffer: deque = deque(maxlen=CONFIG.REPLAY_BUFFER_SIZE)
        
        # ═══ File I/O ═══
        self.file_obj = None
        self.mapped_memory = None
        self._init_file()
        self._read_header()
    
    def _init_file(self):
        """Initialize or load .siel file."""
        if os.path.exists(self.filename):
            # Check if it's a valid .siel file
            try:
                with open(self.filename, 'rb') as f:
                    magic = f.read(4)
                    if magic == self.MAGIC:
                        print(f"📀 Loading existing .siel file: {self.filename}")
                        return
                    else:
                        print(f"⚠️ Existing file is not a .siel file. Backing up...")
                        shutil.copy(self.filename, self.filename + ".bak")
            except:
                pass
        
        # Create new file
        print(f"📀 Creating .siel file: {self.filename} (max {self.max_concepts} concepts)")
        
        with open(self.filename, "wb") as f:
            # Write header
            reserved = b'\x00' * self.RESERVED_SIZE
            header = struct.pack(
                self.FULL_HEADER_FORMAT,
                self.MAGIC,
                self.VERSION,
                self.max_concepts,
                self.embed_dim,
                0,  # concept_count
                -1,  # free_list_head
                reserved
            )
            f.write(header)
            
            # Write empty concept space
            f.write(b'\x00' * self.data_size)
        
        # Write metadata
        self._write_metadata({})
    
    def _read_header(self):
        """Read header from file."""
        with open(self.filename, 'rb') as f:
            header_data = f.read(self.HEADER_SIZE)
            magic, version, max_concepts, embed_dim, concept_count, free_list_head, _ = struct.unpack(
                self.FULL_HEADER_FORMAT, header_data
            )
            
            if magic != self.MAGIC:
                raise ValueError(f"Bad magic: {magic}")
            
            self.max_concepts = max_concepts
            self.embed_dim = embed_dim
            self.concept_count = concept_count
            self.free_list_head = free_list_head
            
            print(f"📀 Loaded .siel: {self.concept_count} concepts, {self.max_concepts} max")
    
    def _write_header(self):
        """Write header to file."""
        reserved = b'\x00' * self.RESERVED_SIZE
        header = struct.pack(
            self.FULL_HEADER_FORMAT,
            self.MAGIC,
            self.VERSION,
            self.max_concepts,
            self.embed_dim,
            self.concept_count,
            self.free_list_head,
            reserved
        )
        
        with open(self.filename, 'r+b') as f:
            f.seek(0)
            f.write(header)
    
    def _write_metadata(self, metadata: Dict):
        """Write metadata to file."""
        # Metadata is stored after the concept data
        metadata_json = json.dumps(metadata, indent=2)
        metadata_bytes = metadata_json.encode('utf-8')
        
        # Find the end of concept data
        offset = self.HEADER_SIZE + self.data_size
        
        with open(self.filename, 'r+b') as f:
            f.seek(offset)
            f.write(metadata_bytes)
    
    def _read_metadata(self) -> Dict:
        """Read metadata from file."""
        offset = self.HEADER_SIZE + self.data_size
        
        try:
            with open(self.filename, 'rb') as f:
                f.seek(offset)
                data = f.read()
                return json.loads(data.decode('utf-8'))
        except:
            return {}
    
    def _get_concept_offset(self, concept_id: int) -> int:
        """Get file offset for a concept."""
        return self.HEADER_SIZE + concept_id * self.concept_bytes
    
    def _read_from_disk(self, concept_id: int) -> np.ndarray:
        """Read a single concept from disk."""
        offset = self._get_concept_offset(concept_id)
        
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            data = f.read(self.concept_bytes)
        
        view = np.frombuffer(data, dtype=np.float32, count=self.concept_floats)
        concept = view[::2] + 1j * view[1::2]
        return concept
    
    def _save_to_disk(self, concept_id: int, data: np.ndarray):
        """Save a single concept to disk."""
        offset = self._get_concept_offset(concept_id)
        flat = np.asarray(data, dtype=np.complex64).ravel()
        flat_float = np.zeros(self.concept_floats, dtype=np.float32)
        flat_float[::2] = flat.real.astype(np.float32)
        flat_float[1::2] = flat.imag.astype(np.float32)
        
        with open(self.filename, 'r+b') as f:
            f.seek(offset)
            f.write(flat_float.tobytes())
    
    def read_concept(self, concept_id: int) -> Optional[np.ndarray]:
        """Read a concept (on-demand, cached)."""
        self._access_count += 1
        
        # Check cache
        if concept_id in self._cache:
            self._cache_hits += 1
            return self._cache[concept_id]
        
        self._cache_misses += 1
        
        # Read from disk
        try:
            concept = self._read_from_disk(concept_id)
        except:
            return None
        
        # Cache it (LRU)
        if len(self._cache) >= self._cache_size:
            self._evict_one()
        
        self._cache[concept_id] = concept
        return concept
    
    def write_concept(self, concept_id: int, data: np.ndarray, metadata: Dict = None):
        """Write a concept (cached, flushed later)."""
        self._cache[concept_id] = data
        self._dirty.add(concept_id)
        
        if metadata:
            self.concept_metadata[concept_id] = metadata
    
    def _evict_one(self):
        """Evict the least recently used concept."""
        if not self._cache:
            return
        
        # Simple LRU: evict first key
        oldest = next(iter(self._cache))
        if oldest in self._dirty:
            self._save_to_disk(oldest, self._cache[oldest])
            self._dirty.remove(oldest)
        del self._cache[oldest]
    
    def flush(self):
        """Flush all dirty concepts to disk."""
        for concept_id in list(self._dirty):
            if concept_id in self._cache:
                self._save_to_disk(concept_id, self._cache[concept_id])
        self._dirty.clear()
        self._write_header()
    
    def allocate_concept(self) -> int:
        """Allocate a new concept ID."""
        # Check free list
        if self.free_list_head >= 0:
            concept_id = self.free_list_head
            # Read next free pointer from concept space
            offset = self._get_concept_offset(concept_id)
            with open(self.filename, 'rb') as f:
                f.seek(offset)
                self.free_list_head = struct.unpack('<q', f.read(8))[0]
            self._write_header()
            return concept_id
        
        # Check if we can grow
        if self.concept_count >= self.max_concepts:
            raise RuntimeError(f"Concept store full ({self.max_concepts} concepts)")
        
        concept_id = self.concept_count
        self.concept_count += 1
        self._write_header()
        return concept_id
    
    def free_concept(self, concept_id: int):
        """Free a concept (put on free list)."""
        offset = self._get_concept_offset(concept_id)
        with open(self.filename, 'r+b') as f:
            f.seek(offset)
            f.write(struct.pack('<q', self.free_list_head))
        self.free_list_head = concept_id
        self._write_header()
        
        # Remove from cache
        if concept_id in self._cache:
            del self._cache[concept_id]
        if concept_id in self.concept_metadata:
            del self.concept_metadata[concept_id]
    
    def add_to_replay(self, concept_id: int, code: str, affinity: float):
        """Add to replay buffer."""
        concept_data = self.read_concept(concept_id)
        if concept_data is not None:
            self.replay_buffer.append({
                'concept_id': concept_id,
                'code': code[:1000],
                'affinity': affinity,
                'timestamp': time.time(),
                'data': concept_data
            })
    
    def sample_replay(self, batch_size: int) -> List[Dict]:
        """Sample from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        import random
        return random.sample(list(self.replay_buffer), batch_size)
    
    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            'concept_count': self.concept_count,
            'max_concepts': self.max_concepts,
            'cache_size': len(self._cache),
            'dirty_count': len(self._dirty),
            'access_count': self._access_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_ratio': self._cache_hits / max(1, self._access_count),
            'replay_size': len(self.replay_buffer),
            'file_size_mb': os.path.getsize(self.filename) / (1024 * 1024),
        }
    
    def close(self):
        """Flush and close."""
        self.flush()
        if self.file_obj:
            self.file_obj.close()