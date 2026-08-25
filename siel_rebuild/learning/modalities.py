"""
Modalities — Detect file types and content types.
Uses magic bytes, extensions, and oscillator resonance.
"""

import os
import re
import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG


class Modality(Enum):
    """All supported modalities."""
    # Text & Code
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    HTML = "html"
    CSS = "css"
    XML = "xml"
    
    # Images
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_GIF = "image/gif"
    IMAGE_BMP = "image/bmp"
    IMAGE_WEBP = "image/webp"
    IMAGE_SVG = "image/svg"
    
    # Video
    VIDEO_MP4 = "video/mp4"
    VIDEO_AVI = "video/avi"
    VIDEO_MOV = "video/mov"
    VIDEO_MKV = "video/mkv"
    VIDEO_WEBM = "video/webm"
    
    # Audio
    AUDIO_WAV = "audio/wav"
    AUDIO_MP3 = "audio/mp3"
    AUDIO_FLAC = "audio/flac"
    AUDIO_OGG = "audio/ogg"
    
    # 3D Models
    MODEL_3D_OBJ = "model/obj"
    MODEL_3D_STL = "model/stl"
    MODEL_3D_GLTF = "model/gltf"
    
    # Executables
    EXECUTABLE_ELF = "executable/elf"
    EXECUTABLE_PE = "executable/pe"
    EXECUTABLE_MACHO = "executable/macho"
    
    # Documents
    DOCUMENT_PDF = "document/pdf"
    DOCUMENT_DOCX = "document/docx"
    DOCUMENT_TXT = "document/txt"
    DOCUMENT_RTF = "document/rtf"
    
    # Archives
    ARCHIVE_ZIP = "archive/zip"
    ARCHIVE_TAR = "archive/tar"
    ARCHIVE_GZ = "archive/gz"
    
    # ML Models
    MODEL_GGUF = "model/gguf"
    MODEL_PYTORCH = "model/pytorch"
    MODEL_SAFETENSORS = "model/safetensors"
    
    # Databases
    DATABASE_SQLITE = "database/sqlite"
    DATABASE_PARQUET = "database/parquet"
    DATABASE_ARROW = "database/arrow"
    DATABASE_CSV = "database/csv"
    
    # Misc
    BINARY = "binary/octet-stream"
    UNKNOWN = "unknown"


class ModalityDetector:
    """
    Detect modality using:
    1. Magic bytes (binary signatures)
    2. File extension
    3. Oscillator resonance (for code vs text)
    """
    
    # Magic byte signatures
    MAGIC_SIGNATURES = {
        (b'\x89PNG\r\n\x1a\n', 8): Modality.IMAGE_PNG,
        (b'\xff\xd8\xff', 3): Modality.IMAGE_JPEG,
        (b'GIF87a', 6): Modality.IMAGE_GIF,
        (b'GIF89a', 6): Modality.IMAGE_GIF,
        (b'BM', 2): Modality.IMAGE_BMP,
        (b'RIFF', 4): Modality.IMAGE_WEBP,
        (b'<svg', 4): Modality.IMAGE_SVG,
        (b'\x1aE\xdf\xa3', 4): Modality.VIDEO_MKV,
        (b'\x1aE\xdf\xa3', 4): Modality.VIDEO_WEBM,
        (b'fLaC', 4): Modality.AUDIO_FLAC,
        (b'OggS', 4): Modality.AUDIO_OGG,
        (b'ID3', 3): Modality.AUDIO_MP3,
        (b'\x7fELF', 4): Modality.EXECUTABLE_ELF,
        (b'MZ', 2): Modality.EXECUTABLE_PE,
        (b'\xca\xfe\xba\xbe', 4): Modality.EXECUTABLE_MACHO,
        (b'%PDF', 4): Modality.DOCUMENT_PDF,
        (b'PK\x03\x04', 4): Modality.ARCHIVE_ZIP,
        (b'\x1f\x8b\x08', 3): Modality.ARCHIVE_GZ,
        (b'SQLite format 3\x00', 16): Modality.DATABASE_SQLITE,
        (b'GGUF', 4): Modality.MODEL_GGUF,
        (b'\x80\x02', 2): Modality.MODEL_PYTORCH,
        (b'\x08\x05', 2): Modality.MODEL_SAFETENSORS,
    }
    
    # Extension to modality map
    EXTENSION_MAP = {
        '.py': Modality.CODE,
        '.js': Modality.CODE,
        '.ts': Modality.CODE,
        '.go': Modality.CODE,
        '.rs': Modality.CODE,
        '.cpp': Modality.CODE,
        '.c': Modality.CODE,
        '.java': Modality.CODE,
        '.rb': Modality.CODE,
        '.php': Modality.CODE,
        '.swift': Modality.CODE,
        '.kt': Modality.CODE,
        '.sh': Modality.CODE,
        '.sql': Modality.CODE,
        '.txt': Modality.TEXT,
        '.md': Modality.MARKDOWN,
        '.json': Modality.JSON,
        '.yaml': Modality.YAML,
        '.yml': Modality.YAML,
        '.toml': Modality.TOML,
        '.html': Modality.HTML,
        '.htm': Modality.HTML,
        '.css': Modality.CSS,
        '.xml': Modality.XML,
        '.png': Modality.IMAGE_PNG,
        '.jpg': Modality.IMAGE_JPEG,
        '.jpeg': Modality.IMAGE_JPEG,
        '.gif': Modality.IMAGE_GIF,
        '.bmp': Modality.IMAGE_BMP,
        '.webp': Modality.IMAGE_WEBP,
        '.svg': Modality.IMAGE_SVG,
        '.mp4': Modality.VIDEO_MP4,
        '.avi': Modality.VIDEO_AVI,
        '.mov': Modality.VIDEO_MOV,
        '.mkv': Modality.VIDEO_MKV,
        '.webm': Modality.VIDEO_WEBM,
        '.wav': Modality.AUDIO_WAV,
        '.mp3': Modality.AUDIO_MP3,
        '.flac': Modality.AUDIO_FLAC,
        '.ogg': Modality.AUDIO_OGG,
        '.obj': Modality.MODEL_3D_OBJ,
        '.stl': Modality.MODEL_3D_STL,
        '.gltf': Modality.MODEL_3D_GLTF,
        '.elf': Modality.EXECUTABLE_ELF,
        '.exe': Modality.EXECUTABLE_PE,
        '.pdf': Modality.DOCUMENT_PDF,
        '.docx': Modality.DOCUMENT_DOCX,
        '.rtf': Modality.DOCUMENT_RTF,
        '.zip': Modality.ARCHIVE_ZIP,
        '.tar': Modality.ARCHIVE_TAR,
        '.gz': Modality.ARCHIVE_GZ,
        '.gguf': Modality.MODEL_GGUF,
        '.pt': Modality.MODEL_PYTORCH,
        '.pth': Modality.MODEL_PYTORCH,
        '.safetensors': Modality.MODEL_SAFETENSORS,
        '.db': Modality.DATABASE_SQLITE,
        '.sqlite': Modality.DATABASE_SQLITE,
        '.parquet': Modality.DATABASE_PARQUET,
        '.arrow': Modality.DATABASE_ARROW,
        '.csv': Modality.DATABASE_CSV,
    }
    
    @classmethod
    def detect(cls, data: bytes, filename: Optional[str] = None,
               wave_field=None) -> Modality:
        """Detect modality from bytes + filename."""
        if len(data) < 4:
            return Modality.UNKNOWN
        
        # 1. Check magic bytes
        for (magic, length), modality in cls.MAGIC_SIGNATURES.items():
            if len(data) >= length and data[:length] == magic:
                # Special cases
                if modality == Modality.IMAGE_WEBP and data[8:12] == b'WEBP':
                    return Modality.IMAGE_WEBP
                if modality == Modality.AUDIO_MP3 and data[:3] == b'ID3':
                    return Modality.AUDIO_MP3
                return modality
        
        # 2. Check extension
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext in cls.EXTENSION_MAP:
                return cls.EXTENSION_MAP[ext]
        
        # 3. If text, use oscillator resonance for code vs text
        if cls._is_text(data):
            if wave_field is not None:
                modality = cls._detect_with_oscillator(data, wave_field)
                if modality in (Modality.CODE, Modality.TEXT):
                    return modality
            
            # Fallback: check for code indicators
            try:
                text = data[:2000].decode('utf-8', errors='ignore')
                code_indicators = [
                    'def ', 'class ', 'import ', 'function ',
                    'print(', 'console.log', 'return ', 'if ',
                    'else ', 'elif ', 'for ', 'while ', 'try '
                ]
                if any(ind in text for ind in code_indicators):
                    return Modality.CODE
            except:
                pass
            return Modality.TEXT
        
        return Modality.BINARY
    
    @classmethod
    def _is_text(cls, data: bytes) -> bool:
        """Check if data is text."""
        sample = data[:1000]
        if not sample:
            return False
        printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
        return printable / len(sample) > 0.85
    
    @classmethod
    def _detect_with_oscillator(cls, data: bytes, wave_field) -> Modality:
        """Use oscillator resonance to detect code vs text."""
        try:
            text = data[:2000].decode('utf-8', errors='ignore')
            # Use simple heuristic for now
            # (DualStreamTrainer will handle real detection)
            code_score = 0
            text_score = 0
            
            code_patterns = ['def', 'class', 'import', 'return', 'if', 'for', 'while']
            text_patterns = ['the', 'and', 'to', 'of', 'for', 'with', 'on', 'at']
            
            for pattern in code_patterns:
                if pattern in text:
                    code_score += 1
            
            for pattern in text_patterns:
                if pattern in text:
                    text_score += 1
            
            if code_score > text_score * 1.5:
                return Modality.CODE
            elif text_score > code_score * 1.5:
                return Modality.TEXT
            
            return Modality.UNKNOWN
        except:
            return Modality.UNKNOWN