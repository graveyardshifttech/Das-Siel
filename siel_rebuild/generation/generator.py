"""
Generator — Unified generation for ALL outputs.
Bundles: Code, Text, Images, Video, 3D, Audio, GGUF, EXE, Documents
Plus Runtime execution with learning feedback.
"""
try:
    from ..utils.sandbox import sandbox
except ImportError:
    from utils.sandbox import sandbox

try:
    from ..utils.security import resource_limits, sanitize_code, detect_secrets
except ImportError:
    from utils.security import resource_limits, sanitize_code, detect_secrets

try:
    from ..utils.state import state_manager
except ImportError:
    from utils.state import state_manager

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG

try:
    from ..core.wave import WaveField
except ImportError:
    from core.wave import WaveField

try:
    from ..utils.trig import SIN_LUT
except ImportError:
    from utils.trig import SIN_LUT

try:
    from ..utils.logger import logger
except ImportError:
    from utils.logger import logger

import os
import sys
import json
import time
import io
import contextlib
import subprocess
import tempfile
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import numpy as np

# Optional imports with graceful fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import soundfile as sf
    import librosa
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


class Generator:
    """
    Unified generator for ALL outputs.
    Code, Text, Images, Video, 3D, Audio, GGUF, EXE, Documents.
    Plus Runtime execution with learning feedback.
    """
    
    def __init__(self, wave: WaveField, memory=None):
        self.wave = wave
        self.memory = memory
        self.execution_count = 0
        self.success_count = 0
        
        # Code generation cache
        self.completion_cache = {}
        
        # Track generated outputs for learning
        self.generation_history = []
        
        # Runtime sandbox
        self.sandbox_path = CONFIG.SANDBOX_DIR
        os.makedirs(self.sandbox_path, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CODE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    def generate_code(self, prompt: str, max_tokens: int = 100,
    language: str = "python", temperature: float = 0.7) -> str:
        """
        Generate code from oscillator states with language-specific formatting.
        
        Args:
            prompt: Code prefix to complete
            max_tokens: Maximum tokens to generate
            language: Programming language (python, javascript, go, rust, etc.)
            temperature: Randomness (0 = deterministic, 1 = creative)
        """
        cache_key = hashlib.md5(f"{prompt}_{language}_{max_tokens}".encode()).hexdigest()[:8]
        if cache_key in self.completion_cache:
            return self.completion_cache[cache_key]
        
        # Propagate wave to settle
        self.wave.propagate_hybrid(dt=0.02, steps=3)
        
        generated = prompt
        
        for i in range(max_tokens):
            # Get oscillator state
            y = (i % CONFIG.GRID_SIZE)
            x = (i // CONFIG.GRID_SIZE) % CONFIG.GRID_SIZE
            phase = self.wave.phase_array[y, x]
            
            # Non-linear mapping (phase → char)
            # Use LUT sin for faster computation
            sin_phase = SIN_LUT.sin(np.array([phase]))[0]
            
            # Map to character range
            if language == "python":
                char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \n\t()[]{}:;=+-*/%<>!&|^~.,\"'#@$"
            elif language == "javascript":
                char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \n\t()[]{}:;=+-*/%<>!&|^~.,\"'#@$"
            else:
                char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \n\t()[]{}:;=+-*/%<>!&|^~.,\"'#@$"
            
            # Map sin_phase (-1 to 1) to index in char_set
            char_idx = int(((sin_phase + 1) / 2) * (len(char_set) - 1))
            char_idx = max(0, min(char_idx, len(char_set) - 1))
            char_idx = int(char_idx + temperature * np.random.randn() * 5)
            char_idx = max(0, min(char_idx, len(char_set) - 1))
            
            generated += char_set[char_idx]
            
            # Propagate wave every few tokens
            if i % 3 == 0:
                self.wave.propagate_hybrid(dt=0.01, steps=2)
            
            # Stop at natural boundaries
            if len(generated) > len(prompt) + 10 and generated[-1] in '.;}]\n':
                break
        
        self.completion_cache[cache_key] = generated
        
        # Store generation history for learning
        self.generation_history.append({
            'type': 'code',
            'prompt': prompt[:100],
            'output': generated[:200],
            'language': language,
            'timestamp': time.time()
        })
        
        return generated
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEXT/LANGUAGE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_text(self, prompt: str, max_tokens: int = 100,
        style: str = "neutral", temperature: float = 0.8) -> str:
        """
        Generate natural language text.
        
        Args:
            prompt: Text prefix
            max_tokens: Maximum tokens to generate
            style: neutral, formal, creative, technical, poetic
            temperature: Randomness (0 = deterministic, 1 = creative)
        """
        # Propagate wave
        self.wave.propagate_hybrid(dt=0.02, steps=2)
        
        generated = prompt
        
        for i in range(max_tokens):
            # Get oscillator state
            y = (i % CONFIG.GRID_SIZE)
            x = (i // CONFIG.GRID_SIZE) % CONFIG.GRID_SIZE
            phase = self.wave.phase_array[y, x]
            
            # Use LUT sin
            sin_phase = SIN_LUT.sin(np.array([phase]))[0]
            
            # Text character set
            char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-\"'()[]{} \n\t"
            
            # Adjust for style
            if style == "formal":
                char_set += "Furthermore,Additionally,Consequently,Therefore"
            elif style == "creative":
                char_set += "wonderfully,beautifully,magically,extraordinarily"
            elif style == "technical":
                char_set += "function,parameter,process,algorithm,implementation"
            
            char_idx = int(((sin_phase + 1) / 2) * (len(char_set) - 1))
            char_idx = max(0, min(char_idx, len(char_set) - 1))
            char_idx = int(char_idx + temperature * np.random.randn() * 3)
            char_idx = max(0, min(char_idx, len(char_set) - 1))
            
            generated += char_set[char_idx]
            
            if i % 3 == 0:
                self.wave.propagate_hybrid(dt=0.01, steps=2)
            
            # Stop at sentence boundaries
            if len(generated) > len(prompt) + 20 and generated[-1] in '.!?':
                break
        
        # Store generation history
        self.generation_history.append({
            'type': 'text',
            'prompt': prompt[:100],
            'output': generated[:200],
            'style': style,
            'timestamp': time.time()
        })
        
        return generated
    
    # ═══════════════════════════════════════════════════════════════════════
    # IMAGE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_image(self, width: int = 512, height: int = 512,
                    cmap: str = "viridis") -> Optional[np.ndarray]:
        """Generate image from wave snapshot."""
        if not HAS_PIL:
            logger.log("PIL not installed for image generation", "ERROR")
            return None
        
        snapshot = self.wave.get_wave_snapshot()
        
        # Normalize to 0-255
        snapshot = (snapshot - snapshot.min()) / (snapshot.max() - snapshot.min() + 1e-8)
        snapshot = (snapshot * 255).astype(np.uint8)
        
        # Resize
        img = Image.fromarray(snapshot)
        img = img.resize((width, height), Image.LANCZOS)
        
        return np.array(img)
    
    def generate_image_and_save(self, path: str = "generated_image.png",
    width: int = 512, height: int = 512) -> bool:
        """Generate and save image."""
        img_array = self.generate_image(width, height)
        if img_array is None:
            return False
        
        img = Image.fromarray(img_array)
        img.save(path)
        logger.log(f"Generated image: {path}")
        
        self.generation_history.append({
            'type': 'image',
            'path': path,
            'width': width,
            'height': height,
            'timestamp': time.time()
        })
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # VIDEO GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_video(self, frames: int = 30, width: int = 256, height: int = 256,
                      dt: float = 0.02) -> List[np.ndarray]:
        """Generate video frames from wave evolution."""
        if not HAS_CV2:
            logger.log("OpenCV not installed for video generation", "ERROR")
            return []
        
        video_frames = []
        snapshot = self.wave.get_wave_snapshot()
        
        for i in range(frames):
            # Get current snapshot
            snapshot = self.wave.get_wave_snapshot()
            
            # Normalize and resize
            snapshot = (snapshot - snapshot.min()) / (snapshot.max() - snapshot.min() + 1e-8)
            snapshot = (snapshot * 255).astype(np.uint8)
            
            if HAS_PIL:
                img = Image.fromarray(snapshot)
                img = img.resize((width, height), Image.LANCZOS)
                video_frames.append(np.array(img))
            
            # Propagate wave
            self.wave.propagate_hybrid(dt=dt, steps=2)
        
        self.generation_history.append({
            'type': 'video',
            'frames': frames,
            'width': width,
            'height': height,
            'timestamp': time.time()
        })
        
        return video_frames
    
    def generate_video_and_save(self, path: str = "generated_video.mp4",
                                frames: int = 30, width: int = 256, height: int = 256) -> bool:
        """Generate and save video."""
        if not HAS_CV2:
            logger.log("OpenCV not installed", "ERROR")
            return False
        
        video_frames = self.generate_video(frames, width, height)
        if not video_frames:
            return False
        
        out = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10,
            (width, height)
        )
        
        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        logger.log(f"Generated video: {path}")
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3D GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_3d(self, resolution: int = 32) -> Optional[np.ndarray]:
        """Generate 3D point cloud from wave phases."""
        if not HAS_TRIMESH:
            logger.log("Trimesh not installed for 3D generation", "ERROR")
            return None
        
        snapshot = self.wave.get_wave_snapshot()
        
        # Create grid
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Use snapshot as Z
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                if i < snapshot.shape[0] and j < snapshot.shape[1]:
                    Z[i, j] = snapshot[i % snapshot.shape[0], j % snapshot.shape[1]]
        
        # Normalize Z
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        
        return np.stack([X, Y, Z], axis=-1)
    
    def generate_3d_and_save(self, path: str = "generated_3d.obj",
                            resolution: int = 32) -> bool:
        """Generate and save 3D object."""
        if not HAS_TRIMESH:
            logger.log("Trimesh not installed", "ERROR")
            return False
        
        points = self.generate_3d(resolution)
        if points is None:
            return False
        
        cloud = trimesh.points.PointCloud(points.reshape(-1, 3))
        cloud.export(path)
        logger.log(f"Generated 3D object: {path}")
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # AUDIO GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_audio(self, duration: float = 2.0, sample_rate: int = 44100) -> Optional[np.ndarray]:
        """Generate audio from oscillators."""
        if not HAS_AUDIO:
            logger.log("Audio libraries not installed", "ERROR")
            return None
        
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples)
        
        # Use multiple oscillators for rich sound
        for i in range(num_samples):
            phase_sum = 0
            for y in range(min(8, CONFIG.GRID_SIZE)):
                for x in range(min(8, CONFIG.GRID_SIZE)):
                    phase = self.wave.phase_array[y, x] + i * 0.001
                    phase_sum += np.sin(phase)
            audio[i] = phase_sum / 64
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def generate_audio_and_save(self, path: str = "generated_audio.wav",
                            duration: float = 2.0, sample_rate: int = 44100) -> bool:
        """Generate and save audio."""
        if not HAS_AUDIO:
            logger.log("Audio libraries not installed", "ERROR")
            return False
        
        audio = self.generate_audio(duration, sample_rate)
        if audio is None:
            return False
        
        sf.write(path, audio, sample_rate)
        logger.log(f"Generated audio: {path}")
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # GGUF GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_gguf(self, architecture: str = "llama",
        size: str = "small") -> bytes:
        """Generate a valid GGUF container containing model metadata."""
        if not HAS_GGUF:
            logger.log("GGUF not installed", "ERROR")
            return None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as handle:
                temp_path = handle.name
            writer = gguf.GGUFWriter(temp_path, architecture)
            writer.add_name(f"Siel_{architecture}_{size}")
            writer.add_description("Generated by Siel v15")
            writer.add_uint32('general.context_length', 2048)
            writer.add_uint32('general.embedding_length',
                            4096 if size == 'large' else 2048)
            writer.add_uint32('general.block_count',
                    32 if size == 'large' else 16)
            writer.add_license('MIT')
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with open(temp_path, 'rb') as handle:
                return handle.read()
        except (AttributeError, TypeError, OSError) as exc:
            logger.log(f"Unable to create GGUF: {exc}", "ERROR")
            return None
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def generate_gguf_and_save(self, path: str = "generated_model.gguf",
        architecture: str = "llama",
                            size: str = "small") -> bool:
        """Generate and save GGUF metadata."""
        gguf_data = self.generate_gguf(architecture, size)
        if gguf_data is None:
            return False
        
        with open(path, 'wb') as f:
            f.write(gguf_data)
        logger.log(f"Generated GGUF: {path}")
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_exe(self) -> bytes:
        """Generate a minimal valid 64-bit PE console executable."""
        if not HAS_PEFILE:
            logger.log("PEfile not installed", "ERROR")
            return None

        import struct
        pe_offset, headers_size = 0x80, 0x200
        file_alignment, section_alignment = 0x200, 0x1000
        code_rva, image_base, raw_size = 0x1000, 0x140000000, 0x200

        dos = bytearray(pe_offset)
        dos[:2] = b'MZ'
        struct.pack_into('<I', dos, 0x3c, pe_offset)
        coff = struct.pack('<HHIIIHH', 0x8664, 1, int(time.time()), 0, 0,
                    0xF0, 0x0022)
        optional = bytearray(0xF0)
        struct.pack_into('<H', optional, 0, 0x20B)
        struct.pack_into('<I', optional, 16, code_rva)
        struct.pack_into('<I', optional, 20, code_rva)
        struct.pack_into('<Q', optional, 24, image_base)
        struct.pack_into('<I', optional, 32, section_alignment)
        struct.pack_into('<I', optional, 36, file_alignment)
        struct.pack_into('<I', optional, 56, 0x2000)
        struct.pack_into('<I', optional, 60, headers_size)
        struct.pack_into('<H', optional, 68, 3)
        struct.pack_into('<I', optional, 108, 16)
        section = struct.pack('<8sIIIIIIHHI', b'.text\0\0\0', 1, code_rva,
                    raw_size, headers_size, 0, 0, 0, 0, 0x60000020)
        header = bytes(dos) + b'PE\0\0' + coff + bytes(optional) + section
        return header.ljust(headers_size, b'\0') + b'\x31\xc0\xc3'.ljust(raw_size, b'\0')
    
    def generate_exe_and_save(self, path: str = "generated_program.exe") -> bool:
        """Generate and save EXE stub."""
        exe_data = self.generate_exe()
        if exe_data is None:
            return False
        
        with open(path, 'wb') as f:
            f.write(exe_data)
        logger.log(f"Generated EXE: {path}")
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # DOCUMENT GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_document(self, content: str = None, doc_type: str = "txt") -> str:
        """Generate document content."""
        if content is None:
            content = "This is a generated document from Siel v15."

        normalized = (content or "").strip()
        if not normalized:
            normalized = "This is a generated document from Siel v15."

        if doc_type == "txt":
            return normalized
        elif doc_type == "md":
            return f"# Generated document\n\n{normalized}\n"
        elif doc_type == "html":
            return f"<html><body><p>{normalized}</p></body></html>"
        elif doc_type == "pdf":
            return f"PDF: {normalized}"
        elif doc_type == "docx":
            return f"DOCX: {normalized}"
        else:
            return normalized

    # ═══════════════════════════════════════════════════════════════════════
    # RUNTIME EXECUTION (WITH LEARNING FEEDBACK)
    # ═══════════════════════════════════════════════════════════════════════

    def _execute_direct(self, code: str, exec_type: str = 'python',
                        timeout: int = 30) -> Dict:
        """Execute code directly in-process with a guarded namespace."""
        try:
            ns = {'__builtins__': __builtins__}
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                compiled = compile(code, '<generator>', 'exec')
                exec(compiled, ns, ns)
            result = {
                'success': True,
                'stdout': stdout.getvalue(),
                'stderr': stderr.getvalue(),
                'returncode': 0,
                'exec_type': exec_type,
            }
            return result
        except Exception as exc:
            return {
                'success': False,
                'stdout': stdout.getvalue() if 'stdout' in locals() else '',
                'stderr': (stderr.getvalue() if 'stderr' in locals() else '') + str(exc),
                'returncode': 1,
                'exec_type': exec_type,
            }

    def _execute_sandboxed(self, code: str, exec_type: str = 'python',
                        timeout: int = 30) -> Dict:
        """Execute code in a temporary sandbox directory."""
        tmpdir = tempfile.mkdtemp(prefix='siel_exec_', dir=self.sandbox_path)
        script_path = os.path.join(tmpdir, 'run.py' if exec_type == 'python' else 'run.txt')
        with open(script_path, 'w', encoding='utf-8') as handle:
            handle.write(code)

        try:
            if exec_type == 'python':
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, 30),
                    cwd=tmpdir,
                )
            else:
                proc = subprocess.run(
                    ['bash', '-lc', code],
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, 30),
                    cwd=tmpdir,
                )
            result = {
                'success': proc.returncode == 0,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'returncode': proc.returncode,
                'exec_type': exec_type,
            }
            return result
        except subprocess.TimeoutExpired as exc:
            return {
                'success': False,
                'stdout': exc.stdout or '',
                'stderr': f'Timeout after {timeout}s',
                'returncode': 124,
                'exec_type': exec_type,
            }

# Replace execute method:

    def execute(self, code: str, exec_type: str = 'python',
            timeout: int = 30, sandboxed: bool = True) -> Dict:
        """
        Execute code SECURELY with sandbox and resource limits.
        """
        # Rate limit check
        if not resource_limits.check_rate_limit():
            return {'error': 'Rate limit exceeded. Please wait.'}

        # Resource check
        if not resource_limits.check_executions():
            return {'error': 'Execution limit reached.'}

        # Sanitize input
        code = sanitize_code(code)

        # Check for secrets
        secrets = detect_secrets(code)
        if secrets:
            logger.log_safety_alert("secrets_detected", f"Potential secrets in code: {secrets[:3]}", "high")
            return {'error': 'Potential secrets detected in code.'}

        # Execute with sandbox
        result = sandbox.execute(code, exec_type, timeout, sandboxed)

        # Update stats
        self.execution_count += 1
        if result.get('success', False):
            self.success_count += 1

        # Save to state
        state_manager.save('last_execution', {
            'timestamp': time.time(),
            'code_hash': hashlib.md5(code.encode()).hexdigest()[:16],
            'success': result.get('success', False)
        })

        return result

    def _learn_from_execution(self, code: str, result: Dict, signal: float):
        """Inject execution feedback into oscillators."""
        for y in range(CONFIG.GRID_SIZE):
            for x in range(CONFIG.GRID_SIZE):
                self.wave.phase_array[y, x] += signal * 0.01 * np.sin(self.wave.phase_array[y, x])
                self.wave.phase_array[y, x] %= (2 * np.pi)

        if self.memory:
            self.memory.remember(
                'execution',
                {
                    'code': code[:200],
                    'success': result.get('success', False),
                    'stdout': result.get('stdout', '')[:200],
                    'stderr': result.get('stderr', '')[:200],
                    'signal': signal,
                },
                importance=0.5 + abs(signal) * 0.5,
            )

        logger.log_learning_update(
            'execution_feedback',
            'wavefield',
            abs(signal),
            'positive' if signal > 0 else 'negative',
        )

    def get_success_rate(self) -> float:
        """Get execution success rate."""
        return self.success_count / max(1, self.execution_count)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_all(self, prompt: str = "hello world", output_dir: str = "./generated") -> Dict:
        """Generate ALL types at once."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Code
        code = self.generate_code(prompt, max_tokens=50)
        results['code'] = code
        with open(f"{output_dir}/generated_code.py", 'w') as f:
            f.write(code)
        
        # Text
        text = self.generate_text(prompt, max_tokens=100)
        results['text'] = text
        with open(f"{output_dir}/generated_text.txt", 'w') as f:
            f.write(text)
        
        # Image
        if HAS_PIL:
            img = self.generate_image()
            if img is not None:
                Image.fromarray(img).save(f"{output_dir}/generated_image.png")
                results['image'] = f"{output_dir}/generated_image.png"
        
        # Audio
        if HAS_AUDIO:
            audio = self.generate_audio()
            if audio is not None:
                sf.write(f"{output_dir}/generated_audio.wav", audio, 44100)
                results['audio'] = f"{output_dir}/generated_audio.wav"
        
        # 3D
        if HAS_TRIMESH:
            points = self.generate_3d()
            if points is not None:
                cloud = trimesh.points.PointCloud(points.reshape(-1, 3))
                cloud.export(f"{output_dir}/generated_3d.obj")
                results['3d'] = f"{output_dir}/generated_3d.obj"
        
        # GGUF
        gguf_data = self.generate_gguf()
        if gguf_data is not None:
            with open(f"{output_dir}/generated_model.gguf", 'wb') as f:
                f.write(gguf_data)
            results['gguf'] = f"{output_dir}/generated_model.gguf"
        
        results['timestamp'] = time.time()
        
        logger.log(f"Generated ALL outputs to {output_dir}")
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        """Get generator statistics."""
        return {
            'executions': self.execution_count,
            'success_rate': self.get_success_rate(),
            'generations': len(self.generation_history),
            'cache_size': len(self.completion_cache),
            'last_generation': self.generation_history[-1] if self.generation_history else None,
        }