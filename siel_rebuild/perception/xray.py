"""
X-Ray Vision 2.0 — See inside ANYTHING.
- Web pages (HTML/CSS/JS extraction)
- Binary executables (PE/ELF/Mach-O parsing)
- Running processes (Linux/Windows)
- Network connections
- Filesystem traversal (with context)
- GGUF model inspection
- Video/Audio metadata
- Document parsing (PDF, DOCX)
"""

import os
import sys
import json
import time
import re
import socket
import subprocess
import urllib.request
import urllib.parse
import ssl
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import deque
import numpy as np

try:
    from ..config import CONFIG
except ImportError:
    from config import CONFIG

try:
    from ..utils.logger import logger
except ImportError:
    from utils.logger import logger

# Optional imports with graceful fallback
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
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False



class XRayVision2:
    """
    Advanced perception system that can SEE inside:
    - Web pages (HTML/CSS/JS)
    - Filesystem (with context and structure)
    - Binary executables (PE/ELF/Mach-O)
    - Running processes (Linux/Windows)
    - Network connections
    - GGUF models
    - Images, Video, Audio
    - Documents (PDF, DOCX)
    """
    
    def __init__(self):
        self.vision_history = deque(maxlen=200)
        self.knowledge_state: Dict[str, Any] = {}
        self.sandbox_path = CONFIG.SANDBOX_DIR
        os.makedirs(self.sandbox_path, exist_ok=True)
        
        # Filesystem cache
        self.fs_cache = {}
        
        # X-Ray session
        self.session = {}
        
        # Safety
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit for text parsing
    
    def perceive(self, target: str, detail: str = None) -> Dict:
        """Perceive ANYTHING with X-ray vision."""
        results = {
            'target': target,
            'timestamp': time.time(),
            'success': False,
            'data': None,
            'depth': detail or 'shallow'
        }
        
        prev_state = self.knowledge_state.get(target, {})
        
        try:
            # Route to appropriate scanner
            if target == 'sandbox':
                results['data'] = self._scan_sandbox()
            elif target == 'webpage':
                results['data'] = self._scan_webpage(detail)
            elif target == 'process':
                results['data'] = self._scan_process(detail)
            elif target == 'binary':
                results['data'] = self._scan_binary(detail)
            elif target == 'network':
                results['data'] = self._scan_network()
            elif target == 'filesystem':
                results['data'] = self._scan_filesystem(detail)
            elif target == 'gguf':
                results['data'] = self._scan_gguf(detail)
            elif target == 'image':
                results['data'] = self._scan_image(detail)
            elif target == 'video':
                results['data'] = self._scan_video(detail)
            elif target == 'audio':
                results['data'] = self._scan_audio(detail)
            elif target == 'document':
                results['data'] = self._scan_document(detail)
            elif target == 'everything':
                results['data'] = self._scan_everything()
            else:
                # Try to interpret as file or URL
                results['data'] = self._scan_generic(target, detail)
            
            results['success'] = True
            results['info_gain'] = self._compute_info_gain(prev_state, results['data'])
            self.knowledge_state[target] = results['data']
            
        except Exception as e:
            results['data'] = {'error': str(e)}
            logger.log(f"X-Ray error: {e}", "ERROR")
        
        self.vision_history.append(results)
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # FILESYSTEM SCANNER — WITH CONTEXT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_filesystem(self, path: str = None) -> Dict:
        """
        Scan filesystem with context.
        Shows structure, file types, sizes, and relationships.
        """
        if path is None:
            path = "."
        
        abs_path = os.path.abspath(path)
        
        if not os.path.exists(abs_path):
            return {'error': f'Path not found: {abs_path}'}
        
        # Use cache
        cache_key = abs_path
        if cache_key in self.fs_cache and time.time() - self.fs_cache[cache_key]['timestamp'] < 60:
            return self.fs_cache[cache_key]['data']
        
        result = {
            'path': abs_path,
            'exists': True,
            'is_dir': os.path.isdir(abs_path),
            'is_file': os.path.isfile(abs_path),
            'size': 0,
            'modified': time.ctime(os.path.getmtime(abs_path)),
            'children': [],
            'type_distribution': {},
            'total_size': 0,
            'file_count': 0,
            'dir_count': 0,
        }
        
        try:
            stat = os.stat(abs_path)
            result['size'] = stat.st_size
            result['permissions'] = oct(stat.st_mode)[-3:]
            result['owner'] = stat.st_uid
        except:
            pass
        
        if os.path.isdir(abs_path):
            # Scan directory
            try:
                for entry in os.listdir(abs_path)[:100]:  # Limit for performance
                    full_path = os.path.join(abs_path, entry)
                    try:
                        stat = os.stat(full_path)
                        is_dir = os.path.isdir(full_path)
                        
                        child = {
                            'name': entry,
                            'is_dir': is_dir,
                            'size': stat.st_size,
                            'modified': time.ctime(stat.st_mtime),
                            'type': self._guess_file_type(entry),
                        }
                        
                        result['children'].append(child)
                        result['total_size'] += stat.st_size
                        
                        if is_dir:
                            result['dir_count'] += 1
                        else:
                            result['file_count'] += 1
                            # Track type distribution
                            ext = os.path.splitext(entry)[1].lower()
                            if ext:
                                result['type_distribution'][ext] = result['type_distribution'].get(ext, 0) + 1
                    except:
                        pass
            except:
                pass
        
        # Cache
        self.fs_cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def _guess_file_type(self, filename: str) -> str:
        """Guess file type from extension."""
        ext = os.path.splitext(filename)[1].lower()
        
        type_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.java': 'Java',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.html': 'HTML',
            '.css': 'CSS',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.xml': 'XML',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.pdf': 'PDF',
            '.docx': 'Word',
            '.xlsx': 'Excel',
            '.pptx': 'PowerPoint',
            '.png': 'PNG Image',
            '.jpg': 'JPEG Image',
            '.jpeg': 'JPEG Image',
            '.gif': 'GIF Image',
            '.bmp': 'BMP Image',
            '.svg': 'SVG Image',
            '.webp': 'WebP Image',
            '.mp4': 'MP4 Video',
            '.avi': 'AVI Video',
            '.mov': 'MOV Video',
            '.mkv': 'MKV Video',
            '.webm': 'WebM Video',
            '.wav': 'WAV Audio',
            '.mp3': 'MP3 Audio',
            '.flac': 'FLAC Audio',
            '.ogg': 'OGG Audio',
            '.obj': '3D Model (OBJ)',
            '.stl': '3D Model (STL)',
            '.gltf': '3D Model (GLTF)',
            '.gguf': 'GGUF Model',
            '.pt': 'PyTorch Model',
            '.pth': 'PyTorch Model',
            '.safetensors': 'Safetensors Model',
            '.db': 'SQLite Database',
            '.sqlite': 'SQLite Database',
            '.parquet': 'Parquet Database',
            '.arrow': 'Arrow Database',
            '.csv': 'CSV Data',
            '.zip': 'ZIP Archive',
            '.tar': 'TAR Archive',
            '.gz': 'GZ Archive',
            '.exe': 'Windows Executable',
            '.dll': 'Windows DLL',
            '.so': 'Linux Shared Library',
            '.dylib': 'macOS Library',
        }
        
        return type_map.get(ext, 'Unknown')
    
    # ═══════════════════════════════════════════════════════════════════════
    # BINARY SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_binary(self, path: str) -> Dict:
        """Scan a binary executable."""
        if not os.path.exists(path):
            return {'error': f'File not found: {path}'}
        
        result = {
            'path': path,
            'type': 'unknown',
            'size': os.path.getsize(path),
            'strings': [],
            'sections': [],
            'symbols': [],
            'entropy': 0,
            'is_pe': False,
            'is_elf': False,
            'is_macho': False,
        }
        
        try:
            with open(path, 'rb') as f:
                data = f.read(1024 * 1024)  # 1MB header
                
                # Detect type
                if data[:4] == b'\x7fELF':
                    result['type'] = 'ELF'
                    result['is_elf'] = True
                    
                    # Extract ELF info
                    e_type = data[16:18]
                    e_type_map = {
                        b'\x00\x02': 'Executable',
                        b'\x00\x03': 'Shared Library',
                        b'\x00\x04': 'Core Dump'
                    }
                    result['elf_type'] = e_type_map.get(e_type, 'Unknown')
                    
                elif data[:2] == b'MZ':
                    result['type'] = 'PE'
                    result['is_pe'] = True
                    
                    if HAS_PEFILE:
                        try:
                            pe = pefile.PE(data=data)
                            for section in pe.sections[:20]:
                                result['sections'].append({
                                    'name': section.Name.decode('utf-8', errors='ignore').strip('\x00'),
                                    'size': section.SizeOfRawData,
                                    'virtual_size': section.Misc_VirtualSize,
                                    'entropy': section.get_entropy() if hasattr(section, 'get_entropy') else 0
                                })
                            
                            # Get imports
                            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                                result['imports'] = [
                                    dll.dll.decode('utf-8', errors='ignore')
                                    for dll in pe.DIRECTORY_ENTRY_IMPORT[:20]
                                ]
                        except:
                            pass
                
                elif data[:4] == b'\xca\xfe\xba\xbe':
                    result['type'] = 'Mach-O'
                    result['is_macho'] = True
                
                # Extract strings (ASCII printable)
                strings = re.findall(b'[\\x20-\\x7E]{4,}', data)
                result['strings'] = [s.decode('utf-8', errors='ignore') for s in strings[:50]]
                
                # Calculate entropy
                if len(data) > 0:
                    hist = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
                    hist = hist / len(data)
                    result['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-8)))
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # WEBPAGE SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_webpage(self, url: str) -> Dict:
        """Scan a webpage."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        result = {
            'url': url,
            'html': '',
            'text': '',
            'links': [],
            'scripts': [],
            'styles': [],
            'images': [],
            'metadata': {},
            'title': '',
            'headings': [],
            'code_blocks': [],
        }
        
        try:
            # Create SSL context that ignores cert errors (for deep scanning)
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Siel-XRay/2.0 (Security Research)'}
            )
            
            with urllib.request.urlopen(req, timeout=10, context=context) as response:
                html = response.read().decode('utf-8', errors='ignore')
                result['html'] = html[:50000]  # Truncate
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
                if title_match:
                    result['title'] = title_match.group(1)
                
                # Extract text (remove tags)
                text = re.sub(r'<[^>]+>', ' ', html)
                text = ' '.join(text.split())
                result['text'] = text[:5000]
                
                # Extract links
                links = re.findall(r'href=[\'"]?([^\'" >]+)', html)
                result['links'] = [l for l in links if l.startswith('http')][:50]
                
                # Extract scripts
                scripts = re.findall(r'<script[^>]*src=[\'"]?([^\'" >]+)', html)
                result['scripts'] = scripts[:20]
                
                # Extract styles
                styles = re.findall(r'<link[^>]*rel=[\'"]?stylesheet[\'"]?[^>]*href=[\'"]?([^\'" >]+)', html)
                result['styles'] = styles[:20]
                
                # Extract images
                images = re.findall(r'<img[^>]*src=[\'"]?([^\'" >]+)', html)
                result['images'] = images[:20]
                
                # Extract headings
                headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html, re.IGNORECASE)
                result['headings'] = [h.strip() for h in headings[:10]]
                
                # Extract code blocks
                code_blocks = re.findall(r'<code[^>]*>(.*?)</code>', html, re.IGNORECASE | re.DOTALL)
                result['code_blocks'] = [c.strip()[:200] for c in code_blocks[:5]]
                
                # Metadata
                meta = re.findall(r'<meta[^>]*name=[\'"]?([^\'"]+)[\'"][^>]*content=[\'"]?([^\'"]+)', html)
                for name, content in meta[:10]:
                    result['metadata'][name] = content
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_process(self, pid: str) -> Dict:
        """Scan a running process."""
        result = {
            'pid': pid,
            'name': '',
            'cmdline': '',
            'memory': {},
            'threads': [],
            'open_files': [],
            'environment': {},
            'cpu_usage': 0,
            'memory_usage': 0,
        }
        
        try:
            pid = int(pid)
            
            if sys.platform == 'linux':
                # Process name
                try:
                    with open(f'/proc/{pid}/comm', 'r') as f:
                        result['name'] = f.read().strip()
                except:
                    pass
                
                # Command line
                try:
                    with open(f'/proc/{pid}/cmdline', 'r') as f:
                        result['cmdline'] = f.read().replace('\0', ' ')[:200]
                except:
                    pass
                
                # Memory info
                try:
                    with open(f'/proc/{pid}/status', 'r') as f:
                        for line in f:
                            if 'VmRSS' in line:
                                result['memory']['rss'] = line.strip()
                            elif 'VmSize' in line:
                                result['memory']['size'] = line.strip()
                            elif 'Threads' in line:
                                result['threads'] = line.strip()
                except:
                    pass
                
                # Open files
                try:
                    fd_path = f'/proc/{pid}/fd'
                    if os.path.exists(fd_path):
                        for fd in os.listdir(fd_path)[:20]:
                            try:
                                link = os.readlink(os.path.join(fd_path, fd))
                                result['open_files'].append(link)
                            except:
                                pass
                except:
                    pass
                
                # Environment
                try:
                    with open(f'/proc/{pid}/environ', 'r') as f:
                        env = f.read().replace('\0', '\n')
                        result['environment'] = dict(
                            line.split('=', 1) for line in env.split('\n')[:20]
                            if '=' in line
                        )
                except:
                    pass
                
            elif sys.platform == 'win32':
                # Windows process info using tasklist
                try:
                    output = subprocess.check_output(
                        ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                        text=True,
                        timeout=2
                    )
                    lines = output.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split(',')
                        if len(parts) > 0:
                            result['name'] = parts[0].strip('"')
                except:
                    pass
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_network(self) -> Dict:
        """Scan network connections."""
        result = {
            'connections': [],
            'interfaces': [],
            'hostname': socket.gethostname(),
            'port_scans': {},
        }
        
        try:
            # Linux netstat
            if sys.platform == 'linux':
                netstat = subprocess.run(
                    ['ss', '-tun'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if netstat.returncode == 0:
                    for line in netstat.stdout.split('\n')[1:30]:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 5:
                                result['connections'].append({
                                    'state': parts[0],
                                    'local': parts[3],
                                    'remote': parts[4]
                                })
                
                # Interfaces
                try:
                    import netifaces
                    for iface in netifaces.interfaces():
                        addrs = netifaces.ifaddresses(iface)
                        if netifaces.AF_INET in addrs:
                            result['interfaces'].append({
                                'name': iface,
                                'ip': addrs[netifaces.AF_INET][0]['addr']
                            })
                except:
                    pass
            
            # Windows netstat
            elif sys.platform == 'win32':
                netstat = subprocess.run(
                    ['netstat', '-n'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if netstat.returncode == 0:
                    for line in netstat.stdout.split('\n')[3:30]:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 3:
                                result['connections'].append({
                                    'protocol': parts[0],
                                    'local': parts[1],
                                    'remote': parts[2],
                                    'state': parts[3] if len(parts) > 3 else 'ESTABLISHED'
                                })
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # GGUF SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_gguf(self, path: str) -> Dict:
        """Scan a GGUF model file."""
        result = {
            'path': path,
            'type': 'GGUF',
            'size': os.path.getsize(path),
            'architecture': None,
            'context_length': None,
            'vocab_size': None,
            'hidden_size': None,
            'layers': None,
            'tensors': [],
            'metadata': {},
        }
        
        if not HAS_GGUF:
            result['error'] = 'GGUF library not installed'
            return result
        
        try:
            reader = gguf.GGUFReader(path)
            
            # Extract metadata
            for key, value in reader.metadata.items():
                try:
                    result['metadata'][key] = value.value
                except:
                    result['metadata'][key] = str(value)
            
            # Architecture
            if 'general.architecture' in reader.metadata:
                result['architecture'] = reader.metadata['general.architecture'].value
            
            # Model params
            if 'llama.context_length' in reader.metadata:
                result['context_length'] = reader.metadata['llama.context_length'].value
            if 'llama.vocab_size' in reader.metadata:
                result['vocab_size'] = reader.metadata['llama.vocab_size'].value
            if 'llama.hidden_size' in reader.metadata:
                result['hidden_size'] = reader.metadata['llama.hidden_size'].value
            if 'llama.block_count' in reader.metadata:
                result['layers'] = reader.metadata['llama.block_count'].value
            
            # Tensors (limited)
            for tensor in reader.tensors[:20]:
                result['tensors'].append({
                    'name': tensor.name,
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'size': tensor.n_elements * tensor.dtype.itemsize
                })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # IMAGE SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_image(self, path: str) -> Dict:
        """Scan an image file."""
        if not HAS_PIL:
            return {'error': 'PIL not installed'}
        
        result = {
            'path': path,
            'type': 'Image',
            'size': os.path.getsize(path),
            'width': 0,
            'height': 0,
            'format': None,
            'mode': None,
            'colors': 0,
            'has_alpha': False,
        }
        
        try:
            img = Image.open(path)
            result['width'] = img.width
            result['height'] = img.height
            result['format'] = img.format
            result['mode'] = img.mode
            result['has_alpha'] = img.mode in ('RGBA', 'LA', 'P') and img.info.get('transparency') is not None
            
            # Color count
            if img.mode == 'RGB':
                colors = img.getcolors(maxcolors=256)
                if colors:
                    result['colors'] = len(colors)
            
            img.close()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # VIDEO SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_video(self, path: str) -> Dict:
        """Scan a video file."""
        if not HAS_CV2:
            return {'error': 'OpenCV not installed'}
        
        result = {
            'path': path,
            'type': 'Video',
            'size': os.path.getsize(path),
            'width': 0,
            'height': 0,
            'fps': 0,
            'frames': 0,
            'duration': 0,
            'codec': 'unknown',
            'has_audio': False,
        }
        
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                result['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                result['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                result['fps'] = cap.get(cv2.CAP_PROP_FPS)
                result['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                result['duration'] = result['frames'] / max(1, result['fps'])
                
                # Codec
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                if fourcc:
                    result['codec'] = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + \
                                     chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
                
                cap.release()
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # AUDIO SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_audio(self, path: str) -> Dict:
        """Scan an audio file."""
        if not HAS_AUDIO:
            return {'error': 'Audio libraries not installed'}
        
        result = {
            'path': path,
            'type': 'Audio',
            'size': os.path.getsize(path),
            'duration': 0,
            'sample_rate': 0,
            'channels': 0,
            'samples': 0,
        }
        
        try:
            audio, sr = librosa.load(path, sr=None, duration=30.0)
            result['sample_rate'] = sr
            result['samples'] = len(audio)
            result['duration'] = len(audio) / sr
            result['channels'] = 1 if audio.ndim == 1 else audio.shape[1]
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # DOCUMENT SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_document(self, path: str) -> Dict:
        """Scan a document file."""
        result = {
            'path': path,
            'type': 'Document',
            'size': os.path.getsize(path),
            'pages': 0,
            'text': '',
            'metadata': {},
        }
        
        ext = os.path.splitext(path)[1].lower()
        
        # PDF
        if ext == '.pdf' and HAS_PYPDF2:
            try:
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    result['pages'] = len(reader.pages)
                    for i, page in enumerate(reader.pages[:5]):
                        text = page.extract_text()
                        if text:
                            result['text'] += text[:500]
                    if reader.metadata:
                        result['metadata'] = {
                            k.replace('/', ''): v for k, v in reader.metadata.items()
                        }
            except Exception as e:
                result['error'] = str(e)
        
        # DOCX
        elif ext == '.docx' and HAS_DOCX:
            try:
                doc = docx.Document(path)
                result['pages'] = 1
                for para in doc.paragraphs[:20]:
                    result['text'] += para.text + '\n'
                result['text'] = result['text'][:2000]
                # Metadata
                if hasattr(doc, 'core_properties'):
                    cp = doc.core_properties
                    if cp.author:
                        result['metadata']['author'] = cp.author
                    if cp.title:
                        result['metadata']['title'] = cp.title
            except Exception as e:
                result['error'] = str(e)
        
        # TXT
        elif ext in ('.txt', '.md', '.rst'):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    result['text'] = f.read()[:2000]
            except:
                pass
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # SANDBOX SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_sandbox(self) -> Dict:
        """Scan the sandbox environment."""
        result = {
            'path': CONFIG.SANDBOX_DIR,
            'exists': os.path.exists(CONFIG.SANDBOX_DIR),
            'files': [],
            'processes': [],
            'environment': {},
        }
        
        if os.path.exists(CONFIG.SANDBOX_DIR):
            for root, dirs, files in os.walk(CONFIG.SANDBOX_DIR):
                for f in files[:20]:
                    path = os.path.join(root, f)
                    try:
                        stat = os.stat(path)
                        result['files'].append({
                            'name': f,
                            'size': stat.st_size,
                            'modified': time.ctime(stat.st_mtime),
                            'path': path,
                        })
                    except:
                        pass
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # GENERIC SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_generic(self, target: str, detail: str = None) -> Dict:
        """Scan unknown target."""
        # Try as file path
        if os.path.exists(target):
            return self._scan_filesystem(target)
        
        # Try as URL
        if '.' in target and not target.startswith('/'):
            return self._scan_webpage(target)
        
        # Try as process
        if target.isdigit():
            return self._scan_process(target)
        
        # Unknown
        return {'target': target, 'info': 'Unknown target type'}
    
    # ═══════════════════════════════════════════════════════════════════════
    # EVERYTHING SCANNER
    # ═══════════════════════════════════════════════════════════════════════
    
    def _scan_everything(self) -> Dict:
        """Scan everything in one go."""
        return {
            'sandbox': self._scan_sandbox(),
            'filesystem': self._scan_filesystem('.'),
            'network': self._scan_network(),
            'timestamp': time.time(),
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════════
    
    def _compute_info_gain(self, prev: Any, new: Any) -> float:
        """Calculate information gain between states."""
        if not prev:
            return 1.0
        
        try:
            prev_str = json.dumps(prev, sort_keys=True, default=str)
            new_str = json.dumps(new, sort_keys=True, default=str)
            
            if prev_str == new_str:
                return 0.0
            
            # Simple word overlap
            prev_words = set(prev_str.split())
            new_words = set(new_str.split())
            
            if not new_words:
                return 0.0
            
            return 1.0 - (len(prev_words & new_words) / len(prev_words | new_words))
        except:
            return 0.5
    
    def get_stats(self) -> Dict:
        """Get X-Ray stats."""
        return {
            'total_scans': len(self.vision_history),
            'knowledge_items': len(self.knowledge_state),
            'cache_size': len(self.fs_cache),
        }