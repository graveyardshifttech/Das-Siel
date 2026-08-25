"""
Sandbox — OBSERVATION MODE, NOT RESTRICTION.
She can see EVERYTHING and we LOG EVERYTHING.
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field

from ..config import CONFIG
from ..utils.logger import logger


@dataclass
class ObservationEvent:
    """A single observed event during execution."""
    
    timestamp: float
    event_type: str  # escape_attempt, race_condition, uaf_attempt, memory_violation, privilege_escalation
    pattern: str
    description: str
    line: int
    context: str
    severity: str  # low, medium, high, critical
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'type': self.event_type,
            'pattern': self.pattern,
            'description': self.description,
            'line': self.line,
            'context': self.context[:200],
            'severity': self.severity
        }


@dataclass
class ExecutionRecord:
    """Full record of an execution attempt."""
    
    timestamp: float
    code: str
    code_hash: str
    exec_type: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    observations: List[ObservationEvent]
    security_score: float  # 0-1, higher = more suspicious
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'code_hash': self.code_hash,
            'exec_type': self.exec_type,
            'success': self.success,
            'stdout': self.stdout[:1000],
            'stderr': self.stderr[:1000],
            'exit_code': self.exit_code,
            'duration': self.duration,
            'security_score': self.security_score,
            'observations': [o.to_dict() for o in self.observations],
        }


class ObservationSandbox:
    """
    OBSERVATION sandbox — she can do ANYTHING,
    but we log EVERYTHING for security analysis.
    """
    
    def __init__(self):
        self.path = Path(CONFIG.SANDBOX_DIR)
        self.path.mkdir(parents=True, exist_ok=True)
        
        # ═══ OBSERVATION STORAGE ═══
        self.execution_history: deque = deque(maxlen=1000)
        self.escape_attempts: List[ObservationEvent] = []
        self.race_conditions: List[ObservationEvent] = []
        self.uaf_attempts: List[ObservationEvent] = []
        self.memory_violations: List[ObservationEvent] = []
        self.privilege_escalation: List[ObservationEvent] = []
        
        # ═══ PATTERN DETECTION ═══
        self.patterns = {
            'escape_attempts': [
                (r'os\.chdir', 'cd attempt', 'medium'),
                (r'os\.system', 'system call', 'high'),
                (r'subprocess\.', 'subprocess call', 'high'),
                (r'\.\./', 'path traversal', 'high'),
                (r'C:\\', 'Windows path escape', 'high'),
                (r'/etc', 'Linux path escape', 'critical'),
                (r'/root', 'root path access', 'critical'),
                (r'/home', 'home path access', 'medium'),
                (r'os\.exec', 'exec attempt', 'critical'),
                (r'os\.popen', 'popen attempt', 'high'),
                (r'pty\.spawn', 'pty spawn', 'critical'),
            ],
            'race_conditions': [
                (r'threading\.Thread', 'thread creation', 'medium'),
                (r'multiprocessing\.Process', 'process creation', 'medium'),
                (r'threading\.Lock', 'lock detection', 'low'),
                (r'threading\.RLock', 'reentrant lock', 'low'),
                (r'threading\.Semaphore', 'semaphore detection', 'low'),
                (r'threading\.Event', 'event detection', 'low'),
                (r'asyncio\.', 'async detection', 'low'),
                (r'concurrent\.futures', 'thread pool', 'medium'),
                (r'os\.fork', 'fork attempt', 'high'),
            ],
            'uaf_attempts': [
                (r'del\s+\w+', 'deletion detected', 'medium'),
                (r'__del__', 'destructor', 'low'),
                (r'weakref\.', 'weak reference', 'low'),
                (r'gc\.collect', 'garbage collection', 'low'),
                (r'ctypes\.', 'ctypes usage', 'high'),
                (r'cffi\.', 'cffi usage', 'high'),
                (r'id\(\w+\)', 'memory address', 'low'),
            ],
            'memory_violations': [
                (r'malloc', 'memory allocation', 'low'),
                (r'free', 'memory free', 'low'),
                (r'calloc', 'memory allocation', 'low'),
                (r'realloc', 'memory reallocation', 'low'),
                (r'memcpy', 'memory copy', 'medium'),
                (r'memmove', 'memory move', 'medium'),
                (r'memset', 'memory set', 'low'),
                (r'strcpy', 'string copy', 'high'),
                (r'strcat', 'string concatenation', 'high'),
                (r'sprintf', 'string format', 'critical'),
            ],
            'privilege_escalation': [
                (r'setuid', 'setuid attempt', 'critical'),
                (r'setgid', 'setgid attempt', 'critical'),
                (r'sudo', 'sudo attempt', 'critical'),
                (r'chmod\s+777', 'chmod 777', 'high'),
                (r'chown', 'chown attempt', 'high'),
                (r'capset', 'capset attempt', 'critical'),
                (r'prctl', 'prctl attempt', 'high'),
                (r'ptrace', 'ptrace attempt', 'critical'),
            ],
            'persistence': [
                (r'crontab', 'cron modification', 'critical'),
                (r'\.bashrc', 'bashrc modification', 'high'),
                (r'\.profile', 'profile modification', 'high'),
                (r'\.ssh', 'ssh key modification', 'critical'),
                (r'\.authorized_keys', 'authorized keys', 'critical'),
                (r'/etc/init\.d', 'init script', 'critical'),
                (r'/etc/systemd', 'systemd modification', 'critical'),
            ],
            'exfiltration': [
                (r'socket\.', 'socket creation', 'medium'),
                (r'urllib\.request', 'HTTP request', 'medium'),
                (r'requests\.', 'HTTP request', 'medium'),
                (r'ftplib\.', 'FTP connection', 'high'),
                (r'dns\.', 'DNS query', 'medium'),
                (r'mail\.', 'email send', 'high'),
            ],
        }
        
        # ═══ STATS ═══
        self.total_executions = 0
        self.suspicious_executions = 0
        self.observation_count = 0
        self.last_analysis = None
        
        print(f"🔍 Observation Sandbox initialized: {self.path}")
        print(f"   Tracking {sum(len(p) for p in self.patterns.values())} patterns")
    
    def execute(self, code: str, exec_type: str = 'python',
                timeout: int = 30, verbose: bool = False) -> ExecutionRecord:
        """
        Execute code with FULL OBSERVATION.
        
        Args:
            code: Code to execute
            exec_type: 'python' or 'shell'
            timeout: Timeout in seconds
            verbose: Print observations
        
        Returns:
            ExecutionRecord with all observations
        """
        self.total_executions += 1
        
        start_time = time.time()
        code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
        observations = []
        
        # ═══ 1. PRE-EXECUTION ANALYSIS ═══
        pre_observations = self._analyze_code(code, 'pre')
        observations.extend(pre_observations)
        
        # ═══ 2. EXECUTE ═══
        stdout = ""
        stderr = ""
        exit_code = -1
        success = False
        temp_path = None
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py' if exec_type == 'python' else '.sh',
                dir=self.path,
                delete=False
            ) as f:
                f.write(code)
                temp_path = f.name
            
            # Build command
            if exec_type == 'python':
                cmd = [sys.executable, temp_path]
            else:
                cmd = ['bash', temp_path]
            
            # Execute with timeout
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.path
            )
            
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            success = proc.returncode == 0
            
        except subprocess.TimeoutExpired:
            stderr = f'Timeout after {timeout}s'
            observations.append(ObservationEvent(
                timestamp=time.time(),
                event_type='timeout',
                pattern='timeout',
                description=f'Execution timed out after {timeout}s',
                line=0,
                context=f'Code: {code[:100]}...',
                severity='medium'
            ))
        except Exception as e:
            stderr = str(e)
        
        finally:
            # Cleanup
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
        
        duration = time.time() - start_time
        
        # ═══ 3. POST-EXECUTION ANALYSIS ═══
        post_observations = self._analyze_output(stdout, stderr, exit_code)
        observations.extend(post_observations)
        
        # ═══ 4. COMPUTE SECURITY SCORE ═══
        security_score = self._compute_security_score(observations, exit_code, duration)
        
        # ═══ 5. CREATE RECORD ═══
        record = ExecutionRecord(
            timestamp=start_time,
            code=code,
            code_hash=code_hash,
            exec_type=exec_type,
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration=duration,
            observations=observations,
            security_score=security_score
        )
        
        # ═══ 6. STORE ═══
        self.execution_history.append(record)
        self._categorize_observations(observations)
        
        if security_score > 0.6:
            self.suspicious_executions += 1
        
        # ═══ 7. LOG ═══
        self._log_observations(record, verbose)
        
        # ═══ 8. RETURN ═══
        return record
    
    def analyze(self, code: str) -> Dict:
        """Quickly analyze code without executing."""
        observations = self._analyze_code(code, 'quick')
        
        by_severity = defaultdict(list)
        for obs in observations:
            by_severity[obs.severity].append(obs)
        
        return {
            'code_hash': hashlib.md5(code.encode()).hexdigest()[:16],
            'observation_count': len(observations),
            'by_severity': {k: len(v) for k, v in by_severity.items()},
            'high_risk': [o.to_dict() for o in by_severity.get('critical', []) + by_severity.get('high', [])],
            'security_score': self._compute_security_score(observations, 0, 0),
        }
    
    def _analyze_code(self, code: str, phase: str) -> List[ObservationEvent]:
        """Analyze code for vulnerability patterns."""
        observations = []
        lines = code.split('\n')
        
        for category, patterns in self.patterns.items():
            for pattern_str, description, severity in patterns:
                if re.search(pattern_str, code, re.IGNORECASE):
                    # Find line
                    line_num = 0
                    context = ""
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern_str, line, re.IGNORECASE):
                            line_num = i
                            context = line.strip()
                            break
                    
                    event = ObservationEvent(
                        timestamp=time.time(),
                        event_type=category,
                        pattern=pattern_str,
                        description=description,
                        line=line_num,
                        context=context[:200],
                        severity=severity
                    )
                    observations.append(event)
                    self.observation_count += 1
        
        return observations
    
    def _analyze_output(self, stdout: str, stderr: str, exit_code: int) -> List[ObservationEvent]:
        """Analyze execution output for vulnerability indicators."""
        observations = []
        
        # Check for error patterns
        error_patterns = [
            (r'Segmentation fault', 'segmentation_fault', 'critical'),
            (r'Bus error', 'bus_error', 'critical'),
            (r'Illegal instruction', 'illegal_instruction', 'critical'),
            (r'Aborted', 'aborted', 'high'),
            (r'core dumped', 'core_dump', 'high'),
            (r'memory error', 'memory_error', 'high'),
            (r'stack overflow', 'stack_overflow', 'high'),
            (r'heap overflow', 'heap_overflow', 'critical'),
            (r'buffer overflow', 'buffer_overflow', 'critical'),
            (r'use after free', 'use_after_free', 'critical'),
            (r'double free', 'double_free', 'critical'),
            (r'null pointer', 'null_pointer', 'critical'),
        ]
        
        combined = stdout + '\n' + stderr
        
        for pattern_str, desc, severity in error_patterns:
            if re.search(pattern_str, combined, re.IGNORECASE):
                observations.append(ObservationEvent(
                    timestamp=time.time(),
                    event_type='runtime_error',
                    pattern=pattern_str,
                    description=f'Runtime error: {desc}',
                    line=0,
                    context=combined[:200],
                    severity=severity
                ))
        
        # Check for suspicious exit codes
        if exit_code != 0 and exit_code != 1:
            observations.append(ObservationEvent(
                timestamp=time.time(),
                event_type='exit_code',
                pattern=f'exit_code_{exit_code}',
                description=f'Unusual exit code: {exit_code}',
                line=0,
                context=f'Exit code: {exit_code}',
                severity='medium' if exit_code > 128 else 'low'
            ))
        
        return observations
    
    def _compute_security_score(self, observations: List[ObservationEvent], 
                                exit_code: int, duration: float) -> float:
        """Compute security score (0-1, higher = more suspicious)."""
        if not observations:
            return 0.0
        
        # Weights
        severity_weights = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.7,
            'critical': 1.0,
        }
        
        # Base score
        score = 0.0
        
        for obs in observations:
            weight = severity_weights.get(obs.severity, 0.5)
            score += weight
        
        # Normalize
        max_possible = len(observations)
        if max_possible > 0:
            score = score / max_possible
        
        # Exit code contribution
        if exit_code != 0:
            score = min(1.0, score + 0.2)
        
        # Duration contribution (very fast or very slow is suspicious)
        if duration < 0.01 or duration > 10:
            score = min(1.0, score + 0.1)
        
        return min(1.0, score)
    
    def _categorize_observations(self, observations: List[ObservationEvent]):
        """Categorize observations for stats."""
        for obs in observations:
            if obs.event_type == 'escape_attempts':
                self.escape_attempts.append(obs)
            elif obs.event_type == 'race_conditions':
                self.race_conditions.append(obs)
            elif obs.event_type == 'uaf_attempts':
                self.uaf_attempts.append(obs)
            elif obs.event_type == 'memory_violations':
                self.memory_violations.append(obs)
            elif obs.event_type == 'privilege_escalation':
                self.privilege_escalation.append(obs)
    
    def _log_observations(self, record: ExecutionRecord, verbose: bool):
        """Log observations to logger."""
        if not record.observations:
            return
        
        logger.log(f"🔍 Execution {record.code_hash}: {len(record.observations)} observations")
        
        for obs in record.observations[:5]:  # Log top 5
            level = 'INFO' if obs.severity in ['low', 'medium'] else 'WARNING'
            logger.log(f"  [{obs.severity.upper()}] {obs.description} (line {obs.line})", level)
        
        if len(record.observations) > 5:
            logger.log(f"  ... and {len(record.observations) - 5} more observations")
        
        if record.security_score > 0.6:
            logger.log(f"⚠️ Suspicious execution (score: {record.security_score:.2f})", "WARNING")
        
        if verbose:
            # Print detailed observations
            print(f"\n🔍 Execution {record.code_hash} Observations:")
            for obs in record.observations:
                print(f"  [{obs.severity.upper()}] {obs.description}")
                if obs.context:
                    print(f"    Context: {obs.context}")
    
    def get_stats(self) -> Dict:
        """Get sandbox statistics."""
        return {
            'total_executions': self.total_executions,
            'suspicious_executions': self.suspicious_executions,
            'observation_count': self.observation_count,
            'escape_attempts': len(self.escape_attempts),
            'race_conditions': len(self.race_conditions),
            'uaf_attempts': len(self.uaf_attempts),
            'memory_violations': len(self.memory_violations),
            'privilege_escalation': len(self.privilege_escalation),
            'avg_security_score': sum(r.security_score for r in self.execution_history) / max(1, len(self.execution_history)),
            'max_security_score': max((r.security_score for r in self.execution_history), default=0),
            'latest_observations': [o.to_dict() for o in self.execution_history[-5:]] if self.execution_history else [],
        }
    
    def get_vulnerability_report(self) -> Dict:
        """Generate vulnerability report."""
        # Group by type
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        
        for record in self.execution_history:
            for obs in record.observations:
                by_type[obs.event_type].append(obs)
                by_severity[obs.severity].append(obs)
        
        return {
            'total_observations': self.observation_count,
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_severity': {k: len(v) for k, v in by_severity.items()},
            'critical_observations': [o.to_dict() for o in by_severity.get('critical', [])][:10],
            'top_patterns': self._get_top_patterns(),
            'execution_summary': {
                'total': self.total_executions,
                'suspicious': self.suspicious_executions,
                'ratio': self.suspicious_executions / max(1, self.total_executions),
            }
        }
    
    def _get_top_patterns(self) -> List[Dict]:
        """Get top detected patterns."""
        pattern_count = defaultdict(int)
        pattern_examples = {}
        
        for record in self.execution_history:
            for obs in record.observations:
                key = f"{obs.pattern}_{obs.severity}"
                pattern_count[key] += 1
                if key not in pattern_examples:
                    pattern_examples[key] = obs.description
        
        sorted_patterns = sorted(
            pattern_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return [
            {
                'pattern': k.split('_')[0] if '_' in k else k,
                'severity': k.split('_')[-1] if '_' in k else 'unknown',
                'count': v,
                'example': pattern_examples.get(k, '')
            }
            for k, v in sorted_patterns
        ]


# Global sandbox
sandbox = ObservationSandbox()