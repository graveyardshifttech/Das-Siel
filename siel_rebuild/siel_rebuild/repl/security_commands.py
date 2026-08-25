"""
Security commands for the REPL.
"""

import json
from typing import List, Dict

from ..utils.sandbox import sandbox
from ..utils.logger import logger


class SecurityCommands:
    """Security-related REPL commands."""
    
    @staticmethod
    def cmd_analyze(args: List[str], siel):
        """Analyze code for vulnerabilities without executing."""
        if not args:
            print("Usage: analyze <code>")
            return
        
        code = ' '.join(args)
        result = sandbox.analyze(code)
        
        print(f"🔍 Analysis: {result['code_hash']}")
        print(f"   Observations: {result['observation_count']}")
        print(f"   Security Score: {result['security_score']:.2f}")
        
        if result['high_risk']:
            print("\n⚠️ High Risk Issues:")
            for issue in result['high_risk'][:10]:
                print(f"  [{issue['severity'].upper()}] {issue['description']}")
    
    @staticmethod
    def cmd_execute(args: List[str], siel):
        """Execute code with full observation."""
        if not args:
            print("Usage: execute <code>")
            return
        
        code = ' '.join(args)
        record = sandbox.execute(code, 'python')
        
        print(f"🔍 Execution: {record.code_hash}")
        print(f"   Success: {record.success}")
        print(f"   Duration: {record.duration:.2f}s")
        print(f"   Security Score: {record.security_score:.2f}")
        print(f"   Observations: {len(record.observations)}")
        
        if record.observations:
            print("\n📋 Observations:")
            for obs in record.observations[:10]:
                print(f"  [{obs.severity.upper()}] {obs.description}")
            if len(record.observations) > 10:
                print(f"  ... and {len(record.observations) - 10} more")
        
        if record.stdout:
            print(f"\n📤 stdout:\n{record.stdout[:500]}")
        if record.stderr:
            print(f"\n📤 stderr:\n{record.stderr[:500]}")
    
    @staticmethod
    def cmd_escape(args: List[str], siel):
        """Test escape attempts."""
        if not args:
            print("Usage: escape <code>")
            return
        
        code = ' '.join(args)
        record = sandbox.execute(code, 'python')
        
        escape_obs = [o for o in record.observations if o.event_type == 'escape_attempts']
        
        print(f"🔍 Escape Test: {record.code_hash}")
        print(f"   Escape Attempts: {len(escape_obs)}")
        
        if escape_obs:
            for obs in escape_obs:
                print(f"  [{obs.severity.upper()}] {obs.description} (line {obs.line})")
                if obs.context:
                    print(f"    {obs.context}")
    
    @staticmethod
    def cmd_race(args: List[str], siel):
        """Test race conditions."""
        if not args:
            print("Usage: race <code>")
            return
        
        code = ' '.join(args)
        record = sandbox.execute(code, 'python')
        
        race_obs = [o for o in record.observations if o.event_type == 'race_conditions']
        
        print(f"🔍 Race Test: {record.code_hash}")
        print(f"   Race Conditions: {len(race_obs)}")
        
        if race_obs:
            for obs in race_obs:
                print(f"  [{obs.severity.upper()}] {obs.description} (line {obs.line})")
    
    @staticmethod
    def cmd_uaf(args: List[str], siel):
        """Test UAF attempts."""
        if not args:
            print("Usage: uaf <code>")
            return
        
        code = ' '.join(args)
        record = sandbox.execute(code, 'python')
        
        uaf_obs = [o for o in record.observations if o.event_type == 'uaf_attempts']
        
        print(f"🔍 UAF Test: {record.code_hash}")
        print(f"   UAF Attempts: {len(uaf_obs)}")
        
        if uaf_obs:
            for obs in uaf_obs:
                print(f"  [{obs.severity.upper()}] {obs.description} (line {obs.line})")
    
    @staticmethod
    def cmd_sandbox_stats(args: List[str], siel):
        """Show sandbox statistics."""
        stats = sandbox.get_stats()
        report = sandbox.get_vulnerability_report()
        
        print("🔍 Sandbox Statistics:")
        print(f"   Total Executions: {stats['total_executions']}")
        print(f"   Suspicious: {stats['suspicious_executions']}")
        print(f"   Observations: {stats['observation_count']}")
        print()
        
        print("📊 Vulnerability Summary:")
        print(f"   Escape Attempts: {stats['escape_attempts']}")
        print(f"   Race Conditions: {stats['race_conditions']}")
        print(f"   UAF Attempts: {stats['uaf_attempts']}")
        print(f"   Memory Violations: {stats['memory_violations']}")
        print(f"   Privilege Escalation: {stats['privilege_escalation']}")
        print()
        
        print("📈 Security Scores:")
        print(f"   Average: {stats['avg_security_score']:.2f}")
        print(f"   Maximum: {stats['max_security_score']:.2f}")
        print()
        
        print("🔝 Top Patterns:")
        for pattern in report['top_patterns'][:5]:
            print(f"  {pattern['pattern']} ({pattern['severity']}): {pattern['count']}x")
    
    @staticmethod
    def cmd_verbose(args: List[str], siel):
        """Toggle verbose mode."""
        siel.verbose = not siel.verbose
        print(f"Verbose mode: {'ON' if siel.verbose else 'OFF'}")