"""
REPL Commands for Siel interactive mode.
"""

from typing import Callable, Dict, Any, Optional, List
from utils.logger import logger
from .security_commands import SecurityCommands


class CommandHandler:
    """Handle REPL commands."""
    
    def __init__(self, siel):
        self.siel = siel
        self.running = True
        self.commands: Dict[str, Callable] = {
            'help': self.cmd_help,
            'adapt': self.cmd_adapt,
            'master': self.cmd_master,
            'stat': self.cmd_stat,
            'status': self.cmd_stat,
            'xray': self.cmd_xray,
            'recall': self.cmd_recall,
            # Security commands
            'analyze': self.cmd_analyze,
            'execute': self.cmd_execute,
            'escape': self.cmd_escape,
            'race': self.cmd_race,
            'uaf': self.cmd_uaf,
            'sandbox_stats': self.cmd_sandbox_stats,
            # Exit
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
        }
    
    def execute(self, cmd: str):
        """Execute a command."""
        parts = cmd.split()
        if not parts:
            return
        
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        if cmd_name in self.commands:
            try:
                self.commands[cmd_name](args)
            except Exception as e:
                logger.error(f"Command failed: {e}")
        else:
            print(f"❓ Unknown command: {cmd_name}")
            print("   Type 'help' for available commands")
    
    def cmd_help(self, args):
        """Show help."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  SIEL COMMANDS                                               ║
╠══════════════════════════════════════════════════════════════╣
║  TRAINING                                                    ║
║  adapt <epochs>    — Train for N epochs                      ║
║  master            — Train until mastery (loss < 0.1)        ║
║                                                              ║
║  PERCEPTION                                                  ║
║  xray <target>     — Perceive and analyze target             ║
║  recall <type>     — Recall memories of type                 ║
║  stat              — Show current statistics                 ║
║                                                              ║
║  SECURITY RESEARCH                                           ║
║  analyze <code>    — Analyze code for vulnerabilities        ║
║  execute <code>    — Execute code in sandbox                 ║
║  escape            — Test sandbox escape vectors             ║
║  race              — Test for race conditions                ║
║  uaf               — Test for use-after-free                 ║
║  sandbox_stats     — Show sandbox statistics                 ║
║                                                              ║
║  SYSTEM                                                      ║
║  exit/quit         — Exit Siel                               ║
║  help              — Show this help                          ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def cmd_adapt(self, args):
        """Train for N epochs."""
        epochs = int(args[0]) if args else 10
        print(f"🧠 Training for {epochs} epochs...")
        result = self.siel.trainer.train(epochs=epochs)
        print(f"✅ Training complete: {result}")
    
    def cmd_master(self, args):
        """Train until mastery."""
        print("🎯 Training until mastery (loss < 0.001)...")
        result = self.siel.trainer.train(epochs=200)
        print(f"✅ Mastery training complete: {result}")
    
    def cmd_stat(self, args):
        """Show statistics."""
        stats = self.siel.trainer.get_stats()
        print("\n📊 STATISTICS")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def cmd_xray(self, args):
        """Use X-Ray Vision."""
        if not args:
            print("❌ Usage: xray <target>")
            return
        
        target = args[0]
        print(f"👁️ X-Ray scanning: {target}...")
        result = self.siel.xray.perceive(target)
        print(f"✅ Perception result: {result}")
    
    def cmd_recall(self, args):
        """Recall memories."""
        mem_type = args[0] if args else None
        print(f"🧠 Recalling memories (type={mem_type})...")
        memories = self.siel.memory.recall(episode_type=mem_type, limit=5)
        for mem in memories:
            print(f"  - {mem.type}: {mem.content}")
    
    def cmd_analyze(self, args: List[str]):
        """Analyze code for vulnerabilities."""
        SecurityCommands.cmd_analyze(args, self.siel)
    
    def cmd_execute(self, args: List[str]):
        """Execute code in sandbox."""
        SecurityCommands.cmd_execute(args, self.siel)
    
    def cmd_escape(self, args: List[str]):
        """Test sandbox escape vectors."""
        SecurityCommands.cmd_escape(args, self.siel)
    
    def cmd_race(self, args: List[str]):
        """Test for race conditions."""
        SecurityCommands.cmd_race(args, self.siel)
    
    def cmd_uaf(self, args: List[str]):
        """Test for use-after-free."""
        SecurityCommands.cmd_uaf(args, self.siel)
    
    def cmd_sandbox_stats(self, args: List[str]):
        """Show sandbox statistics."""
        SecurityCommands.cmd_sandbox_stats(args, self.siel)
    
    def cmd_exit(self, args):
        """Exit."""
        print("👋 Goodbye!")
        self.running = False
