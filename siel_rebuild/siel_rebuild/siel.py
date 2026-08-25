#!/usr/bin/env python3
"""
siel.py — Main entry point for Siel v15.
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from config import CONFIG
from utils.logger import logger
from utils.cpu import cpu_saver
from core.wave import WaveField
from memory.concept_store import StreamingConceptStore
from memory.episodic import EpisodicMemory
from learning.trainer import Trainer
from perception.xray import XRayVision2
from generation.generator import Generator
from repl.commands import CommandHandler


class Siel:
    """Main Siel class — glues everything together."""
    
    def __init__(self, state_file: str = "siel_state.json"):
        self.start_time = time.time()
        self.state_file = state_file
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║  SIEL V15 — THE CODING GODDESS                              ║
║  Modular, Vectorized, O(N) Mean-Field Propagation          ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # ─── Initialize Core ──────────────────────────────────────────────
        print("🔧 Initializing core systems...")
        self.wave = WaveField()
        print(f"   ✅ WaveField: {CONFIG.GRID_SIZE}×{CONFIG.GRID_SIZE}")
        
        # ─── Initialize Memory ────────────────────────────────────────────
        print("💾 Initializing memory...")
        self.store = StreamingConceptStore()
        self.memory = EpisodicMemory()
        print(f"   ✅ Concept Store: {self.store.max_concepts} max, {self.store.concept_count} used")
        
        # ─── Initialize Learning ──────────────────────────────────────────
        print("🧠 Initializing learning system...")
        self.trainer = Trainer(self.wave, self.store, self.memory)
        print(f"   ✅ Trainer: {len(self.trainer.dataset)} train, {len(self.trainer.val_dataset)} val")
        
        # ─── Initialize Perception ────────────────────────────────────────
        print("🔍 Initializing X-Ray Vision...")
        self.xray = XRayVision2()
        print("   ✅ X-Ray Vision ready")
        
        # ─── Initialize Generation ────────────────────────────────────────
        print("⚡ Initializing generator...")
        self.generator = Generator(self.wave, self.memory)
        print("   ✅ Generator ready")
        
        # ─── Load State ──────────────────────────────────────────────────
        self._load_state()
        
        # ─── Auto-Learner ─────────────────────────────────────────────────
        self.auto_learner_running = False
        self.auto_learner_thread = None
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅ SIEL V15 INITIALIZATION COMPLETE                         ║
║  Type 'help' for available commands                         ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def load_dataset(self, source: str):
        """Load dataset from source."""
        logger.log(f"Loading dataset: {source}")
        
        if source.startswith('hf://'):
            # Hugging Face dataset
            dataset_name = source[5:]
            # This would connect to HF API
            print(f"📥 Loading from Hugging Face: {dataset_name}")
            # For now, just add to trainer
            self.trainer.load_dataset(source)
        
        elif source.startswith('github://'):
            # GitHub repository
            repo_url = source[9:]
            print(f"📥 Loading from GitHub: {repo_url}")
            self.trainer.load_dataset(source)
        
        elif source.startswith('preset://'):
            preset = source[9:]
            print(f"📥 Loading preset: {preset}")
            if preset in {'small_code', 'medium_code', 'algorithms'}:
                self.trainer.load_dataset(source)
            else:
                print(f"❌ Unknown preset: {preset}")
        
        else:
            print(f"❌ Unknown source: {source}")
            print("   Supported: hf://, github://, preset://")
    
    def _load_state(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                import json
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                # Restore memory
                if 'memory' in state:
                    # Would restore memory state
                    pass
                
                # Restore store
                if 'store' in state:
                    # Would restore store state
                    pass
                
                logger.log(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.log(f"Failed to load state: {e}", "WARN")
    
    def _save_state(self):
        """Save state to file."""
        try:
            import json
            state = {
                'timestamp': time.time(),
                'memory': self.memory.get_summary(),
                'store': self.store.get_stats(),
                'trainer': self.trainer.get_stats(),
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.log(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.log(f"Failed to save state: {e}", "WARN")
    
    def start_auto_learner(self):
        """Start the auto-learner thread."""
        if self.auto_learner_thread and self.auto_learner_thread.is_alive():
            print("🤖 Auto-learner already running")
            return
        
        self.auto_learner_running = True
        self.auto_learner_thread = threading.Thread(target=self._auto_learner_loop, daemon=True)
        self.auto_learner_thread.start()
        print("🤖 Auto-learner started")
    
    def stop_auto_learner(self):
        """Stop the auto-learner thread."""
        self.auto_learner_running = False
        if self.auto_learner_thread:
            self.auto_learner_thread.join(timeout=2)
        print("🤖 Auto-learner stopped")
    
    def _auto_learner_loop(self):
        """Auto-learner background loop."""
        import time
        watch_dir = Path(CONFIG.WATCH_DIR)
        watch_dir.mkdir(parents=True, exist_ok=True)
        
        processed = set()
        
        while self.auto_learner_running:
            try:
                # Scan for new files
                new_files = []
                for f in watch_dir.rglob('*'):
                    if f.is_file() and f.suffix in ('.py', '.txt', '.js', '.go', '.rs', '.cpp', '.java', '.rb'):
                        if str(f) not in processed:
                            processed.add(str(f))
                            new_files.append(f)
                
                if new_files:
                    print(f"📄 Found {len(new_files)} new files")
                    for f in new_files[:5]:
                        print(f"   - {f.name}")
                    
                    # Learn from new files
                    self.trainer.load_dataset()
                    result = self.trainer.train(epochs=5, master=False)
                    print(f"✅ Learned from {len(new_files)} files: loss={result['final_loss']:.4f}")
                
                time.sleep(CONFIG.WATCH_POLL_INTERVAL)
                
            except Exception as e:
                logger.log(f"Auto-learner error: {e}", "ERROR")
                time.sleep(10)
    
    def run(self):
        """Run the REPL."""
        handler = CommandHandler(self)
        
        print("\n💬 Welcome to Siel v15!")
        print("   Type 'help' for commands, 'exit' to quit.\n")
        
        while handler.running:
            try:
                cmd = input("> ").strip()
                handler.execute(cmd)
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
        
        # Cleanup
        self._save_state()
        self.stop_auto_learner()
        self.store.close()
        logger.flush()
        print("💾 State saved. Goodbye!")


def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Siel v15 — The Coding Goddess')
    parser.add_argument('--watch', '-w', action='store_true', help='Start auto-learner')
    parser.add_argument('--train', '-t', action='store_true', help='Train immediately')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Epochs to train')
    parser.add_argument('--state', '-s', default='siel_state.json', help='State file')
    args = parser.parse_args()
    
    # Initialize Siel
    siel = Siel(state_file=args.state)
    
    # Start auto-learner if requested
    if args.watch:
        siel.start_auto_learner()
    
    # Train if requested
    if args.train:
        siel.trainer.train(epochs=args.epochs)
    
    # Run REPL
    siel.run()


if __name__ == "__main__":
    main()