# SIEL v15 — The Oscillating Orb Of Chaos

Siel is an **oscillator neural network** designed to learn and understand code through wave interference patterns in phase space.

## Architecture

- **WaveField**: 64×64 grid of coupled phase oscillators
- **Trainer v4.4**: Binary bit prediction with MCTS planning and attention
- **Memory**: Episodic + concept store (204 MB mmap'd .siel format)
- **Perception**: X-Ray Vision for introspecting any system
- **Generation**: Code generation from phase patterns

## Installation

```bash
# Install dependencies
pip install numpy torch psutil

# Clone and set up
git clone https://github.com/graveyardshifttech/Das-Siel
cd siel_rebuild
python siel.py
```

## Running Siel

```bash
# Interactive REPL
python siel.py

# From REPL:
> adapt 60           # Train for 60 epochs
> master            # Train until mastery
> stat              # Show statistics
> xray filesystem   # Perceive filesystem
> exit              # Exit
```

## File Structure

```
siel/
├── siel.py              # Main entry point
├── config.py            # Global configuration
├── core/                # Oscillator & wave field
│   ├── oscillator.py
│   ├── coupling.py
│   ├── wave.py
│   └── wave_simd.py
├── memory/              # Episodic & concept storage
│   ├── concept_store.py
│   └── episodic.py
├── learning/            # Training & modalities
│   ├── trainer.py       # v4.4 with MCTS + attention
│   ├── modalities.py
│   └── dual_stream.py
├── perception/          # X-Ray Vision
│   └── xray.py
├── generation/          # Code generation
│   └── generator.py
├── repl/                # Interactive shell
│   └── commands.py
├── utils/               # Utilities
│   ├── logger.py
│   ├── cpu.py
│   └── trig.py
└── README.md
```

## Training

The trainer uses:

- **8-bit binary output** (easier than 256-way)
- **MCTS planning** with 25 simulations per step
- **Attention mechanism** over oscillator phases
- **Curriculum learning** (coupling anneals 2.0 → 0.3)
- **Multi-domain chimera initialization**

Expected performance:
- Epoch 1-10: Loss decreases (3.0 → 1.5)
- Epoch 10-40: Convergence (1.5 → 0.2)
- Epoch 40+: Plateau (< 0.1 = mastery)

## Key Features

### WaveField
- 4096 oscillators in 64×64 grid
- Sparse local coupling (35% edge density)
- Phase-based computation

### Trainer
- Binary cross-entropy (8 bits, not 256 classes)
- MCTS lookahead planning
- Integrated attention layer
- Gradient clipping & numerical stability
- DataLoader with proper batching

### Memory
- StreamingConceptStore: 50k concepts, 100-item LRU cache
- .siel binary format (mmap'd for speed)
- Episodic memory with importance weighting
- Replay buffer for catastrophic forgetting

### Perception
- X-Ray Vision can inspect:
  - Filesystems (with type distribution)
  - Binary executables (PE/ELF/Mach-O)
  - Webpages (structure extraction)
  - GGUF models (architecture introspection)
  - Running processes
  - Network connections
  - Media files (image, video, audio)

## Performance

On ThinkPad T480 (i7-8550U, 16GB RAM):
- **Speed**: ~15 seconds per epoch (64×64 grid)
- **Memory**: ~1GB total
- **Stability**: 0 NaN/Inf violations

## Debugging

Check `siel_logs/` for detailed training logs.

```python
# Inspect training state
trainer.get_stats()

# Check stability
trainer.stability.get_stats()

# View loss curves
import matplotlib.pyplot as plt
plt.plot(trainer.loss_history)
plt.show()
```

## Next Steps

- [ ] Curriculum learning for longer sequences
- [ ] Multi-task learning (code + language)
- [ ] Hierarchical planning (MCTS with learned value)
- [ ] Integration with HumanEval / MBPP benchmarks

## License

MIT

---

