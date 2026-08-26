"""
Attention Mechanisms for Oscillator Neural Networks
- PhaseCoherentAttention: Phase-aware attention (fastest)
- FrequencyAttention: Frequency-space attention (best for oscillators)
- GraphOscillatorAttention: Graph-based attention (most powerful)
- MultiHeadAttentionArsenal: Combines all three
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Union


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1: PHASE-COHERENT ATTENTION (Fastest)
# ═══════════════════════════════════════════════════════════════════════════

class PhaseCoherentAttention(nn.Module):
    """
    Attention that respects oscillator phase relationships.
    Uses complex-valued attention weights for phase-aware processing.
    """
    
    def __init__(self, n_osc: int, n_heads: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        
        # Phase-aware projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable phase bias (key innovation!)
        self.phase_bias = nn.Parameter(torch.randn(n_osc, n_osc) * 0.01)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.last_attn_weights = None
        
    def forward(self, phases: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phases: (B, N) oscillator phases
            hidden: (B, N, D) hidden states
            
        Returns:
            out: (B, N, D) attended hidden states
            attn_weights: (B, heads, N, N) attention weights
        """
        B, N, D = hidden.shape
        
        # Standard QKV
        Q = self.q_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 🔥 KEY: Add phase-aware bias using oscillator relationships
        # Phase difference matrix: (B, N, N)
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
        phase_similarity = torch.cos(phase_diff)  # 1 when same phase, -1 when opposite
        
        # Apply learned phase bias
        phase_bias = self.phase_bias.unsqueeze(0) * phase_similarity  # (B, N, N)
        
        # Add phase bias to attention before softmax
        attn = attn + phase_bias.unsqueeze(1)  # (B, heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Store for visualization
        self.last_attn_weights = attn.detach()
        
        # Output
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out, attn


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2: FREQUENCY-SPACE ATTENTION (Best for Oscillators)
# ═══════════════════════════════════════════════════════════════════════════

class FrequencyAttention(nn.Module):
    """
    Attention in frequency space - oscillators naturally resonate at frequencies.
    This allows the network to focus on specific frequency bands.
    """
    
    def __init__(self, n_osc: int, n_freq_bands: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_freq_bands = n_freq_bands
        self.dim = dim
        
        # Learnable frequency filters
        self.freq_filters = nn.Parameter(torch.randn(n_freq_bands, n_osc) * 0.01)
        
        # Attention per frequency band
        self.band_attention = nn.Sequential(
            nn.Linear(n_osc, n_osc),
            nn.ReLU(),
            nn.Linear(n_osc, n_osc),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Reconstruct from frequency bands
        self.reconstruct = nn.Sequential(
            nn.Linear(n_osc * n_freq_bands, n_osc),
            nn.ReLU(),
            nn.Linear(n_osc, n_osc),
        )
        
        # Store band weights for visualization
        self.last_band_weights = None
        
    def forward(self, phases: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phases: (B, N) oscillator phases
            hidden: (B, N, D) hidden states (not used directly, but kept for interface)
            
        Returns:
            out: (B, N, D) attended phases
            attn_weights: (B, N, n_freq_bands) attention weights per band
        """
        B, N = phases.shape
        
        # ───────────────────────────────────────────────────────────────────
        # Step 1: Transform to frequency space
        # ───────────────────────────────────────────────────────────────────
        freq_repr = []
        for i in range(self.n_freq_bands):
            # Apply frequency filter
            filtered = torch.sin(phases + self.freq_filters[i].unsqueeze(0))
            freq_repr.append(filtered)
        
        freq_stack = torch.stack(freq_repr, dim=1)  # (B, n_freq_bands, N)
        
        # ───────────────────────────────────────────────────────────────────
        # Step 2: Apply attention per frequency band
        # ───────────────────────────────────────────────────────────────────
        attended = []
        band_weights = []
        for i in range(self.n_freq_bands):
            # Attention weights for this frequency band
            attn_weights = self.band_attention(freq_stack[:, i, :])  # (B, N)
            band_weights.append(attn_weights)
            # Apply attention
            attended_band = freq_stack[:, i, :] * attn_weights
            attended.append(attended_band)
        
        attended_stack = torch.stack(attended, dim=1)  # (B, n_freq_bands, N)
        band_weights_stack = torch.stack(band_weights, dim=1)  # (B, N, n_freq_bands)
        
        # Store for visualization
        self.last_band_weights = band_weights_stack.detach()
        
        # ───────────────────────────────────────────────────────────────────
        # Step 3: Reconstruct phase information
        # ───────────────────────────────────────────────────────────────────
        flat = attended_stack.view(B, -1)  # (B, n_freq_bands * N)
        reconstructed = self.reconstruct(flat)  # (B, N)
        
        # Normalize to valid phase range
        reconstructed = torch.remainder(reconstructed, 2 * np.pi)
        
        return reconstructed, band_weights_stack


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3: GRAPH-BASED OSCILLATOR ATTENTION (Most Powerful)
# ═══════════════════════════════════════════════════════════════════════════

class GraphOscillatorAttention(nn.Module):
    """
    Graph attention over oscillator network.
    Each oscillator attends to its neighbors + a few global nodes.
    """
    
    def __init__(self, n_osc: int, neighbor_indices: List[List[int]], 
                 n_global: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_osc = n_osc
        self.n_global = n_global
        self.dim = dim
        self.neighbor_indices = neighbor_indices
        
        # Global node embeddings (learned)
        self.global_nodes = nn.Parameter(torch.randn(n_global, dim) * 0.01)
        
        # Edge attention weights (per oscillator)
        self.edge_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
        # Message passing
        self.message_proj = nn.Linear(dim, dim)
        self.update_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store edge weights for visualization
        self.last_edge_weights = None
        
    def forward(self, phases: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phases: (B, N) oscillator phases
            hidden: (B, N, D) hidden states
            
        Returns:
            out: (B, N, D) updated hidden states
            edge_weights: (B, N, max_neighbors) attention weights
        """
        B, N, D = hidden.shape
        
        # ───────────────────────────────────────────────────────────────────
        # Step 1: Add global nodes as context
        # ───────────────────────────────────────────────────────────────────
        global_expanded = self.global_nodes.unsqueeze(0).expand(B, -1, -1)  # (B, n_global, D)
        all_nodes = torch.cat([hidden, global_expanded], dim=1)  # (B, N + n_global, D)
        
        # ───────────────────────────────────────────────────────────────────
        # Step 2: Compute attention for each oscillator's neighbors
        # ───────────────────────────────────────────────────────────────────
        updated = []
        max_neighbors = max(len(idx) for idx in self.neighbor_indices) + 1 + self.n_global
        edge_weights_padded = torch.zeros(B, N, max_neighbors, device=hidden.device)
        
        for i in range(N):
            # Get neighbors (plus self and global nodes)
            neighbor_idx = [i] + self.neighbor_indices[i] + list(range(N, N + self.n_global))
            neighbor_hidden = all_nodes[:, neighbor_idx, :]  # (B, n_neighbors + 1 + n_global, D)
            
            # Compute attention weights
            center_expanded = hidden[:, i:i+1, :].expand(-1, neighbor_hidden.size(1), -1)
            combined = torch.cat([center_expanded, neighbor_hidden], dim=-1)  # (B, neigh, 2D)
            attn_logits = self.edge_attn(combined).squeeze(-1)  # (B, neigh)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, neigh)
            attn_weights = self.dropout(attn_weights)
            
            # Store edge weights
            edge_weights_padded[:, i, :attn_weights.size(1)] = attn_weights
            
            # Weighted sum of neighbor messages
            messages = self.message_proj(neighbor_hidden)  # (B, neigh, D)
            weighted_message = (attn_weights.unsqueeze(-1) * messages).sum(dim=1)  # (B, D)
            
            # Update hidden state
            updated_i = self.update_proj(hidden[:, i, :] + weighted_message)
            updated.append(updated_i)
        
        # Store for visualization
        self.last_edge_weights = edge_weights_padded.detach()
        
        return torch.stack(updated, dim=1), edge_weights_padded


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 4: THE ARSENAL (Combines all three)
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttentionArsenal(nn.Module):
    """
    Combines all three attention mechanisms in a multi-head fashion.
    Each head uses a different attention strategy for maximum coverage.
    """
    
    def __init__(self, n_osc: int, neighbor_indices: List[List[int]], 
                 n_heads: int = 8, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_osc = n_osc
        self.dim = dim
        self.n_heads = n_heads
        
        # Determine how to split heads
        n_phase_heads = n_heads // 3
        n_freq_heads = n_heads // 3
        n_graph_heads = n_heads - n_phase_heads - n_freq_heads
        
        # Sub-attention modules
        self.phase_attn = PhaseCoherentAttention(n_osc, n_phase_heads, dim // 2, dropout)
        self.freq_attn = FrequencyAttention(n_osc, n_freq_heads, dim // 2, dropout)
        self.graph_attn = GraphOscillatorAttention(n_osc, neighbor_indices, n_graph_heads, dim // 2, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        # Store attention weights
        self.last_weights = {}
        
    def forward(self, phases: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            phases: (B, N) oscillator phases
            hidden: (B, N, D) hidden states
            
        Returns:
            out: (B, N, D) attended hidden states
            weights: Dictionary of attention weights for visualization
        """
        # Split hidden states
        split_size = self.dim // 3
        h_phase = hidden[:, :, :split_size]
        h_freq = hidden[:, :, split_size:2*split_size]
        h_graph = hidden[:, :, 2*split_size:3*split_size]
        
        # Apply each attention mechanism
        out_phase, w_phase = self.phase_attn(phases, h_phase)
        out_freq, w_freq = self.freq_attn(phases, h_freq)
        out_graph, w_graph = self.graph_attn(phases, h_graph)
        
        # Concatenate outputs
        combined = torch.cat([out_phase, out_freq, out_graph], dim=-1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Store weights for visualization
        self.last_weights = {
            'phase': w_phase,
            'freq': w_freq,
            'graph': w_graph,
        }
        
        return fused, self.last_weights


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY: Create the right attention mechanism
# ═══════════════════════════════════════════════════════════════════════════

def create_attention(attention_type: str, n_osc: int, neighbor_indices: List[List[int]] = None,
                     n_heads: int = 8, dim: int = 512, dropout: float = 0.1) -> nn.Module:
    """
    Factory function to create attention mechanisms.
    
    Args:
        attention_type: 'phase', 'freq', 'graph', 'arsenal', 'none'
        n_osc: Number of oscillators
        neighbor_indices: Neighbor indices (required for graph attention)
        n_heads: Number of attention heads
        dim: Hidden dimension
        dropout: Dropout rate
        
    Returns:
        nn.Module: Attention module
    """
    if attention_type == 'phase' or attention_type == 'phase_coherent':
        return PhaseCoherentAttention(n_osc, n_heads, dim, dropout)
    
    elif attention_type == 'freq' or attention_type == 'frequency':
        return FrequencyAttention(n_osc, n_heads, dim, dropout)
    
    elif attention_type == 'graph' or attention_type == 'graph_oscillator':
        if neighbor_indices is None:
            raise ValueError("neighbor_indices required for graph attention")
        return GraphOscillatorAttention(n_osc, neighbor_indices, n_heads, dim, dropout)
    
    elif attention_type == 'arsenal' or attention_type == 'multi':
        if neighbor_indices is None:
            raise ValueError("neighbor_indices required for arsenal attention")
        return MultiHeadAttentionArsenal(n_osc, neighbor_indices, n_heads, dim, dropout)
    
    elif attention_type == 'none' or attention_type is None:
        return nn.Identity()
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")