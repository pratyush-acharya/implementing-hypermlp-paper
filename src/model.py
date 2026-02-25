import torch
import torch.nn as nn
from .config import HyperMLPConfig
from .hypermlp import HyperMLPBlock
from typing import Optional, List, Tuple
import torch.nn.functional as F

class FullTransformerBlock(nn.Module):
    """
    Standard Transformer block using HyperMLP instead of standard Attention.
    Implements Pre-LN and residual connections as per Figure 3 in the paper.
    """
    def __init__(self, config: HyperMLPConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.hypermlp = HyperMLPBlock(config, layer_idx)
        
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )
        
    def forward(self, x, kv_cache=None):
        # x is shape (batch_size, 1, d_model)
        attn_out, new_cache = self.hypermlp(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class HyperMLPModel(nn.Module):
    """
    A Transformer-like model that uses HyperMLP blocks instead of standard Attention.
    
    Structure:
    - Token Embeddings
    - Positional Embeddings (if not using RoPE or other mechanisms integral to the block)
    - Stack of N `HyperMLPBlock` layers (alternating with MLPs/FFNs is standard but 
      HyperMLP itself is "Attention as MLP", so the block might be self-sufficient or
      follow standard Transformer `Block = Attention + MLP` pattern).
      
      Paper says: "We match the parameter budget of vanilla attention... HyperMLP will allocate
      a larger parameter share than vanilla attention in Transformer blocks due to additional low-rank
      sequence-mixing operators... for fair comparison... we match the parameter budget."
      
      This implies it's a drop-in replacement for the Attention layer in a Transformer Block.
    """
    
    def __init__(self, config: HyperMLPConfig):
        # Initialize components
        # - embeddings: nn.Embedding(vocab_size, d_model)
        # - blocks: nn.ModuleList([HyperMLPBlock(...) for _ in range(n_layers)])
        # - ln_f: LayerNorm
        # - head: Linear(d_model, vocab_size)
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList([FullTransformerBlock(self.config, idx) for idx in range(self.config.n_layers)])
        self.ln_f = nn.LayerNorm(self.embeddings.embedding_dim)
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size)
        

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple]] = None):
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            labels: LongTensor of shape (batch_size, seq_len) for loss calculation
            
        Returns:
            logits or (loss, logits)
        """
        # Implement forward pass
        # 1. Embed inputs
        # 2. Iterate through blocks (handle residual connections and norms here if not in block)
        # 3. Final norm and projection
        # 4. Compute loss if labels are provided
        x = self.embeddings(input_ids)
        new_caches = []

        # Pass the sequence entirely through the depth of the network
        for idx, block in enumerate(self.blocks):
            layer_cache = past_key_values[idx] if past_key_values is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)
        
        logits = self.head(self.ln_f(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, logits, new_caches) if labels is not None else (logits, new_caches)
