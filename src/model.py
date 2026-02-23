import torch
import torch.nn as nn
from .config import HyperMLPConfig
from .hypermlp import HyperMLPBlock
from typing import Optional
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
        

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
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
        x_embed = self.embeddings(input_ids)
        _, seq_len, _ = x_embed.shape
        layer_caches = [None] * len(self.blocks) 
        all_hidden_states = []
        for t in range(seq_len):
            x_t = x_embed[:, t:t+1, :]
            for idx, block in enumerate(self.blocks):
                x, kv_cache = block(x_t, layer_caches[idx])
                x_t = x
                layer_caches[idx] = kv_cache
            all_hidden_states.append(x_t)

        full_sequence = torch.cat(all_hidden_states, dim=1)

        # now pass the full sequnece to the final layers
        logits = self.head(self.ln_f(full_sequence))

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # Shape: [Batch*Seq_Len, Vocab]
                labels.view(-1) # Shape: [Batch*Seq_len]
            )
            return loss, logits
        
        return logits
