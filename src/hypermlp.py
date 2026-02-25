import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import HyperMLPConfig
from .layers import DPLRSequenceMixing, LowRankFeatureMixing

class HyperMLPHead(nn.Module):
    """
    A single head of the HyperMLP/HyperGLU layer.
    
    This module implements the "Attention as a Dynamic Two-Layer MLP" logic.
    Instead of standard QKVO attention, it defines:
    
    $$ h_t = x_t W^{(1)}_{MLP}(X_{t:1}) \\in \\mathbb{R}^{1 \\times t} $$
    $$ o_t = \sigma(h_t) W^{(2)}_{MLP}(X_{t:1}) \\in \\mathbb{R}^{1 \\times d} $$
    
    where the weights are instantiated from the context $X_{t:1}$ using:
    $$ W^{(1)}_{MLP}(X_{t:1}) := L^{(1)}(x_t) X_{t:1}^\\top R^{(1)}(x_t) $$
    $$ W^{(2)}_{MLP}(X_{t:1}) := R^{(2)\\top}(x_t) X_{t:1} L^{(2)\\top}(x_t) $$
    
    Architecture Components:
    1. Feature Mixing (L): Uses `LowRankFeatureMixing`.
    2. Sequence Mixing (R): Uses `DPLRSequenceMixing`.
    3. Activation: 
       - Standard HyperMLP: $\\sigma(z) = \\text{ReLU}(\\text{L2Norm}_t(z))$
       - HyperGLU: $a_t = \\text{Softplus}(h_t^{scale}) \\odot \\text{ReLU}(\\text{L2Norm}_t(h_t^{gate}))$
    
    Note on HyperGLU:
    If `use_hyperglu` is True, the first Layer $W^{(1)}$ is essentially duplicated (or split) 
    to produce two outputs: `h_gate` and `h_scale`.
    """
    
    def __init__(self, config: HyperMLPConfig, layer_idx: int):
        super().__init__()
        # Initialize components.
        # - feature_mix_qk: LowRankFeatureMixing (implements L^(1))
        # - feature_mix_vo: LowRankFeatureMixing (implements L^(2))
        # - seq_mix_1: DPLRSequenceMixing (implements R^(1))
        # - seq_mix_2: DPLRSequenceMixing (implements R^(2))
        # - If HyperGLU, handle the split logic for QK.
        self.config = config
        self.layer_idx = layer_idx
        self.feature_mix_qk = LowRankFeatureMixing(config=self.config, is_readout=False)
        self.feature_mix_vo = LowRankFeatureMixing(config=self.config, is_readout=True)
        self.seq_mix_1 = DPLRSequenceMixing(config=self.config, layer_idx=self.layer_idx)
        self.seq_mix_2 = DPLRSequenceMixing(config=self.config, layer_idx=self.layer_idx)
        if config.use_hyperglu:
            self.feature_mix_qk_scale = LowRankFeatureMixing(config=config, is_readout=False)
            self.seq_mix_1_scale = DPLRSequenceMixing(config=self.config, layer_idx=self.layer_idx)


    def _apply_causal_lag(self, z: torch.Tensor) -> torch.Tensor:
        """Transforms standard forward interactions into a causal lag layout."""
        T = z.size(-1)
        rows = torch.arange(T, device=z.device).view(-1, 1)
        cols = torch.arange(T, device=z.device).view(1, -1)
        
        # Calculate lag index: t - age
        indices = rows - cols 
        safe_indices = torch.clamp(indices, min=0)
        safe_indices = safe_indices.expand(z.size(0), z.size(1), T, T)
        
        # Gather and apply causal mask to annihilate clamped future garbage
        z_lag = torch.gather(z, dim=-1, index=safe_indices)
        return torch.tril(z_lag)

    def forward(self, x_t: torch.Tensor, past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """
        Forward pass for one step (or sequence of steps).
        
        Args:
            x_t: Input features.
            past_key_values: Cache for autoregressive generation.
            
        Returns:
            Output tensor o_t.
        """
        # Implement the dynamic MLP forward pass.
        #
        # 1. Feature Mixing (QK side):
        #    Combine x_t with L^(1) parameters to get effective Q and K projection factors.
        #    Recall: x L X^T R = (x W_q) (X W_k)^T R (roughly)
        #
        # 2. Sequence Instantiation & Mixing (R^(1)):
        #    Compute the "hidden scores" h_t.
        #    - Instantiate pool from history X using QK factors.
        #    - Apply DPLR mixing R^(1): h_t = (EffectiveQK) @ R^(1).
        #
        # 3. Activation:
        #    - Apply L2Norm along sequence dimension.
        #    - Apply ReLU (or HyperGLU logic).
        #
        # 4. Sequence Mixing (R^(2)):
        #    - Apply R^(2) to the activated scores: w_t = \sigma(h_t) R^(2)^T.
        #
        # 5. Readout (VO side):
        #    - Aggregate values from history X using w_t as weights.
        #    - Apply L^(2) feature mixing (VO factors).
        v_t, m_vo, w_o = self.feature_mix_vo(x_t)
        
        if self.config.use_hyperglu:
            q_t, m_gate, k_t = self.feature_mix_qk(x_t)
            q_t_scale, m_gate_scale, k_t_scale = self.feature_mix_qk_scale(x_t)
            
            if past_key_values is not None:
                # Autoregressive Generation: Append to lag-ordered prefix (dim=2)
                k = torch.cat((k_t, past_key_values[0]), dim=2)
                k_scale = torch.cat((k_t_scale, past_key_values[1]), dim=2)
                v = torch.cat((v_t, past_key_values[2]), dim=2)
                past_key_values = (k, k_scale, v)
                
                z = torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
                z_scale = torch.mul(q_t_scale, m_gate_scale) @ k_scale.transpose(-2, -1)
            else:
                # Parallel Training: Full sequence projection
                k, k_scale, v = k_t, k_t_scale, v_t
                
                z_raw = torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
                z_scale_raw = torch.mul(q_t_scale, m_gate_scale) @ k_scale.transpose(-2, -1)
                
                # Align tensor geometry to lag layout
                z = self._apply_causal_lag(z_raw)
                z_scale = self._apply_causal_lag(z_scale_raw)
                past_key_values = None 

            h_mix = self.seq_mix_1(x_t, z)
            norm = torch.linalg.norm(h_mix, dim=-1, keepdim=True)
            h_mix = h_mix / (norm + 1e-8)
            h_mix_scale = self.seq_mix_1_scale(x_t, z_scale)
            h = torch.nn.functional.softplus(h_mix_scale) * torch.relu(h_mix)

        else:
            q_t, m_gate, k_t = self.feature_mix_qk(x_t)
            
            if past_key_values is not None:
                k = torch.cat((k_t, past_key_values[0]), dim=2)
                v = torch.cat((v_t, past_key_values[1]), dim=2)
                past_key_values = (k, v)
                z = torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
            else:
                k, v = k_t, v_t
                z_raw = torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
                z = self._apply_causal_lag(z_raw)
                past_key_values = None

            h_mix = self.seq_mix_1(x_t, z)
            norm = torch.linalg.norm(h_mix, dim=-1, keepdim=True)
            h_mix = h_mix / (norm + 1e-8)
            h = torch.relu(h_mix)
        
        # Final Readout Mixing 
        readout_weights = self.seq_mix_2(x_t, h)
        context_vector = torch.matmul(readout_weights, v)
        
        # Fuse multi-head dimensions back to d_model
        intermediate = context_vector * m_vo
        output = intermediate.permute(0, 2, 1, 3).reshape(intermediate.size(0), intermediate.size(2), -1) @ w_o.T

        return output, past_key_values


class HyperMLPBlock(nn.Module):
    """
    Multi-Head HyperMLP Block.
    
    Aggregates outputs from $n_{heads}$ parallel `HyperMLPHead`s.
    Following the paper, multi-head attention is simply a sum of parallel dynamic MLPs.
    Since $W_o$ is absorbed into the head parameterization $L^{(2)}$,
    we just sum the outputs (or concatenate and project if defined differently, 
    but paper says "concatenating... followed by shared output projection is equivalent...").
    
    The paper suggests $n_{heads}=2$ is sufficient.
    """
    def __init__(self, config: HyperMLPConfig, layer_idx: int):
        super().__init__()
        # Initialize list of HyperMLPHead
        self.config = config
        self.layer_idx = layer_idx
        self.heads = HyperMLPHead(self.config, self.layer_idx)

    def forward(self, x: torch.Tensor, past_key_values=None):
        # Run each head and sum/aggregate outputs.
        # sum = 0
        # all_kv_cache: list = []
        # for i, head in enumerate(self.heads):
        #     head_cache = past_key_values[i] if past_key_values is not None else None
        #     output, new_cache = head(x, head_cache)
        #     sum += output
        #     all_kv_cache.append(new_cache)
        
        # return sum, all_kv_cache

        output, kv_cache = self.heads(x, past_key_values)
        return output, kv_cache
        
