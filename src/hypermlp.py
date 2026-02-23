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
        if self.config.use_hyperglu:
            gate_path = self.feature_mix_qk(x_t)
            scale_path = self.feature_mix_qk_scale(x_t)
            q_t, m_gate, k_t = gate_path
            q_t_scale, m_gate_scale, k_t_scale = scale_path
            v_t, m_vo, w_o= self.feature_mix_vo(x_t)
            if past_key_values:
                updated_k_t = torch.cat((past_key_values[0], k_t), dim=1)
                updated_k_t_scale = torch.cat((past_key_values[1], k_t_scale), dim=1)
                updated_v_t = torch.cat((past_key_values[2], v_t), dim=1)
                past_key_values = (updated_k_t, updated_k_t_scale, updated_v_t)
            else:
                past_key_values = (k_t, k_t_scale, v_t)
            
            k = past_key_values[0]
            k_scale = past_key_values[1]
            z = torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
            z_scale = torch.mul(q_t_scale, m_gate_scale) @ k_scale.transpose(-2, -1)
            h_mix = self.seq_mix_1(x_t, z)
            norm = torch.linalg.norm(h_mix, dim=-1, keepdim=True)
            h_mix = h_mix/(norm + 1e-8)
            h_mix_scale =  self.seq_mix_1_scale(x_t, z_scale)
            h = torch.nn.functional.softplus(h_mix_scale) * torch.relu(h_mix)

        else:
            gate_path = self.feature_mix_qk(x_t)
        
            q_t, m_gate, k_t = gate_path
            v_t, m_vo, w_o = self.feature_mix_vo(x_t)
            if past_key_values:
                updated_k_t = torch.cat((past_key_values[0], k_t), dim=1)
                updated_v_t = torch.cat((past_key_values[1], v_t), dim=1)
                past_key_values = (updated_k_t, updated_v_t)
            else:
                past_key_values = (k_t, v_t)
            k = past_key_values[0]
            z =  torch.mul(q_t, m_gate) @ k.transpose(-2, -1)
            h_mix = self.seq_mix_1(x_t, z)
            norm = torch.linalg.norm(h_mix, dim=-1, keepdim=True)
            h_mix = h_mix/(norm + 1e-8)
            h = torch.relu(h_mix)
        

        readout_weights = self.seq_mix_2(x_t, h)
        context_vector = torch.matmul(readout_weights, past_key_values[-1])

        output = (context_vector * m_vo) @ w_o.T

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
        self.heads = nn.ModuleList([HyperMLPHead(self.config, self.layer_idx) for i in range(self.config.n_heads)])

    def forward(self, x: torch.Tensor, past_key_values=None):
        # Run each head and sum/aggregate outputs.
        sum = 0
        all_kv_cache: list = []
        for i, head in enumerate(self.heads):
            head_cache = past_key_values[i] if past_key_values is not None else None
            output, new_cache = head(x, head_cache)
            sum += output
            all_kv_cache.append(new_cache)
        
        return sum, all_kv_cache
        
