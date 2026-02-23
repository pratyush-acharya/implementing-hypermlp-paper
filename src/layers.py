import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import HyperMLPConfig

class DPLRSequenceMixing(nn.Module):
    """
    Diagonal-Plus-Low-Rank (DPLR) Sequence Mixing Operator.

    This layer implements the input-conditioned sequence-space mixing $R^{(j)}(x_t) \\in \\mathbb{R}^{t \\times t}$.
    It is parameterized as:
    
    $$ R^{(j)}(x_t) = D^{(j)} + A^{(j)} \\text{Diag}(s^{(j)}(x_t)) B^{(j)\\top} $$
    
    where:
    - $D^{(j)} \\in \\mathbb{R}^{t \\times t}$ is a diagonal term (typically $I + \\text{Diag}(p)$).
    - $A^{(j)} \\in \\mathbb{R}^{t \\times r_s}$ and $B^{(j)} \\in \\mathbb{R}^{t \\times r_s}$ are low-rank factor banks.
    - $s^{(j)}(x_t) = \\phi(x_t W^{(j)}_S) \\in \\mathbb{R}^{r_s}$ is the input-dependent gate.
    - $W^{(j)}_S \\in \\mathbb{R}^{d \\times r_s}$ is a learnable projection.
    - $\\phi$ is the activation, typically Sigmoid.

    Implementation Note:
    To support autoregressive generation efficiently, we do NOT materialize the full $t \\times t$ matrix.
    Instead, we use the vector-DPLR multiplication identity (Lemma G.1 in the paper) to compute $y R$ or $y R^\\top$
    in $O(t r_s)$ time.
    
    For fixed maximum length $L$, parameters $A^{(j)}, B^{(j)} \\in \\mathbb{R}^{L \\times r_s}$ and diagonal $p \\in \\mathbb{R}^L$
    are stored. At step $t$, we slice the first $t$ rows (lag-ordered) to form $A^{(j)}_{1:t}, B^{(j)}_{1:t}$.
    
    Slicing Rule (Lag Layout):
    $s_{start} = L - t$.
    $A_{1:t} \\leftarrow A[s_{start}:]$, $B_{1:t} \\leftarrow B[s_{start}:]$, etc.
    """
    
    def __init__(self, config: HyperMLPConfig, layer_idx: int):
        super().__init__()
        # Initialize learnable parameters:
        # - W_s: Linear(d_model, rank_s)
        # - A, B: Parameter(max_seq_len, rank_s)
        # - p: Parameter(max_seq_len)
        self.w_s = nn.Linear(config.d_model, config.rank_s)
        self.A = nn.Parameter(torch.randn(config.max_seq_len, config.rank_s))
        self.B = nn.Parameter(torch.randn(config.max_seq_len, config.rank_s))
        self.p = nn.Parameter(torch.randn(config.max_seq_len))
        self.layer_idx = layer_idx

    def forward(self, x_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the sequence mixing operator to a vector sequence y.
        
        Args:
            x_t: The current timestep input feature $x_t$ used to compute the gate $s(x_t)$. 
                 Shape: (batch_size, d_model).
            y: The sequence vector(s) to be mixed. This is typically the pre-activation $z_t$ or 
               intermediate state. Shape: (batch_size, heads, seq_len) or similar.
               
        Returns:
            The mixed sequence $y R^{(j)}(x_t)$ (or transpose depending on usage).
        """
        # Implement the DPLR multiplication logic.
        # 1. Compute gate s_t = sigmoid(x_t @ W_s)
        # 2. Slice A, B, p for current sequence length t.
        # 3. Apply Lemma G.1: yR = y \odot (1+p) + ((yA) \odot s^T) B^T
        # computing gate s_t
        s_t = torch.sigmoid(self.w_s(x_t))
        # getting the slice
        t = y.shape[2] # shape (batch_size, heads, seq_len)
        start = self.p.shape[0] - t
        A_t = self.A[start:]
        B_t = self.B[start:]
        p_t = self.p[start:]

        # first diagonal Term
        D = torch.mul(y, (1+p_t)) 
        # then let's project y down using A
        y_proj_down = y @ A_t
        # gating and expansion
        gating = torch.matmul(torch.mul(y_proj_down, s_t), B_t.T)

        return D + gating


class LowRankFeatureMixing(nn.Module):
    """
    Low-Rank Input-Conditioned Feature Mixing.

    This layer implements the feature-space mixing operators $L^{(1)}$ (for QK) or $L^{(2)}$ (for VO).
    
    For the first layer (QK path):
    $$ L^{(1)}(x_t) = W_q M^{(1)}(x_t) W_k^\\top $$
    with $W_q, W_k \\in \\mathbb{R}^{d \\times d_{qk}}$.
    
    For the second layer (VO path):
    $$ L^{(2)\\top}(x_t) = W_v M^{(2)}(x_t) W_o^\\top $$
    with $W_v, W_o \\in \\mathbb{R}^{d \\times d_{vo}}$.
    
    The diagonal core $M(x_t)$ provides input-conditioning (gating):
    $$ M^{(k)}(x_t) = \\text{Diag}(\\phi(x_t W^{(k)}_M)) $$
    where $W^{(k)}_M \\in \\mathbb{R}^{d \\times d_{core}}$ (where core rank is $d_{qk}$ or $d_{vo}$).
    
    This structure explains why we can compress QK ranks without losing too much expressivity,
    as they primarily constrain the "routing" geometry.
    """
    
    def __init__(self, config: HyperMLPConfig, is_readout: bool = False):
        """
        Args:
            config: HyperMLPConfig object.
            is_readout: If False, implements L^(1) (QK side). If True, implements L^(2) (VO side).
                        This determines whether we use d_qk or d_vo.
        """
        super().__init__()
        # If is_readout (VO):
        #   - W_v: Linear(d_model, d_vo)
        #   - W_o: Linear(d_vo, d_model)  <- Note direction for L^(2)^T
        #   - W_m: Linear(d_model, d_vo) for the gate M
        # Else (QK):
        #   - W_q: Linear(d_model, d_qk)
        #   - W_k: Linear(d_model, d_qk)
        #   - W_m: Linear(d_model, d_qk) for the gate M
        self.is_readout = is_readout
        if self.is_readout:
            self.w_v = nn.Linear(config.d_model, config.d_vo, bias=False)
            self.w_o = nn.Linear(config.d_vo, config.d_model, bias=False)
            self.w_m = nn.Linear(config.d_model, config.d_vo, bias=False)
        else:
            self.w_q = nn.Linear(config.d_model, config.d_qk, bias=False)
            self.w_k = nn.Linear(config.d_model, config.d_qk, bias=False)
            self.w_m = nn.Linear(config.d_model, config.d_qk, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the components needed for the mixing.
        
        Args:
            x: Input tensor (batch_size, ..., d_model).
            
        Returns:
            A tuple of (part1, gate, part2).
            For QK: returns (q, m_gate, k) where q = x W_q, k = x W_k.
            For VO: returns (v, m_gate, o) where v = x W_v, o = x W_o.
            
            The actual matrix $L$ is never formed fully. 
            The effective operation is $x L X^T = (x W_q \\odot m) (X W_k)^T$.
        """
        if self.is_readout:
            v = self.w_v(x)
            o = self.w_o.weight
            m_gate = torch.sigmoid(self.w_m(x))
            return (v, m_gate, o)

        q = self.w_q(x)
        k = self.w_k(x)
        m_gate = torch.sigmoid(self.w_m(x))

        return (q, m_gate, k)


