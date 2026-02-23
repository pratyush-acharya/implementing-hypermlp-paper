from dataclasses import dataclass

@dataclass
class HyperMLPConfig:
    """
    Configuration class for HyperMLP model.

    This class holds all the hyperparameters required to instantiate the HyperMLP layers and model.
    It should include dimensions for features, ranks for low-rank approximations, and flags for
    architecture variants (e.g., HyperGLU).

    Attributes:
        d_model (int): The embedding dimension ($d$).
        n_heads (int): Number of heads ($n_{heads}$). Paper recommends 2 for HyperMLP.
        max_seq_len (int): Maximum sequence length ($T$ or $L$) for pre-allocating DPLR parameters.
        
        # Dimensions for factorized feature mixing
        d_qk (int): Rank for the first-layer (query/key) feature mixing ($d_{qk}$).
                    Typically $d_{qk} \\approx d / (4 \\times n_{heads})$.
        d_vo (int): Rank for the second-layer (value/output) feature mixing ($d_{vo}$).
                    Typically $d_{vo} \\approx d / n_{heads}$.
        
        # Dimensions for sequence mixing
        rank_s (int): Rank for the diagonal-plus-low-rank (DPLR) sequence mixing ($r_s$).
                      Paper uses 16.
        
        use_hyperglu (bool): If True, use the HyperGLU variant with split routing/scale paths.
                             If False, use standard HyperMLP with ReLU activation.
                             
        # Regularization (Optional)
        dropout (float): Dropout probability.
    """
    d_model: int = 768
    n_heads: int = 2
    max_seq_len: int = 1024
    d_qk: int = 96  # 768 / (4 * 2) = 96
    d_vo: int = 384 # 768 / 2 = 384
    rank_s: int = 16
    use_hyperglu: bool = True
    dropout: float = 0.0
    batch_size: int = 8
    vocab_size: int =1000
    n_layers: int = 2