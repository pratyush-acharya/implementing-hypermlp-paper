# HyperMLP: Attention as a Dynamic Two-Layer MLP

This project implements the **HyperMLP** architecture, a novel approach to sequence modeling that reinterprets the self-attention mechanism as a dynamic two-layer Multi-Layer Perceptron (MLP). based on the research paper _"HyperMLP: Attention as a Dynamic Two-Layer MLP"_.

## Project Overview

The core idea of HyperMLP is to replace the standard Query-Key-Value-Output (QKVO) attention mechanism with a dynamic weight generation process. Instead of computing attention scores directly, the model instantiates the weights of two MLP layers, $W^{(1)}_{MLP}$ and $W^{(2)}_{MLP}$, from the input context itself.

This architecture aims to provide a more expressive and potentially more efficient alternative to standard Transformers by leveraging:
- **Dynamic Weight Instantiation**: Weights are generated on-the-fly using the input sequence history.
- **Low-Rank Feature Mixing**: Efficient parameterization of the feature mixing matrices ($L^{(1)}$, $L^{(2)}$).
- **DPLR Sequence Mixing**: Diagonal Plus Low-Rank decomposition for sequence mixing matrices ($R^{(1)}$, $R^{(2)}$).
- **HyperGLU**: A Gated Linear Unit variant of the HyperMLP layer for enhanced performance.

