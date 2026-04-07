"""Utility functions for the flow model."""

import torch
import torch.nn as nn


class AttentionMapExtractor:
    """Extract attention maps from PointCloudDiT model.

    Since Flash Attention doesn't return attention weights, this class uses
    hooks to capture QKV tensors and computes attention maps externally.

    Example:
        >>> extractor = AttentionMapExtractor(model)
        >>> extractor.register_hooks()
        >>> with torch.no_grad():
        ...     output, _ = model(x, timesteps, latent, scales, anchor_indices)
        >>> attn_maps = extractor.compute_attention_maps(B, N)
        >>> extractor.remove_hooks()
        >>> # Access: attn_maps[(layer_idx, 'global')] -> (B, num_heads, N, N)
    """

    def __init__(self, model: nn.Module):
        """Initialize the attention map extractor.

        Args:
            model: PointCloudDiT model instance.
        """
        self.model = model
        self.attention_maps = {}
        self.qkv_cache = {}
        self.hooks = []

        # Cache model config
        self.num_heads = model.num_heads
        self.head_dim = model.embed_dim // model.num_heads

    def _make_hook(self, layer_idx: int, attn_type: str):
        """Create a forward hook for capturing QKV tensors.

        Args:
            layer_idx: Index of the transformer layer.
            attn_type: Type of attention ('part' or 'global').
        """
        def hook(module, input, output):
            self.qkv_cache[(layer_idx, attn_type)] = output.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks on all QKV projection layers."""
        self.clear()
        for i, layer in enumerate(self.model.transformer_layers):
            h1 = layer.self_qkv_proj.register_forward_hook(
                self._make_hook(i, 'part'))
            h2 = layer.global_qkv_proj.register_forward_hook(
                self._make_hook(i, 'global'))
            self.hooks.extend([h1, h2])

    def compute_attention_maps(
        self,
        batch_size: int,
        seq_len: int,
        softcap: float = 0.0,
    ) -> dict:
        """Compute attention maps from cached QKV tensors.

        Must be called after a forward pass with hooks registered.

        Args:
            batch_size: Batch size (B).
            seq_len: Sequence length (N).
            softcap: Soft cap for attention scores (matches model config).

        Returns:
            Dictionary mapping (layer_idx, attn_type) to attention maps
            of shape (B, num_heads, N, N).
        """
        self.attention_maps = {}

        for (layer_idx, attn_type), qkv in self.qkv_cache.items():
            # Reshape QKV: (B*N, 3*dim) or (B, N, 3*dim) -> (B, N, 3, num_heads, head_dim)
            if qkv.dim() == 2:
                qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            else:
                qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

            q, k, v = qkv.unbind(dim=2)  # Each: (B, N, num_heads, head_dim)
            q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
            k = k.transpose(1, 2)

            # Compute attention scores
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)

            # Apply softcap if specified
            if softcap > 0:
                attn_scores = attn_scores / softcap
                attn_scores = torch.tanh(attn_scores) * softcap

            # Softmax to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)
            self.attention_maps[(layer_idx, attn_type)] = attn_weights.cpu()

        return self.attention_maps

    def get_layer_attention(
        self,
        layer_idx: int,
        attn_type: str = 'global'
    ) -> torch.Tensor:
        """Get attention map for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.
            attn_type: Type of attention ('part' or 'global').

        Returns:
            Attention map of shape (B, num_heads, N, N).
        """
        key = (layer_idx, attn_type)
        if key not in self.attention_maps:
            raise KeyError(f"Attention map for {key} not found. "
                          f"Available: {list(self.attention_maps.keys())}")
        return self.attention_maps[key]

    def get_mean_attention(self, attn_type: str = 'global') -> torch.Tensor:
        """Get mean attention map across all layers and heads.

        Args:
            attn_type: Type of attention ('part' or 'global').

        Returns:
            Mean attention map of shape (B, N, N).
        """
        maps = [v for (_, t), v in self.attention_maps.items() if t == attn_type]
        if not maps:
            raise ValueError(f"No attention maps found for type '{attn_type}'")
        stacked = torch.stack(maps, dim=0)  # (num_layers, B, num_heads, N, N)
        return stacked.mean(dim=(0, 2))  # (B, N, N)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        """Clear cached data and remove hooks."""
        self.remove_hooks()
        self.attention_maps = {}
        self.qkv_cache = {}

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class RepresentationExtractor:
    """Extract intermediate representations from PointCloudDiT transformer layers.

    This class uses hooks to capture the output of each transformer layer,
    similar to the repr_dict in PointCloudDiT but with more flexibility.

    Example:
        >>> extractor = RepresentationExtractor(model)
        >>> extractor.register_hooks()
        >>> with torch.no_grad():
        ...     output, _ = model(x, timesteps, latent, scales, anchor_indices)
        >>> representations = extractor.get_representations()
        >>> extractor.remove_hooks()
        >>> # Access: representations[layer_idx] -> (B, N, embed_dim)
    """

    def __init__(self, model: nn.Module, layers: list = None, detach: bool = True, to_cpu: bool = True):
        """Initialize the representation extractor.

        Args:
            model: PointCloudDiT model instance.
            layers: List of layer indices to extract (0-indexed). If None, extract all layers.
            detach: Whether to detach tensors from computation graph. Default True.
            to_cpu: Whether to move tensors to CPU. Default True.
        """
        self.model = model
        self.layers = layers
        self.detach = detach
        self.to_cpu = to_cpu
        self.representations = {}
        self.hooks = []

        # Cache model config
        self.num_layers = model.num_layers
        self.embed_dim = model.embed_dim

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for capturing layer output.

        Args:
            layer_idx: Index of the transformer layer (0-indexed).
        """
        def hook(_module, _input, output):
            repr_tensor = output
            if self.detach:
                repr_tensor = repr_tensor.detach()
            if self.to_cpu:
                repr_tensor = repr_tensor.cpu()
            self.representations[layer_idx] = repr_tensor
        return hook

    def register_hooks(self):
        """Register forward hooks on transformer layers."""
        self.clear()
        for i, layer in enumerate(self.model.transformer_layers):
            # Skip if specific layers are requested and this isn't one
            if self.layers is not None and i not in self.layers:
                continue
            h = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

    def get_representations(self) -> dict:
        """Get all captured representations.

        Returns:
            Dictionary mapping layer_idx to tensor of shape (B, N, embed_dim).
        """
        return self.representations

    def get_layer_representation(self, layer_idx: int) -> torch.Tensor:
        """Get representation for a specific layer.

        Args:
            layer_idx: Index of the transformer layer (0-indexed).

        Returns:
            Tensor of shape (B, N, embed_dim).
        """
        if layer_idx not in self.representations:
            raise KeyError(f"Representation for layer {layer_idx} not found. "
                          f"Available: {list(self.representations.keys())}")
        return self.representations[layer_idx]

    def get_repa_representation(self, repa_layer: int = None) -> torch.Tensor:
        """Get representation at the REPA layer (for representation alignment).

        Args:
            repa_layer: REPA layer index (1-indexed, as in model config).
                       If None, uses model.repa_layer.

        Returns:
            Tensor of shape (B, N, embed_dim).
        """
        if repa_layer is None:
            repa_layer = self.model.repa_layer
        # Convert from 1-indexed (model config) to 0-indexed (our storage)
        layer_idx = repa_layer - 1
        return self.get_layer_representation(layer_idx)

    def get_stacked_representations(self, layers: list = None) -> torch.Tensor:
        """Get representations stacked along a new dimension.

        Args:
            layers: List of layer indices to stack. If None, stack all available.

        Returns:
            Tensor of shape (num_layers, B, N, embed_dim).
        """
        if layers is None:
            layers = sorted(self.representations.keys())
        tensors = [self.representations[i] for i in layers]
        return torch.stack(tensors, dim=0)

    def compute_layer_similarity(self, layer_a: int, layer_b: int) -> torch.Tensor:
        """Compute cosine similarity between representations of two layers.

        Args:
            layer_a: First layer index.
            layer_b: Second layer index.

        Returns:
            Tensor of shape (B, N) with cosine similarity for each point.
        """
        repr_a = self.get_layer_representation(layer_a)
        repr_b = self.get_layer_representation(layer_b)
        # Normalize
        repr_a = repr_a / repr_a.norm(dim=-1, keepdim=True)
        repr_b = repr_b / repr_b.norm(dim=-1, keepdim=True)
        # Cosine similarity
        return (repr_a * repr_b).sum(dim=-1)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        """Clear cached data and remove hooks."""
        self.remove_hooks()
        self.representations = {}

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


if __name__ == "__main__":
    # Example usage demonstrating both extractors
    from .dit import PointCloudDiT

    # Create a dummy model
    model = PointCloudDiT(
        in_dim=64,
        out_dim=6,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.0,
    )
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"Num layers: {model.num_layers}, Embed dim: {model.embed_dim}")
    print(f"REPA layer: {model.repa_layer}")

    # --- Representation Extractor Example ---
    print("\n--- RepresentationExtractor ---")
    repr_extractor = RepresentationExtractor(model)
    repr_extractor.register_hooks()
    print(f"Registered hooks on {len(repr_extractor.hooks)} layers")

    # Note: Actual forward pass requires proper inputs (x, timesteps, latent, scales, anchor_indices)
    # This is just to show the API:
    #
    # with torch.no_grad():
    #     output, interm_repr = model(x, timesteps, latent, scales, anchor_indices)
    #
    # representations = repr_extractor.get_representations()
    # print(f"Captured {len(representations)} layer representations")
    #
    # # Get specific layer
    # layer_2_repr = repr_extractor.get_layer_representation(2)  # 0-indexed
    # print(f"Layer 2 representation shape: {layer_2_repr.shape}")
    #
    # # Get REPA layer representation
    # repa_repr = repr_extractor.get_repa_representation()
    # print(f"REPA layer representation shape: {repa_repr.shape}")
    #
    # # Stack all representations
    # stacked = repr_extractor.get_stacked_representations()
    # print(f"Stacked representations shape: {stacked.shape}")

    repr_extractor.remove_hooks()
    print("Hooks removed")

    # --- Attention Map Extractor Example ---
    print("\n--- AttentionMapExtractor ---")
    attn_extractor = AttentionMapExtractor(model)
    attn_extractor.register_hooks()
    print(f"Registered hooks on {len(attn_extractor.hooks)} QKV projections")

    # Note: Actual forward pass requires proper inputs
    # This is just to show the API:
    #
    # with torch.no_grad():
    #     output, _ = model(x, timesteps, latent, scales, anchor_indices)
    #
    # B, N = x.shape[:2]
    # attn_maps = attn_extractor.compute_attention_maps(B, N)
    # print(f"Captured {len(attn_maps)} attention maps")
    #
    # # Get specific layer attention
    # layer_0_global = attn_extractor.get_layer_attention(0, 'global')
    # print(f"Layer 0 global attention shape: {layer_0_global.shape}")
    #
    # # Get mean attention across layers
    # mean_attn = attn_extractor.get_mean_attention('global')
    # print(f"Mean global attention shape: {mean_attn.shape}")

    attn_extractor.remove_hooks()
    print("Hooks removed")
