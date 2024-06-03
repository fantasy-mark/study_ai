#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 17:36
# @Author  : F1243749 Mark
# @File    : vit.py
# @Depart  : NPI-SW
# @Desc    :
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for units in hidden_units:
            self.linears.append(nn.Linear(in_features, units))
            in_features = units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for linear in self.linears:
            x = F.gelu(linear(x))
            x = self.dropout(x)
        return x


class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, channels)
        return patches


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection = nn.Linear(channels, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        positions = torch.arange(0, self.position_embedding.num_embeddings).unsqueeze(1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, patch_size, num_patches, projection_dim, transformer_layers, num_heads,
                 transformer_units, mlp_head_units, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.input_shape = input_shape
        self.channels = input_shape[0]
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(projection_dim, num_heads) for _ in range(transformer_layers)])
        self.mlp_head = MLP(projection_dim, mlp_head_units, 0.5)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x):
        # Augment data (data augmentation is not implemented here)
        # augmented = data_augmentation(x)

        # Create patches.
        patches = self.patch_encoder(x)

        # Transformer layers.
        for layer in self.transformer_layers:
            patches = layer(patches)

        # Representation.
        representation = patches.mean(dim=1)  # Global average pooling
        representation = self.mlp_head(representation)

        # Classification.
        logits = self.classifier(representation)
        return logits


# Helper class for the Transformer layer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, [d_model * 4], 0.1)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x)[0])
        x = self.norm2(x + self.mlp(x))
        return x


# Example usage:
input_shape = (1, 28, 28)  # Example input shape (C, H, W)
patch_size = 14
num_patches = (28 // 14) ** 2
projection_dim = 50
transformer_layers = 12
num_heads = 12
transformer_units = [3072, 768]
mlp_head_units = [3072, 768]
num_classes = 1000

vit_classifier = VisionTransformer(input_shape, patch_size, num_patches, projection_dim, transformer_layers, num_heads,
                                   transformer_units, mlp_head_units, num_classes)
