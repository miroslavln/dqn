import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, num_actions):
        super(VisionTransformer, self).__init__()
        self.patch_size = 14
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_layers = 2
        self.num_actions = num_actions
        self.num_patches = (84 // self.patch_size) ** 2

        self.projection = nn.Linear(self.patch_size * self.patch_size * 4, self.projection_dim)
        self.positional_embedding = nn.Embedding(self.num_patches, self.projection_dim)
        self.transformer_blocks = nn.ModuleList(
            [self.transformer_block() for _ in range(self.transformer_layers)]
        )
        self.classification_head = nn.Linear(self.projection_dim * self.num_patches, self.num_actions)

    def transformer_block(self):
        return nn.Sequential(
            nn.LayerNorm(self.projection_dim),
            nn.MultiheadAttention(
                embed_dim=self.projection_dim,
                num_heads=self.num_heads,
                dropout=0.1,
                batch_first=True,
            ),
            nn.LayerNorm(self.projection_dim),
            nn.Linear(self.projection_dim, self.projection_dim * 2),
            nn.GELU(),
            nn.Linear(self.projection_dim * 2, self.projection_dim),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), 4, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(x.size(0), self.num_patches, -1)

        projected_patches = self.projection(patches)

        positions = torch.arange(start=0, end=self.num_patches).expand(x.size(0), -1)
        positional_embeddings = self.positional_embedding(positions)
        encoded_patches = projected_patches + positional_embeddings

        for transformer_block in self.transformer_blocks:
            encoded_patches, _ = transformer_block[1](encoded_patches, encoded_patches, encoded_patches)

        representation = nn.LayerNorm(self.projection_dim)(encoded_patches)
        representation = representation.flatten(start_dim=1)
        representation = nn.Dropout(0.5)(representation)

        logits = self.classification_head(representation)
        return logits
