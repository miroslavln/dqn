import tensorflow as tf
from tensorflow.keras import layers

class VisionTransformer(tf.keras.Model):
    def __init__(self, num_actions):
        super(VisionTransformer, self).__init__()
        self.patch_size = 14
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_layers = 2
        self.num_actions = num_actions

        self.data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.Resizing(72, 72),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.02),
                layers.experimental.preprocessing.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        self.projection = layers.Dense(self.projection_dim)
        self.positional_embedding = layers.Embedding(
            input_dim=25, output_dim=self.projection_dim
        )
        self.transformer_blocks = [
            self.transformer_block() for _ in range(self.transformer_layers)
        ]
        self.classification_head = layers.Dense(self.num_actions)

    def transformer_block(self):
        return tf.keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
                ),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(self.projection_dim * 2, activation=tf.nn.gelu),
                layers.Dense(self.projection_dim),
            ]
        )

    def call(self, inputs):
        augmented = self.data_augmentation(inputs)
        patches = tf.image.extract_patches(
            images=augmented,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [-1, (72 // self.patch_size) ** 2, patch_dims])

        projected_patches = self.projection(patches)
        positions = tf.range(start=0, limit=(72 // self.patch_size) ** 2, delta=1)
        positional_embeddings = self.positional_embedding(positions)
        encoded_patches = projected_patches + positional_embeddings

        for transformer_block in self.transformer_blocks:
            encoded_patches = transformer_block(encoded_patches)

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        logits = self.classification_head(representation)
        return logits
