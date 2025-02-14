from tensorflow.keras import layers
import tensorflow as tf

class SwinTransformerBlock(layers.Layer):
    def __init__(self, num_heads, embed_dim, window_size, mlp_dim):
        super(SwinTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.mlp_dim = mlp_dim

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, attention_axes=(1, 2))

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])

    def call(self, x):
        # Multi-head Self Attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = layers.Add()([residual, x])

        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = layers.Add()([residual, x])

        return x

def window_partition(x, window_size):
    """Partition image into non-overlapping windows."""
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # (B, num_windows_H, num_windows_W, window_size, window_size, C)
    x = tf.reshape(x, (-1, window_size, window_size, C))
    return x

def window_reverse(windows, window_size, H, W):
    """Reverse partitioning of windows to reconstruct the image."""
    B = tf.shape(windows)[0] // (H // window_size * W // window_size)
    #B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, (B, H, W, -1))
    return x

