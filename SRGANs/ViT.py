import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Conv2D, Dense, Flatten, LayerNormalization, Reshape

# --- Vision Transformer Block ---
def vit_block_v1(input_tensor, num_patches, projection_dim, transformer_layers=2):
    """Applies Vision Transformer on the intermediate feature map."""
    print('vit v1 block-----------')
    print('input_tensor 1', input_tensor.shape)
    # Patch Embedding
    # Flatten layer: flattens 4D to 1D
    #x = Flatten()(input_tensor)
    #print("After Flatten:", x.shape)

    x = tf.keras.layers.Reshape((num_patches, -1))(input_tensor)  # Reshape to patches
    #x = tf.keras.layers.Reshape((num_patches, projection_dim))(input_tensor)  # Reshape to patches
    print('x after 1st Reshape', x.shape)
    x = tf.keras.layers.Dense(projection_dim)(x)  # Linear projection of patches
    print('x after Dense', x.shape)

    # Add Transformer Encoder Layers
    for _ in range(transformer_layers):
        # Layer Norm
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x_norm, x_norm)
        # Skip Connection
        x = tf.keras.layers.Add()([x, attention_output])
        # Layer Norm and Feedforward Network
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = tf.keras.layers.Dense(projection_dim, activation="relu")(x_norm)
        x = tf.keras.layers.Add()([x, ff_output])
        print('x shape 0.2', x.shape)

    # Reshape back to 2D image structure
    x = tf.keras.layers.Reshape((int(input_tensor.shape[1]), int(input_tensor.shape[2]), projection_dim))(x)
    #x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(x)
    print('x return', x.shape)
    print('End vit v1 block-----------')
    return x

def vit_block_v1_origin(input_tensor, num_patches, projection_dim, transformer_layers=2):
    """Applies Vision Transformer on the intermediate feature map."""
    print('vit v1 block-----------')
    print('input_tensor', input_tensor.shape)
    # Patch Embedding
    x = tf.keras.layers.Reshape((num_patches, -1))(input_tensor)  # Reshape to patches
    #x = Reshape((num_patches, projection_dim))(input_tensor)  # Reshape to patches
    print('x after 1st Reshape', x.shape)
    x = tf.keras.layers.Dense(projection_dim)(x)  # Linear projection of patches
    print('x after Dense', x.shape)
    #"""
    # Add Transformer Encoder Layers
    for _ in range(transformer_layers):
        # Layer Norm
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        print('x_norm shape 1', x_norm.shape)
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x_norm, x_norm)
        print('attention_output shape', attention_output.shape)
        # Skip Connection
        x = tf.keras.layers.Add()([x, attention_output])
        print('x shape 0.1', x.shape)
        # Layer Norm and Feedforward Network
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        print('x_norm shape 2', x_norm.shape)
        ff_output = tf.keras.layers.Dense(projection_dim, activation="relu")(x_norm)
        print('ff_output  shape', ff_output.shape)
        x = tf.keras.layers.Add()([x, ff_output])
        print('x shape 0.2', x.shape)
        
    # Reshape back to 2D image structure
    x = tf.keras.layers.Reshape((int(input_tensor.shape[1]), int(input_tensor.shape[2]), projection_dim))(x)
    print('x after 2 Reshape', x.shape)
    return x

def vit_block_v2(input_tensor, num_patches, projection_dim, transformer_layers=2):
    """Applies Vision Transformer on the intermediate feature map."""

    # Split into patches
    x = tf.keras.layers.Conv2D(
        filters=projection_dim,
        kernel_size=7,
        strides=1,
        padding="valid",
    )(input_tensor)  # Shape: (batch_size, num_patches_h, num_patches_w, projection_dim)
    print('vit v2 block-----------')
    print('x shape', x.shape)
    print('input_tensor shape', input_tensor.shape)
    print('num_patches', num_patches)
    x = tf.keras.layers.Reshape((num_patches, projection_dim))(x)  # Flatten patches into sequences
    print('x shape 0', x.shape)
    # Patch Embedding
    #x = Reshape((num_patches, -1))(input_tensor)  # Reshape to patches
    #x = Dense(projection_dim)(x)  # Linear projection of patches

    # Add Transformer Encoder Layers
    for _ in range(transformer_layers):
        # Layer Norm
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        print('x_norm shape 1', x_norm.shape)
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x_norm, x_norm)
        print('attention_output shape', attention_output.shape)
        # Skip Connection
        x = tf.keras.layers.Add()([x, attention_output])
        print('x shape 0.1', x.shape)
        # Layer Norm and Feedforward Network
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        print('x_norm shape 2', x_norm.shape)
        ff_output = tf.keras.layers.Dense(projection_dim, activation="relu")(x_norm)
        print('ff_output  shape', ff_output.shape)
        x = tf.keras.layers.Add()([x, ff_output])
        print('x shape 0.2', x.shape)

    # Reshape back to 2D image structure
    print('x shape 1', x.shape)
    print('input_tensor shape 1', input_tensor.shape)
    x = tf.keras.layers.Reshape((int(input_tensor.shape[1]), int(input_tensor.shape[2]), projection_dim))(x)
    # Upsample to match original resolution
    patch_size = 4
    #x = layers.UpSampling2D(size=(patch_size, patch_size))(x)
    print('x shape 2', x.shape)
    print('end of vit block-----------')
    return x


def vit_block_v3(input_tensor, patch_size, projection_dim, transformer_layers=2):
    """
    Vision Transformer Block with fixed reshaping for patches.

    Args:
    - input_tensor: 4D tensor, input feature map.
    - patch_size: Size of each patch (e.g., 2x2, 4x4).
    - projection_dim: Dimension for patch embeddings.
    - transformer_layers: Number of transformer encoder layers.

    Returns:
    - Tensor after ViT processing.
    """
    # Compute the number of patches
    batch_size, height, width, channels = input_tensor.shape
    num_patches = (height // patch_size) * (width // patch_size)
    print('start of vit block v3-----------')
    print('batch_size, height, width, channels', batch_size, height, width, channels)
    print('num_patches', num_patches)

    # Split into patches
    x = tf.keras.layers.Conv2D(
        filters=projection_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="valid",
    )(input_tensor)  # Shape: (batch_size, num_patches_h, num_patches_w, projection_dim)
    print('x shape 1', x.shape)

    #x = layers.Reshape((num_patches, projection_dim))(x)  # Flatten patches into sequences
    x = tf.keras.layers.Reshape((num_patches, -1))(x)  # Flatten patches into sequences
    print('x reshape before', x.shape)

    # Transformer Encoder Layers
    for _ in range(transformer_layers):
        # Layer Norm
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x_norm, x_norm)
        # Skip Connection
        x = tf.keras.layers.Add()([x, attention_output])
        # Layer Norm and Feedforward Network
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = tf.keras.layers.Dense(projection_dim, activation="relu")(x_norm)
        x = tf.keras.layers.Add()([x, ff_output])

    # Reshape back to image dimensions
    patch_height = height // patch_size
    patch_width = width // patch_size
    print('patch_height, patch_width', patch_height, patch_width)
    x = tf.keras.layers.Reshape((patch_height, patch_width, projection_dim))(x)
    print('x reshape after', x.shape)

    # Upsample to match original resolution
    x = tf.keras.layers.UpSampling2D(size=(patch_size, patch_size))(x)
    print('x upsampling2d', x.shape)
    print('end of vit block v3-----------')
    return x


