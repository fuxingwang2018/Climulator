import tensorflow as tf
import sys
sys.path.insert(0, '..')
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LayerNormalization, Reshape
from SRGANs import ViT

# --- Discriminator ---
def build_discriminator_v1():
    inputs = layers.Input(shape=(88, 88, 1))  # High-resolution precipitation as input
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="discriminator")


# --- GAN Model ---
class VitSRGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(VitSRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")

    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn):
        super(VitSRGAN, self).compile()
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        low_res_inputs, real_high_res = data
        orography = low_res_inputs[1]
        low_res_inputs = low_res_inputs[0]

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            fake_high_res = self.generator([low_res_inputs, orography], training=True)
            real_logits = self.discriminator(real_high_res, training=True)
            fake_logits = self.discriminator(fake_high_res, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_logits), real_logits) + \
                     self.loss_fn(tf.zeros_like(fake_logits), fake_logits)

        grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train generator
        with tf.GradientTape() as gen_tape:
            fake_high_res = self.generator([low_res_inputs, orography], training=True)
            fake_logits = self.discriminator(fake_high_res, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_logits), fake_logits)

        grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker]

"""
# Instantiate models
generator = build_vit_generator()
discriminator = build_discriminator()
vit_srgan = VitSRGAN(generator, discriminator)

# Compile and train the GAN
vit_srgan.compile(
    generator_optimizer=tf.keras.optimizers.Adam(1e-4),
    discriminator_optimizer=tf.keras.optimizers.Adam(1e-4),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)
"""

def build_vit_generator(nx, nz):

    inputs = layers.Input(shape=(nx, nz, 4))  # Input low-res variables
    orography = layers.Input(shape=(nx * 4, nz * 4, 1))  # Orography as input

    # Initial convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Vision Transformer Block
    #vit_output = vit_block(x, patch_size=4, projection_dim=64)
    vit_output = ViT.vit_block_v1(x, num_patches = nx * nz, projection_dim=64, transformer_layers=2)

    # Upsampling and concatenation with orography
    x = layers.Conv2DTranspose(64, (4, 4), strides=(4, 4), padding='same', activation='relu')(vit_output)
    x = layers.Concatenate()([x, orography])
    x = layers.Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same', activation='relu')(x)

    # Final output layer
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model([inputs, orography], outputs, name="vit_generator")



nx, nz = 22, 26
test_size = 3000
# Prepare inputs and orography
inputs = tf.random.normal((test_size, nx, nz, 4))  # Batch size 1, low-res variables
orography = tf.random.normal((test_size, nx * 4, nz * 4, 1))  # Batch size 1, orography

# Predict
generator = build_vit_generator(nx, nz)
output = generator.predict([inputs, orography])
print("Generated output shape:", output.shape)

