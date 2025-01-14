import tensorflow as tf
from tensorflow.keras import models, metrics


class SRGAN(models.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super(SRGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.loss_tracker_1 = metrics.Mean(name="gen_loss")
        self.loss_tracker_2 = metrics.Mean(name="disc_loss")
        

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super(SRGAN, self).compile()
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.gen_loss = generator_loss
        self.disc_loss = discriminator_loss

    @tf.function
    def train_step(self, data):
        lr_predic = data[0]
        hr_predic = data[1]

        with tf.GradientTape(persistent = True) as tape:

            generated_batch = self.generator(lr_predic, training=True)
            
            real_ptv = self.discriminator(hr_predic, training=True)
            fake_ptv = self.discriminator(generated_batch, training=True)
            gen_loss = self.gen_loss(fake_ptv, generated_batch, hr_predic)
            disc_loss = self.disc_loss(real_ptv, fake_ptv)

        gradients_of_generator = tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        self.loss_tracker_1.update_state(gen_loss)
        self.loss_tracker_2.update_state(disc_loss)
        return {"gen_loss": self.loss_tracker_1.result(), "disc_loss": self.loss_tracker_2.result()}

    @tf.function
    def test_step(self, data):
        lr_predic = data[0]
        hr_predic = data[1]
        generated_batch = self.generator(lr_predic, training=False)

        real_ptv = self.discriminator(hr_predic, training=False)
        fake_ptv = self.discriminator(generated_batch, training=False)

        gen_loss = self.gen_loss(fake_ptv, generated_batch, hr_predic)
        disc_loss = self.disc_loss(real_ptv, fake_ptv)
          
        self.loss_tracker_1.update_state(gen_loss)
        self.loss_tracker_2.update_state(disc_loss)
        return {"gen_loss": self.loss_tracker_1.result(), "disc_loss": self.loss_tracker_2.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
