import tensorflow as tf
from tensorflow.keras import models, metrics
from utils import gpus_func 


num_gpus = gpus_func.get_num_gpus()

def conditional_tf_function(func):
    """Apply @tf.function only if num_gpus < 2."""
    if num_gpus < 2:
        return tf.function(func)
    return func  # Return the function as-is if num_gpus >= 2


class SRGAN(models.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super(SRGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.loss_tracker_1 = metrics.Mean(name="gen_loss")
        self.loss_tracker_2 = metrics.Mean(name="disc_loss")
        self.content_loss_tracker = metrics.Mean(name="content_loss")
        self.adv_loss_tracker = metrics.Mean(name="adv_loss")
        self.real_loss_tracker = metrics.Mean(name="real_loss")
        self.fake_loss_tracker = metrics.Mean(name="fake_loss") 

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super(SRGAN, self).compile()
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.gen_loss = generator_loss
        self.disc_loss = discriminator_loss

    #@tf.function
    @conditional_tf_function
    def train_step(self, data):
        lr_predic = data[0]
        hr_predic = data[1]

        with tf.GradientTape(persistent = True) as tape:

            generated_batch = self.generator(lr_predic, training=True)
            
            real_ptv = self.discriminator(hr_predic, training=True)
            fake_ptv = self.discriminator(generated_batch, training=True)
            #print('lr_predic:', type(lr_predic), lr_predic)
            #print('fake_ptv.shape:', fake_ptv.shape)
            #print('generated_batch.shape:', generated_batch.shape)
            #print('hr_predic.shape:', hr_predic.shape)
            gen_loss, content_loss, adversarial_loss = self.gen_loss(fake_ptv, generated_batch, hr_predic)
            disc_loss, real_loss, fake_loss = self.disc_loss(real_ptv, fake_ptv)

        gradients_of_generator = tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        self.loss_tracker_1.update_state(gen_loss)
        self.loss_tracker_2.update_state(disc_loss)
        self.content_loss_tracker.update_state(content_loss)
        self.adv_loss_tracker.update_state(adversarial_loss)
        self.real_loss_tracker.update_state(real_loss)
        self.fake_loss_tracker.update_state(fake_loss)

        return {"gen_loss": self.loss_tracker_1.result(), 
                "disc_loss": self.loss_tracker_2.result(),
                "content_loss": self.content_loss_tracker.result(),
                "adv_loss": self.adv_loss_tracker.result(),
                "real_loss": self.real_loss_tracker.result(),
                "fake_loss": self.fake_loss_tracker.result(),
                }

    #@tf.function
    @conditional_tf_function
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
        return {"gen_loss": self.loss_tracker_1.result(), "disc_loss": self.loss_tracker_2.result(),
                "content_loss": self.content_loss_tracker.result(),
                "adv_loss": self.adv_loss_tracker.result(),
                "real_loss": self.real_loss_tracker.result(),
                "fake_loss": self.fake_loss_tracker.result(),
                }

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2, self.content_loss_tracker, self.adv_loss_tracker, 
                self.real_loss_tracker, self.fake_loss_tracker]
