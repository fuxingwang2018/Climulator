import numpy as np
import os
from scipy.io import loadmat, savemat
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, optimizers, metrics, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from SRGANs.Model_Generator import model_generator
from SRGANs.Model_Discriminator import model_discriminator
from SRGANs.Losses import generator_loss, discriminator_loss
from SRGANs.srgan import SRGAN


class TrainModel(object):

    def __init__(self, wdir):
        self.wdir = wdir

    def training(self, batch_size, EPOCH_INIT, EPOCHS, dataset_train, dataset_valid):

        # https://mmsankosho.com/en/nlp-for-learners-interrupt-and-resume-trainingmodelcheckpoint/
        # https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
        # https://towardsdatascience.com/checkpointing-deep-learning-models-in-keras-a652570b8de6
 
        model_name = f"model_1"

        #generator_optimizer = optimizers.Adam(1e-4)
        #discriminator_optimizer = optimizers.Adam(1e-4)
        generator_optimizer = optimizers.legacy.Adam(1e-4)
        discriminator_optimizer = optimizers.legacy.Adam(1e-4)

        subsampling_lr = 4
        n_res_block = 8
        input_channels = 1
        output_channels = 1

        nx = 104
        nz = 88

        checkpoint_filepath = self.wdir + 'checkpoint_NN'
        print ('checkpoint_filepath', checkpoint_filepath)

        generator =  model_generator(nx, nz, input_channels, subsampling_lr, n_res_block, batch_size)
        discriminator = model_discriminator(nx, nz, output_channels, batch_size)

        model = SRGAN(generator, discriminator)
        model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)

        # Load checkpoint:
        if os.path.isfile(checkpoint_filepath + '.index'):

            # Load model:
            #generator = load_model(self.wdir + f'{model_name}_generator.h5')
            #discriminator = load_model(self.wdir + f'{model_name}_discriminator.h5')
            #model = SRGAN(generator, discriminator)
            #model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
           
            # Finding the epoch index from which we are resuming
            #initial_epoch = get_init_epoch(checkpoint_filepath)

            # Calculating the correct value of count
            #count = initial_epoch*batches_per_epoch

            # Update the value of count in callback instance
            #callbacks[1].count = count

            #latest = tf.train.latest_checkpoint(os.path.dirname(self.wdir))
            #model.load_weights(latest)
            model.load_weights(checkpoint_filepath)
            loss, acc = model.evaluate(dataset_train, verbose=2)
            print('loss, acc 1 =', loss, acc)

        #else:
        #    initial_epoch = 0

        # Create a callback that saves the model's weights
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_gen_loss',
            mode='min',
            save_best_only=True)

        callbacks_list = [checkpoint]

        # Start/resume training
        # Train the model with the new callback
        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        hist = model.fit(dataset_train, epochs = EPOCHS, callbacks = callbacks_list, validation_data = dataset_valid, verbose = 1, initial_epoch = EPOCH_INIT)

        #print ('model compile:', model.summary())
        print ('history:', hist.history)
        savemat(self.wdir + f'loss_{model_name}.mat', hist.history)

        # The model weights (that are considered the best) are loaded into the model.
        model.load_weights(checkpoint_filepath)

        # Save the model to a HDF5 file.
        generator.save(self.wdir + f'{model_name}_generator.h5')
        discriminator.save(self.wdir + f'{model_name}_discriminator.h5')

        return generator


    def prediction(self, generator, X_test, y_test):

        pr_pred = generator.predict(X_test)
        np.savez_compressed(self.wdir + 'preds', hr = y_test, hr_p = pr_pred)

        return pr_pred


    def get_init_epoch(self, checkpoint_path):

        # util function to get the initial epoch number from the checkpoint name
        filename = os.path.basename(checkpoint_path)
        print ('filename', filename)
        filename = os.path.splitext(filename)[0]
        print ('filename', filename)
        init_epoch = filename.split("-")[1]

        return int(init_epoch)
