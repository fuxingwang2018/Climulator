# Modified from SRGANs/srgans_fw.py

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
from utils import read

class TrainModel(object):

    def __init__(self, wdir):

        self.wdir = wdir


    def training(self, batch_size, EPOCH_INIT, EPOCHS, dataset_train, dataset_valid):

        # https://www.tensorflow.org/tutorials/keras/save_and_load 
        # https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
        # https://mmsankosho.com/en/nlp-for-learners-interrupt-and-resume-trainingmodelcheckpoint/
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

        #checkpoint_filepath = self.wdir + 'checkpoint_NN'
        checkpoint_filepath = self.wdir + 'checkpoint_{epoch:04d}'
        print ('checkpoint_filepath', checkpoint_filepath)

        # Create a callback that saves the model's weights
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_gen_loss',
            mode='min',
            save_best_only=True)

        callbacks_list = [checkpoint]

        checkpoint_filepath_iniepoch = os.path.dirname(os.path.dirname(self.wdir)) + '/EPOCH' + str(EPOCH_INIT) + '/'
        # Load checkpoint:
        if os.path.isfile( checkpoint_filepath_iniepoch + '/checkpoint'):

            print ('file exists', checkpoint_filepath_iniepoch + 'checkpoint')
            # Load model:
            generator = load_model(checkpoint_filepath_iniepoch + f'{model_name}_generator.h5')
            discriminator = load_model(checkpoint_filepath_iniepoch + f'{model_name}_discriminator.h5')
            model = SRGAN(generator, discriminator)
            model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
            #print ('model compile:', model.summary())
           
            # Finding the epoch index from which we are resuming
            initial_epoch = self.get_init_epoch(checkpoint_filepath_iniepoch + 'checkpoint')
            print ('initial_epoch', initial_epoch)
       
            # Calculating the correct value of count
            count = initial_epoch * batch_size
            print ('count', count)

            # Update the value of count in callback instance
            #callbacks[1].count = count

            model.load_weights(checkpoint_filepath_iniepoch + 'checkpoint_' + f"{initial_epoch:04d}")
            loss, acc = model.evaluate(dataset_train, verbose=2)
            print('loss, acc using initial_epoch weights =', loss, acc)

        else:
            initial_epoch = 0

            generator =  model_generator(nx, nz, input_channels, subsampling_lr, n_res_block, batch_size)
            discriminator = model_discriminator(nx, nz, output_channels, batch_size)

            model = SRGAN(generator, discriminator)
            model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
            #print ('model compile:', model.summary())

        # Start/resume training
        # Train the model with the new callback
        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        hist = model.fit(dataset_train, 
            epochs = EPOCHS, 
            callbacks = callbacks_list, 
            validation_data = dataset_valid, 
            verbose = 1, 
            initial_epoch = EPOCH_INIT)

        print ('history:', hist.history)
        savemat(self.wdir + f'loss_{model_name}.mat', hist.history)

        # The latest model weights (that are considered the best) are loaded into the model.
        latest = tf.train.latest_checkpoint(os.path.dirname(self.wdir))
        print('latest', latest)
        model.load_weights(latest)
        loss, acc = model.evaluate(dataset_train, verbose=2)
        print('loss, acc using latest weights =', loss, acc)

        # Save the model to a HDF5 file.
        generator.save(self.wdir + f'{model_name}_generator.h5')
        discriminator.save(self.wdir + f'{model_name}_discriminator.h5')

        return generator


    def prediction(self, generator, X_test, y_test):

        pr_pred = generator.predict(X_test)
        np.savez_compressed(self.wdir + 'preds', hr = y_test, hr_p = pr_pred)

        return pr_pred


    def get_init_epoch(self, checkpoint_path):

        ## util function to get the initial epoch number from the checkpoint name
        #filename = os.path.basename(checkpoint_path)
        #print ('filename', filename)
        #filename = os.path.splitext(filename)[0]
        #print ('filename', filename)
        #init_epoch = filename.split("_")[1]

        reader = read.Read('ascii')
        col1, col2 = reader.read_ascii(checkpoint_path)
        print('col1',col1)
        print('col2',col2)
        print('col2 [0]',col2[0].strip())

        init_epoch = col2[0].split('_')[1].strip('"')
        print('init_epoch=', init_epoch)

        return int(init_epoch)
