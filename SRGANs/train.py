# Modified from SRGANs/srgans_fw.py

import numpy as np
import os
from scipy.io import loadmat, savemat
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras import layers, models, optimizers
#from tensorflow.keras.callbacks import ModelCheckpoint
from SRGANs.Model_Generator import model_generator
#from SRGANs.Model_Generator import model_generator_no_const_input
from SRGANs.Model_Discriminator import model_discriminator
from SRGANs.Losses import generator_loss, discriminator_loss
from SRGANs.srgan import SRGAN
from utils import read
from SRGANs.Network import Generator
import postprocess
from keras.models import Model
import glob

class TrainModel(object):

    def __init__(self, wdir):

        self.wdir = wdir


    def training(self, BATCH_SIZE, EPOCH_INIT, EPOCHS, 
        SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
        METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
        dataset_train, dataset_valid):

        # https://www.tensorflow.org/tutorials/keras/save_and_load 
        # https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
        # https://mmsankosho.com/en/nlp-for-learners-interrupt-and-resume-trainingmodelcheckpoint/
        # https://towardsdatascience.com/checkpointing-deep-learning-models-in-keras-a652570b8de6
 
        model_name = f"model_1"

        """
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE['GENERATOR'], decay_steps=10000, decay_rate=0.9
            )
        lr_schedule_dis = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE['DISCRIMINATOR'], decay_steps=10000, decay_rate=0.9
            )
        """
        generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE['GENERATOR']) #1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE['DISCRIMINATOR']) #1e-4)
        #generator_optimizer = tf.keras.optimizers.legacy.Adam(LEARNING_RATE) # (1e-4)
        #discriminator_optimizer = tf.keras.optimizers.legacy.Adam(LEARNING_RATE) #(1e-4)

        #subsampling_lr = 4
        #n_res_block = 8
        #input_channels = 1
        #output_channels = 1

        #nx = 104
        #ny = 88


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_gen_loss',  # Monitor validation generator loss
            patience=5,  # Stop training if no improvement for 5 epochs
            restore_best_weights=True,  # Restore best model weights
            verbose=1,
            mode="min"
        )
        #checkpoint_filepath = self.wdir + 'checkpoint_NN'
        checkpoint_filepath = self.wdir + 'checkpoint_{epoch:04d}.weights.h5'
        print ('checkpoint_filepath', checkpoint_filepath)

        # Create a callback that saves the model's weights
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_gen_loss',
            mode='min',
            save_best_only=True)
        #save_freq='epoch',

        callbacks_list = [checkpoint]

        if EARLY_STOP:
            callbacks_list = [checkpoint, early_stopping]

        checkpoint_filepath_iniepoch = os.path.dirname(os.path.dirname(self.wdir)) + '/EPOCH' + str(EPOCH_INIT) + '/'

        # Load checkpoint:
        if os.path.isfile( checkpoint_filepath_iniepoch + '/checkpoint'):

            print ('file exists', checkpoint_filepath_iniepoch + 'checkpoint')
            # Load model:
            generator = tf.keras.models.load_model(checkpoint_filepath_iniepoch + f'{model_name}_generator.h5')
            discriminator = tf.keras.models.load_model(checkpoint_filepath_iniepoch + f'{model_name}_discriminator.h5')
            model = SRGAN(generator, discriminator)
            model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
            #print ('model compile:', model.summary())
           
            # Finding the epoch index from which we are resuming
            initial_epoch = self.get_init_epoch(checkpoint_filepath_iniepoch + 'checkpoint')
            print ('initial_epoch', initial_epoch)
       
            # Calculating the correct value of count
            count = initial_epoch * BATCH_SIZE
            print ('count', count)

            # Update the value of count in callback instance
            #callbacks[1].count = count

            model.load_weights(checkpoint_filepath_iniepoch + 'checkpoint_' + f"{initial_epoch:04d}")
            loss, acc = model.evaluate(dataset_train, verbose=2)
            print('loss, acc using initial_epoch weights =', loss, acc)

        else:

            initial_epoch = 0

            #if np.shape(dataset_train.element_spec[0])[0] > 1:
            generator = model_generator(NX, NY, INPUT_CHANNELS, OUTPUT_CHANNELS, SUBSAMPLING_LR, N_RES_BLOCK, BATCH_SIZE, METHOD, DROPOUT_RATE)
            #else:
            #    generator = model_generator_no_const_input(NX, NY, INPUT_CHANNELS, SUBSAMPLING_LR, N_RES_BLOCK, BATCH_SIZE)
            #generator = Generator((int(NY / SUBSAMPLING_LR), int(NX / SUBSAMPLING_LR), INPUT_CHANNELS)).generator()
            discriminator = model_discriminator(NX, NY, OUTPUT_CHANNELS, BATCH_SIZE, DROPOUT_RATE)

            model = SRGAN(generator, discriminator)
            model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
            #print ('model compile:', model.summary())

        #model.enable_gradient_checkpointing()
        # Start/resume training
        # Train the model with the new callback
        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        #model.build(input_shape=(int(BATCH_SIZE / max(NUM_GPUS, !)), NX, NY, INPUT_CHANNELS))

        hist = model.fit(dataset_train, 
            epochs = EPOCHS, 
            callbacks = callbacks_list, 
            validation_data = dataset_valid, 
            verbose = 2, 
            initial_epoch = EPOCH_INIT,
            )

        print ('history:', type(hist.history), hist.history)
        savemat(self.wdir + f'loss_{model_name}.mat', hist.history)
        path_figure_loss = self.wdir + '/Loss_Function_EPOCHS' + str(EPOCHS) + '.png'
        postproc = postprocess.PostProcess()
        postproc.plot_lines(hist.history, path_figure_loss)


        # The latest model weights (that are considered the best) are loaded into the model.
        checkpoint_files = glob.glob(os.path.join(self.wdir, "*.weights.h5"))
        """
        if checkpoint_files:
            latest = max(checkpoint_files, key=os.path.getctime)  # Get latest modified file
            print("Manually found checkpoint:", latest)
            model.built = True
            model.load_weights(latest)
        else:
            print("No checkpoint files found in:", self.wdir)
        """
        #latest = tf.train.latest_checkpoint(os.path.dirname(self.wdir))
        latest = max(checkpoint_files, key=os.path.getctime)  # Get latest modified file
        #model.load_weights(latest)
        loss, acc = model.evaluate(dataset_train, verbose=2)
        print('loss, acc using latest weights =', loss, acc)

        # Save the model to a HDF5 file.
        generator.save(self.wdir + f'{model_name}_generator.h5')
        discriminator.save(self.wdir + f'{model_name}_discriminator.h5')

        return generator


    def prediction(self, generator, X_test, const_test, y_test, batch_size):

        """
        #tf.debugging.enable_check_numerics()
        print('X_test, const_test shape:', X_test.shape, const_test.shape)
        #generator.trainable = False
        for layer in generator.layers:
            print(f"{layer.name}: {layer.input_shape} -> {layer.output_shape}, {generator.input_shape} ")
            intermediate_model = Model(inputs=generator.input, outputs=layer.output)
            intermediate_output = intermediate_model.predict([X_test, const_test], batch_size=batch_size)
            print(f"Layer {layer.name} output shape: {intermediate_output.shape}")
        print('End of Debug')
        """
        y_pred = generator.predict([X_test, const_test], batch_size=batch_size)
        np.savez_compressed(self.wdir + 'preds', hr = y_test, hr_p = y_pred)

        return y_pred


    def get_init_epoch(self, checkpoint_path):

        ## util function to get the initial epoch number from the checkpoint name
        #filename = os.path.basename(checkpoint_path)
        #print ('filename', filename)
        #filename = os.path.splitext(filename)[0]
        #print ('filename', filename)
        #init_epoch = filename.split("_")[1]

        reader = read.Read('ascii')
        col1, col2 = reader.read_ascii(checkpoint_path)

        init_epoch = col2[0].split('_')[1].strip('"')
        print('init_epoch=', init_epoch)

        return int(init_epoch)
