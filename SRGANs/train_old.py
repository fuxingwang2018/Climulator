def training_old(self, batch_size, EPOCHS, dataset_train, dataset_valid):

        model_name = f"model_1"

        generator_optimizer = optimizers.Adam(1e-4)
        discriminator_optimizer = optimizers.Adam(1e-4)

        subsampling_lr = 4
        n_res_block = 8
        input_channels = 1
        output_channels = 1

        nx = 104
        nz = 88

        generator =  model_generator(nx, nz, input_channels, subsampling_lr, n_res_block, batch_size)
        discriminator = model_discriminator(nx, nz, output_channels, batch_size)

        model = SRGAN(generator, discriminator)
        model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)

        checkpoint_filepath = self.wdir + 'checkpoint_NN'

        # Create a callback that saves the model's weights
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_gen_loss',
            mode='min',
            save_best_only=True)

        callbacks_list = [checkpoint]

        # Train the model with the new callback
        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        hist = model.fit(dataset_train, epochs = EPOCHS, callbacks = callbacks_list, validation_data = dataset_valid, verbose = 1)
        savemat(self.wdir + f'loss_{model_name}.mat', hist.history)
        print ('history:', hist.history)

        # The model weights (that are considered the best) are loaded into the model.
        model.load_weights(checkpoint_filepath)

        # Save the model to a HDF5 file.
        generator.save(self.wdir + f'{model_name}_generator.h5')
        discriminator.save(self.wdir + f'{model_name}_discriminator.h5')

        return generator

