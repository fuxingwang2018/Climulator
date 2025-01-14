from tensorflow.keras.layers import Conv2D

def const_upscale_block(const_input, steps, filters):
    # Map (N x kH x kW x C) to (N x H x W x f), where k is downscaling factor
    const_output = Conv2D(filters=filters, kernel_size=(step, step), strides=step, padding="valid", activation="relu")(const_input)
    conv_1 = layers.Conv2D(filters=64, kernel_size=7, strides=1, activation='linear', padding='same')(inputs)
    return const_output

