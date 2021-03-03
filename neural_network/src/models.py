from tensorflow.python.keras import regularizers, Input, Model
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, SeparableConv2D, Add, GlobalAveragePooling2D


def model1(model, input_size, n_classes, activation_fct='relu'):
    # first convolution layer
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3), activation=activation_fct, padding='same', input_shape=input_size))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation_fct))
    model.add(BatchNormalization())

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # Second Convolution layer
    model.add(Conv2D(128, (3, 3), activation=activation_fct, padding='same'))
    model.add(Conv2D(128, (3, 3), activation=activation_fct))
    model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    model.add(Conv2D(256, (3, 3), activation=activation_fct))

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Full Connection
    # one dimensional vector
    # model.add(Dense(1024, activation=activation_fct, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=1024, activation=activation_fct))
    model.add(Dropout(0.5))

    # Output Layer
    # the activation function softmax makes sure to rescale the final values between zero and one and
    # that the sum of the values of all out layer neurons is equal to just 1
    model.add(Dense(units=n_classes, activation='softmax'))
    return model


def model_vgg_net(model, input_size, n_classes, activation_fct="relu"):

    # CONV => RELU => POOL
    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), activation=activation_fct, padding='same', input_shape=input_size))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation_fct))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # first (and only) Full Connection
    model.add(Dense(units=1024, activation=activation_fct))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(units=n_classes, activation='softmax'))
    return model


def x_ception(model, input_size, n_classes, activation_fct="relu", number_blocks=8):
    inputs = Input(shape=input_size)
    outputs = exit_flow(
        middle_flow(
            entry_flow(inputs, activation_fct=activation_fct),
            activation_fct=activation_fct,
            num_blocks=number_blocks),
        n_classes=n_classes,
        activation_fct=activation_fct)
    xception = Model(inputs, outputs)

    return xception


MODELS = {
    "model1": model1,
    "vgg_net": model_vgg_net,
    "xception": x_ception
}


def entry_flow(inputs, activation_fct) :
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fct)(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fct)(x)

    previous_block_activation = x

    for size in [128, 256, 728]:
        x = Activation(activation_fct)(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation(activation_fct)(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = Add()([x, residual])
        previous_block_activation = x

    return x


def middle_flow(x, activation_fct, num_blocks=8):
    previous_block_activation = x

    for _ in range(num_blocks):
        x = Activation(activation_fct)(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation(activation_fct)(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation(activation_fct)(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, previous_block_activation])
        previous_block_activation = x

    return x


def exit_flow(x, n_classes, activation_fct):
    previous_block_activation = x

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = Add()([x, residual])

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=n_classes, activation='linear')(x)

    return x
