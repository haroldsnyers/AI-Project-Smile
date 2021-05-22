# from tensorflow.python.keras import regularizers, Input, Model
from tensorflow import Tensor
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, SeparableConv2D, Add, GlobalAveragePooling2D, ReLU, PReLU
from tensorflow.python.keras.layers import AveragePooling2D

class ActivationFunction:
    relu = 'relu'
    prelu = 'prelu'


def model1(model, input_size, n_classes, activation_fct='relu'):
    if activation_fct == ActivationFunction.relu:
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
    elif activation_fct == ActivationFunction.prelu:
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=input_size))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # (CONV-128 => PReLumodel.add(Activation(PReLU) * 2  => (CONV-256 => PReLumodel.add(Activation(PReLU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(Conv2D(128, (3, 3)))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(Conv2D(256, (3, 3)))
        model.add(PReLU(alpha_initializer='zeros'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # Flattening
        model.add(Flatten())

        # Full Connection
        # one dimensional vector
        # model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(units=1024))
        model.add(Dropout(0.5))

        # Output Layer
        # the activation function softmax makes sure to rescale the final values between zero and one and
        # that the sum of the values of all out layer neurons is equal to just 1
        model.add(Dense(units=n_classes, activation='softmax'))
        model.add(PReLU(alpha_initializer='zeros'))
        return model


def model_vgg_net(model, input_size, n_classes, activation_fct="relu"):

    # CONV => RELU => POOL
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3), activation=activation_fct, padding='same', input_shape=input_size))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation_fct))
    # model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), activation=activation_fct, padding='same'))
    # model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation=activation_fct))
    # model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # first (and only) Full Connection
    model.add(Dense(units=4096, activation=activation_fct))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # first (and only) Full Connection
    model.add(Dense(units=4096, activation=activation_fct))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # first (and only) Full Connection
    model.add(Dense(units=1000, activation=activation_fct))
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
        activation_fct=activation_fct)

    outputs = Dense(units=n_classes, activation='softmax')(outputs)

    xception = Model(inputs, outputs)

    return xception


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

        x = MaxPooling2D(2, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = Add()([x, residual])
        previous_block_activation = x

    return x


def middle_flow(x, activation_fct, num_blocks=4):
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


def exit_flow(x, activation_fct):
    previous_block_activation = x

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(2, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = Add()([x, residual])

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation(activation_fct)(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    return x


def create_res_net(model, input_size, n_classes, activation_fct='relu'):
    inputs = Input(shape=input_size)
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(units=n_classes, activation='softmax')(t)

    model = Model(inputs, outputs)

    return model


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


MODELS = {
    "model1": model1,
    "vgg_net": model_vgg_net,
    "xception": x_ception,
    "res_net": create_res_net
}

class Models:
    model1 = "model1"
    vgg_net = "vgg_net"
    xception = "xception"
    res_net = "res_net"
