# from tensorflow.python.keras import regularizers, Input, Model
from tensorflow import Tensor
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, BatchNormalization, \
    Activation, SeparableConv2D, Add, GlobalAveragePooling2D, ReLU, PReLU
from tensorflow.python.keras.layers import AveragePooling2D
from icecream import ic

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
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation=activation_fct))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL
    model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    model.add(BatchNormalization())
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
    x = Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fct)(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fct)(x)

    previous_block_activation = x

    for size in [256, 256, 728]:
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

def create_res_net(model, input_size, n_classes, activation_fct='relu', res_net_version='resnet_v'):
    inputs = Input(shape=input_size)
    num_filters = 32

    t = BatchNormalization()(inputs)
    t = Conv2D(
            # kernel_size=7,
            kernel_size=3,
            strides=1,
            filters=8,
            padding="same",
            name="conv1")(t)
    t = relu_bn(t, activation_fct)
    ic(t)
    # t = MaxPooling2D(3, strides=2, padding='same')(t)

    num_blocks_list = RESNET_VERSION[res_net_version]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        # n_filters = num_filters * (i + 1)
        for j in range(num_blocks):
            strides = 1 if i == 0 else 2
            t = residual_block(t, downsample=(j == 0), filters=num_filters, name="conv" + str(i+2) + '_' + str(j+1), 
                               activation_fct=activation_fct, s=strides)
        ic(t)
        num_filters *= 2

    t = AveragePooling2D(2)(t)
    t = Flatten()(t)

    # first (and only) Full Connection
    # model.add(Dense(units=1000, activation=activation_fct))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    t = Dense(units=512, activation=activation_fct)(t)
    # t = BatchNormalization()(t)
    # t = Dropout(0.5)(t)

    outputs = Dense(units=n_classes, activation='softmax')(t)

    model = Model(inputs, outputs)

    return model


def relu_bn(inputs: Tensor, activation_fct) -> Tensor:
    if activation_fct == 'relu':
        relu = ReLU()(inputs)
    else:
        relu = PReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(input_tensor: Tensor, downsample: bool, filters: int, name="conv", activation_fct='relu', s=2) -> Tensor:
    x = Conv2D(kernel_size=1,
               strides= (1 if not downsample else s),
               filters=filters,
               padding="same",
               name=name + 'a')(input_tensor)
    x = relu_bn(x, activation_fct)

    x = Conv2D(kernel_size=3,
               strides=1,
               filters=filters,
               padding="same",
               name=name + 'b')(x)
    x = relu_bn(x, activation_fct)

    x = Conv2D(kernel_size=1,
            strides=1,
            filters=filters*4,
            padding="same",
            name=name + 'c')(x)
    x = BatchNormalization()(x)

    if downsample:
        input_tensor = Conv2D(kernel_size=1,
                   strides=s,24x24
                   filters=filters*4,
                   padding="same",
                   name=name)(input_tensor)

    out = Add()([input_tensor, x])
    out = relu_bn(out, activation_fct)
    return out


MODELS = {
    "model1": model1,
    "vgg_net": model_vgg_net,
    "xception": x_ception,
    "resnet_v": create_res_net,
    "resnet_v2_50": create_res_net,
    "resnet_v2_101": create_res_net,
    "resnet_v2_152": create_res_net,
}

class Models:
    model1 = "model1"
    vgg_net = "vgg_net"
    xception = "xception"
    res_net = "resnet_v"
    res_net_v50 = "resnet_v2_50"
    res_net_v101 = "resnet_v2_101"
    res_net_v152 = "resnet_v2_152"


RESNET_VERSION = {
    'resnet_v': [2, 5, 5, 2],
    'resnet_v2_50': [3, 4, 6, 3], 
    'resnet_v2_101': [3, 4, 23, 3],
    'resnet_v2_152': [3, 8, 36, 3]
}
