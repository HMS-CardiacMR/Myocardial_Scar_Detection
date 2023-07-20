from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv3D, ReLU, BatchNormalization, \
    Add, AveragePooling3D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

input_shape_3d = (64, 64, 200, 1)

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv3D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv3D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv3D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net(input_shape_3d):
    inputs = Input(shape=(input_shape_3d))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv3D(kernel_size=3,
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

    t = AveragePooling3D(4)(t)
    t = Flatten()(t)
    outputs = Dense(1, activation='sigmoid')(t)

    model = Model(inputs, outputs)

    opt = tf.keras.optimizers.SGD(learning_rate=0.001, decay=0.0001)
    model.compile(optimizer=opt, loss="BinaryCrossentropy", metrics="binary_accuracy")

    return model