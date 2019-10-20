import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from coinrun_adventure.config import ExpConfig

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)

        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[: shape[0], : shape[1]]).astype(np.float32)

    return _ortho_init


def conv(
    scope,
    *,
    nf,
    ks,
    strides,
    activation,
    padding="valid",
    init_scale=1.0,
    data_format="channels_last",
):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Conv2D(
            filters=nf,
            kernel_size=ks,
            strides=strides,
            padding=padding,
            data_format=data_format,
            kernel_initializer=ortho_init(init_scale),
            activation=activation,
        )

    return layer


@register("fc")
def fc(scope, *, x_input, units, init_scale=1.0, init_bias=0.0):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Dense(
            units=units,
            kernel_initializer=ortho_init(init_scale),
            bias_initializer=tf.keras.initializers.Constant(init_bias),
        )(x_input)

    return layer


@register("nature")
def nature_cnn(x_input):
    """
    CNN from Nature paper.
    """
    h = x_input
    h = tf.cast(h, tf.float32) / 255.0
    h = conv("c1", nf=32, ks=8, strides=4, activation="relu", init_scale=np.sqrt(2))(h)
    h2 = conv("c2", nf=64, ks=4, strides=2, activation="relu", init_scale=np.sqrt(2))(h)
    h3 = conv("c3", nf=64, ks=3, strides=1, activation="relu", init_scale=np.sqrt(2))(
        h2
    )
    h3 = layers.Flatten()(h3)
    h3 = layers.Dense(
        units=512,
        kernel_initializer=ortho_init(np.sqrt(2)),
        name="fc1",
        activation="relu",
    )(h3)

    return h3


@register("impala")
def impala_cnn(x_input, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    use_batch_norm = ExpConfig.USE_BATCH_NORM
    dropout_ratio = ExpConfig.DROPOUT

    def conv_layer(out, depth):
        out = layers.Conv2D(
            filters=depth,
            kernel_size=3,
            padding="same",
            data_format="channels_last",
            activation=None,
        )(out)
        if dropout_ratio > 0:
            out = layers.Dropout(dropout_ratio)(out)

        if use_batch_norm:
            out = layers.BatchNormalization(center=True, scale=True)(out)

        return out

    def residual_block(inputs):
        depth = inputs.shape[-1]

        out = layers.ReLU()(inputs)
        out = conv_layer(out, depth)
        out = layers.ReLU()(out)
        out = conv_layer(out, depth)
        return layers.Add()([out, inputs])

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(out)
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = x_input
    out = tf.cast(out, tf.float32) / 255.0
    for depth in depths:
        out = conv_sequence(out, depth)

    out = layers.Flatten()(out)
    out = layers.ReLU()(out)
    out = layers.Dense(units=256, activation="relu")(out)

    return out


@register("impalalarge")
def impala_cnn_large(x_input, depths=[16, 32, 32]):
    return impala_cnn(x_input, depths=[32, 64, 64, 64, 64])


def get_network_builder(name):
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown network type: {}".format(name))
