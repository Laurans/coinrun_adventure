import numpy as np
import tensorflow as tf
from loguru import logger

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
        )

    return layer


def fc(scope, *, input_shape, units, init_scale=1.0, init_bias=0.0):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Dense(
            units=units,
            kernel_initializer=ortho_init(init_scale),
            bias_initializer=tf.keras.initializers.Constant(init_bias),
        )
        layer.build(input_shape)

    return layer


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator
    Returns a function that builds fully connected network with a given input / placeholder
    """

    def network_fn(input_shape):
        logger.info(f"input shape is {input_shape}")
        x_input = tf.keras.Input(shape=input_shape)

        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(
                units=num_hidden,
                kernel_initializer=ortho_init(np.sqrt(2)),
                name=f"mlp_fc{i}",
                activation=activation,
            )(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network

    return network_fn


@register("nature-cnn")
def nature_cnn():
    def network_fn(input_shape):
        """
        CNN from Nature paper.
        """
        logger.info(f"input shape is {input_shape}")
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = x_input
        h = tf.cast(h, tf.float32) / 255.0
        h = conv(
            "c1", nf=32, ks=8, strides=4, activation="relu", init_scale=np.sqrt(2)
        )(h)
        h2 = conv(
            "c2", nf=64, ks=4, strides=2, activation="relu", init_scale=np.sqrt(2)
        )(h)
        h3 = conv(
            "c3", nf=64, ks=3, strides=1, activation="relu", init_scale=np.sqrt(2)
        )(h2)
        h3 = tf.keras.layers.Flatten()(h3)
        h3 = tf.keras.layers.Dense(
            units=512,
            kernel_initializer=ortho_init(np.sqrt(2)),
            name="fc1",
            activation="relu",
        )(h3)
        network = tf.keras.Model(inputs=[x_input], outputs=[h3])
        return network

    return network_fn


def get_network_builder(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown network type: {}".format(name))
