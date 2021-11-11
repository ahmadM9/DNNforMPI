import tensorflow as tf


def denoise_network():

    inpt = tf.keras.Input(shape=(None, None, 1))

    x = tf.keras.layers.Conv2D(8, 3, padding='same')(inpt)
    x = tf.keras.activations.relu(x)

    for i in range(3):
        x = tf.keras.layers.Conv2D(8, 3, padding='same')(x)
        x = tf.keras.activations.relu(x)

    x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    x = tf.keras.layers.Subtract()([inpt, x])
    x = tf.keras.activations.relu(x)

    model = tf.keras.Model(inputs=inpt, outputs=x)

    return model

