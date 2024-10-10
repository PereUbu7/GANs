import tensorflow as tf
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
import numpy as np

print(tf.__version__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# hyper-parameters
ITERATION = 10000
Z_DIM = 3
BATCH_SIZE = 256
D_LR = 0.0002
G_LR = 0.0002
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

test_z = tf.random.normal([36, Z_DIM])


def get_random_z(z_dim, batch_size):
    return tf.random.normal([batch_size, z_dim])


# define discriminator
def make_discriminaor(input_shape, generator_output_shape):
    y_input = tf.keras.Input(generator_output_shape, name=f'y_Input')
    y_layer1 = tf.keras.layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.L2(0.1), name=f'y_layer_1')(y_input)
    y_layer2 = tf.keras.layers.Dense(units=20, activation='relu', kernel_regularizer=regularizers.L2(0.1), name=f'y_layer_2')(y_layer1)

    p_input = tf.keras.Input(input_shape, name=f'p_Input')
    p_layer1 = tf.keras.layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.L2(0.1), name=f'p_layer_1')(p_input)
    p_layer2 = tf.keras.layers.Dense(units=20, activation='relu', kernel_regularizer=regularizers.L2(0.1), name=f'p_layer_2')(p_layer1)

    core_input = tf.keras.layers.Add()([y_layer2, p_layer2])
    c_layer1 = tf.keras.layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.L2(0.1), name=f'c_layer_1')(core_input)
    c_layer2 = tf.keras.layers.Dense(units=1, activation='sigmoid', name=f'c_layer_2')(c_layer1)

    return tf.keras.Model(inputs=[y_input, p_input], outputs=c_layer2)


# define generator
def make_generator(input_shape):

    z_input = tf.keras.Input(Z_DIM, name=f'z_Input')
    z_layer1 = tf.keras.layers.Dense(units=10, activation='relu', name=f'z_layer_1')(z_input)
    z_layer2 = tf.keras.layers.Dense(units=20, activation='relu', name=f'z_layer_2')(z_layer1)

    p_input = tf.keras.Input(input_shape, name=f'p_Input')
    p_layer1 = tf.keras.layers.Dense(units=10, activation='relu', name=f'p_layer_1')(p_input)
    p_layer2 = tf.keras.layers.Dense(units=20, activation='relu', name=f'p_layer_2')(p_layer1)

    core_input = tf.keras.layers.Add()([z_layer2, p_layer2])
    # core_input = tf.keras.layers.Concatenate(axis=1)([z_layer2, p_layer2])
    c_layer1 = tf.keras.layers.Dense(units=20, activation='relu', name=f'c_layer_1')(core_input)
    c_layer1 = tf.keras.layers.Dense(units=10, activation='relu', name=f'c_layer_1')(core_input)
    c_layer2 = tf.keras.layers.Dense(units=1, name=f'c_layer_2')(c_layer1)

    return tf.keras.Model(inputs=[z_input, p_input], outputs=c_layer2)


# define loss function
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        # return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        return -tf.reduce_mean(tf.math.log(fake_logits + 1e-5) + tf.math.log(1. - real_logits + 1e-5))

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(tf.math.log(fake_logits/(1-fake_logits)))
        # return -tf.reduce_mean(tf.math.log(fake_logits + 1e-5))

    return d_loss_fn, g_loss_fn


# data load & preprocessing
train_x = tf.convert_to_tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
train_y = tf.convert_to_tensor([10, 9,  8,   7,   6,   5,   4,   3,   2,   1,   0, 13, 12, 11,  10,   9,   8,   4,   3,   2,   1,  0])


# generator & discriminator
G = make_generator(1)
D = make_discriminaor(1, 1)

# optimizer
g_optim = tf.keras.optimizers.Adam(G_LR)
d_optim = tf.keras.optimizers.Adam(D_LR)

# loss function
d_loss_fn, g_loss_fn = get_loss_fn()

@tf.function
def train_discriminator(x, y):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_y = G((z, x), training=False)

        fake_logits = D((fake_y, x), training=True)
        real_logits = D((y, x), training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))

    return d_loss

@tf.function
def train_generator(x, y):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_y = G((z, x), training=True)

        fake_logits = D((fake_y, x), training=False)

        g_loss = g_loss_fn(fake_logits)

    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss

@tf.function
def train_step(x, y):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_y = G((z, x), training=True)

        fake_logits = D((fake_y, x), training=True)
        real_logits = D((y, x), training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss


# training loop
def train(x_data, y_data, log_freq=20):

    fig, ax = plt.subplots(2, 2)
    ax.flat[0].set_ylim(0, 15)
    ax.flat[0].set_xlim(0, 15)
    ax.flat[1].set_ylim(0, 15)
    ax.flat[1].set_xlim(0, 15)
    ax.flat[2].set_ylim(0, 15)
    ax.flat[2].set_xlim(0, 15)
    ax.flat[3].set_ylim(0, 15)
    ax.flat[3].set_xlim(0, 15)

    # for bigEpoch in range(ITERATION):
    for epoch in range(1):
        for step in range(x_data.shape[0]):
            x = tf.slice(x_data, [step], [1])
            y = tf.slice(y_data, [step], [1])
            d_loss = train_discriminator(x, y)
            d_loss_metrics(d_loss)
        if epoch % log_freq == 0:
            template = '[{}/{}] D_loss={:.5f}'
            print(template.format(epoch, ITERATION, d_loss_metrics.result()))
            d_loss_metrics.reset_states()
    
    z = get_random_z(Z_DIM, BATCH_SIZE)
    def t(frame):
        for step in range(x_data.shape[0]):
            x = tf.slice(x_data, [step], [1])
            y = tf.slice(y_data, [step], [1])
            g_loss, d_loss = train_step(x, y)
            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
        if frame % log_freq == 0:
            template = '[{}/{}] G_loss={:.5f} D_loss={:.5f}'
            print(template.format(frame, ITERATION, g_loss_metrics.result(), d_loss_metrics.result()))
            g_loss_metrics.reset_states()

            dist1 = G((z, tf.convert_to_tensor([0.1]))).numpy()
            dist2 = G((z, tf.convert_to_tensor([0.3]))).numpy()
            dist3 = G((z, tf.convert_to_tensor([0.6]))).numpy()
            dist4 = G((z, tf.convert_to_tensor([0.9]))).numpy()
            # Clear the previous histogram
            ax.flat[0].clear()
            ax.flat[1].clear()
            ax.flat[2].clear()
            ax.flat[3].clear()
            # Replot the histogram
            ax.flat[0].hist(dist1, bins=20)
            ax.flat[1].hist(dist2, bins=20)
            ax.flat[2].hist(dist3, bins=20)
            ax.flat[3].hist(dist4, bins=20)
            # Set limits again after clearing
            ax.flat[0].set_ylim(0, 20)
            ax.flat[0].set_xlim(0, 20)
            ax.flat[1].set_ylim(0, 20)
            ax.flat[1].set_xlim(0, 20)
            ax.flat[2].set_ylim(0, 20)
            ax.flat[2].set_xlim(0, 20)
            ax.flat[3].set_ylim(0, 20)
            ax.flat[3].set_xlim(0, 20)

    ani = FuncAnimation(fig, t, frames=np.arange(0, ITERATION), interval=1)

    plt.show()


if __name__ == "__main__":

    train(train_x, train_y)

    z = get_random_z(Z_DIM, BATCH_SIZE)

    dist = G((z, tf.convert_to_tensor([0.3]))).numpy()

    print(dist)

    plt.subplot(2, 5, 1)
    plt.hist(G((z, tf.convert_to_tensor([0.0]))).numpy())

    plt.subplot(2, 5, 2)
    plt.hist(G((z, tf.convert_to_tensor([0.1]))).numpy())

    plt.subplot(2, 5, 3)
    plt.hist(G((z, tf.convert_to_tensor([0.2]))).numpy())

    plt.subplot(2, 5, 4)
    plt.hist(G((z, tf.convert_to_tensor([0.3]))).numpy())

    plt.subplot(2, 5, 5)
    plt.hist(G((z, tf.convert_to_tensor([0.4]))).numpy())

    plt.subplot(2, 5, 6)
    plt.hist(G((z, tf.convert_to_tensor([0.5]))).numpy())

    plt.subplot(2, 5, 7)
    plt.hist(G((z, tf.convert_to_tensor([0.6]))).numpy())

    plt.subplot(2, 5, 8)
    plt.hist(G((z, tf.convert_to_tensor([0.7]))).numpy())

    plt.subplot(2, 5, 9)
    plt.hist(G((z, tf.convert_to_tensor([0.8]))).numpy())

    plt.subplot(2, 5, 10)
    plt.hist(G((z, tf.convert_to_tensor([0.9]))).numpy())

    plt.show() 