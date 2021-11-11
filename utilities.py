import numpy as np
from numpy import linalg as LA
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from scipy.linalg import norm
import math


def calculate_lip_constant(A):
    # return (max(LA.eigvals(np.dot(A.transpose(), A))) ** 2) * 1.5
    # return 2 * max(LA.eigvals(np.dot(A.transpose(), A)))
    return norm(np.dot(A, A.transpose()), 2)


def scaled_denoiser(u, scaler, path):
    model = keras.models.load_model(path)
    u = u.reshape(40, 40, 1, 1)
    u = scaler * u
    out = (1.0 / scaler) * model.predict(u)
    out = out.reshape(1600, )
    return out


def prox(u, alpha_):
    return np.maximum(np.maximum(np.abs(u) - alpha_, 0.) * np.sign(u), 0.)


def compute_gradient(A, b, y):
    return np.dot(np.dot(A.transpose(), A), y) - np.dot(A.transpose(), b)


def compute_function_error(original_img, reconstructed_img):
    reconstructed_img = reconstructed_img.reshape(40, 40)

    mse = np.sum((original_img.astype('float') - reconstructed_img.astype('float')) ** 2)
    mse /= float(original_img.shape[0] * reconstructed_img.shape[1])

    return mean_squared_error(original_img.astype('float'), reconstructed_img.astype('float'))


# Computing the peak signal-to-noise ratio between images
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Computing the signal to noise ratio
def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# Step decay schedule drops the learning rate by half every 10 epochs
def step_decay(epoch):
    initial_lr = .0001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lr)
    return lr


def datagen(y, sigma, batch_size=8):
    indices = list(range(y.shape[0]))
    while (True):
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y[indices[i:i + batch_size]]
            # Adding noise
            noise = np.random.normal(0, sigma / 255.0, ge_batch_y.shape)
            ge_batch_x = ge_batch_y + noise
            yield ge_batch_x, ge_batch_y
