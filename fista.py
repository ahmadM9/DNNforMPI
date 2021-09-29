__author__ = "Ahmad Mohammad"

import numpy as np
from numpy import linalg as LA
from tensorflow import keras


def calculate_lip_constant(A):
    # times 2 ??
    return 2 * max(LA.eigvals(np.dot(A.transpose(), A)))


def scaled_denoiser(u, scaler):
    model = keras.models.load_model('./Trained_Models/model.h5')
    u = u.reshape(40, 40, 1, 1)
    u = scaler * u
    out = (1.0 / scaler) * model.predict(u)
    out = out.reshape(1600,)
    return out


def prox(u, alpha_):
    return np.maximum(np.abs(u) - alpha_, 0.) * np.sign(u)


def compute_gradient(A, b, y):
    return np.dot(np.dot(A.transpose(), A), y) - np.dot(A.transpose(), b)


class Fista:

    """
    Fast iterative shrinking/thresholding algorithm
    """

    def __init__(self, lambda_=.5, iterations=5000):
        self.lambda_ = lambda_
        self.iterations = iterations

    def fista(self, A, b, scaler, network, lip_constant=None):

        (n_samples, n_features) = A.shape
        n_kernels = int(n_features/n_samples)
        self.n_samples, self.n_kernels = n_samples, n_kernels

        if lip_constant is None:
            lip_constant = calculate_lip_constant(A)

        lipinv = 1 / lip_constant
        lambdaLipinv = self.lambda_ * lipinv

        # Initializing the coefficients (Step 0)
        x_current = np.zeros(n_features, dtype=np.float)
        x_next = np.zeros(n_features, dtype=np.float)
        y_current = np.copy(x_next)
        y_next = np.zeros(n_features, dtype=np.float)
        tau_old = 1.0
        tau_new = 0.0

        # A loop for the algorithm
        for i in range(self.iterations):

            # Compute the gradient
            tmp_grad = compute_gradient(A, b, y_current)
            grad = y_current - lipinv * tmp_grad

            # Apply projection
            if network is True:
                x_next = scaled_denoiser(grad, scaler)
            else:
                x_next = prox(grad, lambdaLipinv)

            tau_new = (1.0 + np.sqrt(1.0 + 4.0 * tau_old**2)) / 2.0

            y_next = x_next + (tau_old - 1) / tau_new * (x_next - x_current)

            # Update
            x_current = x_next
            tau_old = tau_new
            y_current = y_next

        return x_next
