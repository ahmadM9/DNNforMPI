__author__ = "Ahmad Mohammad"

import numpy as np
from numpy import linalg as LA
import scipy.io
import matplotlib.pyplot as plt


def calculate_lip_constant(A):
    # times 2 ??
    return 2 * max(LA.eigvals(np.dot(A.transpose(), A)))


def prox(u, alpha_):
    return np.maximum(np.abs(u) - alpha_, 0.) * np.sign(u)


def compute_gradient(A, b, y):
    return np.dot(np.dot(A.transpose(), A), y) - np.dot(A.transpose(), b)


class Fista:

    """
    Fast iterative shrinking/thresholding algorithm
    """

    def __init__(self, lambda_=.5, iterations=1000):
        self.lambda_ = lambda_
        self.iterations = iterations

    def fista(self, A, b, lip_constant=None):

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
        tau_old = 1
        tau_new = 0

        # A loop for the algorithm
        for i in range(self.iterations):

            # Compute the gradient
            tmp_grad = compute_gradient(A, b, y_current)
            grad = y_current - lipinv * tmp_grad

            # Apply projection
            x_next = prox(grad, lambdaLipinv)

            tau_new = (1 + np.sqrt(1 + 4 * tau_old**2)) / 2

            y_next = x_next + (tau_old - 1) / tau_new * (x_next - x_current)

            # Update
            x_current = x_next
            tau_old = tau_new
            y_current = y_next

        return x_next


def get_voltage_signals():
    freq = scipy.io.loadmat('./Datasets/FreqVoltages40x40.mat')

    xcoil = freq.get('f1')
    ycoil = freq.get('f2')

    return xcoil, ycoil


def get_system_matrices():
    sysmat = scipy.io.loadmat('./Datasets/Vessel_System2.mat')

    Af = sysmat.get('Af')
    Bf = sysmat.get('Bf')

    return Af, Bf


def preprocess(xcoil, ycoil, af, bf):
    af = af.reshape(817, -1)
    af = np.concatenate((af.real, af.imag), axis=0)
    bf = bf.reshape(817, -1)
    bf = np.concatenate((bf.real, bf.imag), axis=0)

    sys_ges = np.concatenate((af, bf), axis=0)

    x = xcoil[10, :]
    y = ycoil[10, :]

    x = np.concatenate((x.real, x.imag))
    y = np.concatenate((y.real, y.imag))

    coil_ges = np.concatenate((x, y))

    return sys_ges, coil_ges


def get_image():
    imgs = scipy.io.loadmat('./Datasets/Vessel40x40.mat')
    imgs_all = imgs.get('images_all')
    img = imgs_all[10]

    return img


def plot_results(x, img):

    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.title('Reconstructed Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('Original Image')

    plt.show()


if __name__ == "__main__":
    xcoil, ycoil = get_voltage_signals()
    Af, Bf = get_system_matrices()

    A, b = preprocess(xcoil, ycoil, Af, Bf)

    fista = Fista()
    x = fista.fista(A, b)
    x = x.reshape(40, 40)

    img = get_image()

    plot_results(x, img)

    # Calculate the difference
    print(np.max(np.abs(img - x)))
