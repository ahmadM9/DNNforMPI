import scipy.io
import numpy as np
from fista import Fista
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--nn', dest='nn', type=bool, default=False, help='Reconstructing using DNN')
args = parser.parse_args()


def get_voltage_signals(freq_voltages, index=0):
    xcoil = freq_voltages.get('f1')
    ycoil = freq_voltages.get('f2')

    x = xcoil[index, :]
    y = ycoil[index, :]

    x = np.concatenate((x.real, x.imag))
    y = np.concatenate((y.real, y.imag))

    coil_ges = np.concatenate((x, y))

    return coil_ges


def get_system_matrix(sysmat):

    af = sysmat.get('Af')
    bf = sysmat.get('Bf')

    af = af.reshape(817, -1)
    af = np.concatenate((af.real, af.imag), axis=0)
    bf = bf.reshape(817, -1)
    bf = np.concatenate((bf.real, bf.imag), axis=0)

    sys_ges = np.concatenate((af, bf), axis=0)

    return sys_ges


def get_image(index):
    imgs = scipy.io.loadmat('./Datasets/Vessel40x40.mat')
    imgs_all = imgs.get('images_all')
    img = imgs_all[index]

    return img


if __name__ == '__main__':

    # Reconstruct the image
    freq = scipy.io.loadmat('./Datasets/FreqVoltages40x40.mat')
    sysmat = scipy.io.loadmat('./Datasets/Vessel_System2.mat')

    sys_matrix = get_system_matrix(sysmat)
    coil_ges = get_voltage_signals(freq)
    fista = Fista()

    if args.nn is True:
        x = fista.fista(sys_matrix, coil_ges, scaler=1.0, network=True)
    else:
        x = fista.fista(sys_matrix, coil_ges, scaler=.2, network=False)

    x = x.reshape(40, 40)
    print(np.sum(np.isnan(x)))

    img = get_image(0)

    mse = np.sum((img.astype('float') - x.astype('float')) ** 2)
    mse /= float(img.shape[0] * x.shape[1])

    f, axarr = plt.subplots(1, 2)

    axarr[0].imshow(img)
    axarr[0].set_title('Original')

    axarr[1].imshow(x)
    axarr[1].set_title('Reconstructed, MSE: {:.2f}'.format(mse))

    if args.nn is True:
        plt.savefig('DNN.png')
    else:
        plt.savefig('fista.png')

    plt.show()







