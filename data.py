import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split


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


def split_data(dataset):
    train_data, test_data = train_test_split(dataset, test_size=.2)
    test_data, val_data = train_test_split(test_data, test_size=.5)

    return train_data, val_data, test_data


def get_image(index):
    imgs = scipy.io.loadmat('./Datasets/Vessel40x40.mat')
    imgs_all = imgs.get('images_all')
    img = imgs_all[index]

    return img


class Data:

    def __init__(self):
        self.freq = scipy.io.loadmat('./Datasets/FreqVoltages40x40.mat')
        self.sysmat = scipy.io.loadmat('./Datasets/Vessel_System2.mat')
        self.sys_matrix = get_system_matrix(self.sysmat)
        self.coil_ges = get_voltage_signals(self.freq)
        self.imgs = scipy.io.loadmat('./Datasets/Vessel40x40.mat')
        self.imgs_all = self.imgs.get('images_all')
        self.train_data, self.val_data, self.test_data = split_data(self.imgs_all)

