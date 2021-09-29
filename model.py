import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import scipy.io
import argparse
import os
import shutil
from skimage.measure import compare_ssim
import pandas as pd
import math
from tensorflow_core.python.keras.callbacks import LearningRateScheduler

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=.0001, help='initial learning rate')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--sigma', default=25, type=int, help='noise_level')
parser.add_argument('--test_model', default=False, type=bool, help='Only test')
args = parser.parse_args()


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


def denoise_network():

    inpt = tf.keras.Input(shape=(None, None, 1))

    x = tf.keras.layers.Conv2D(8, 3, padding='same')(inpt)
    x = tf.keras.activations.relu(x)

    for i in range(3):
        x = tf.keras.layers.Conv2D(8, 3, padding='same')(x)
        x = tf.keras.activations.relu(x)

    x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    x = tf.keras.layers.Subtract()([inpt, x])

    model = tf.keras.Model(inputs=inpt, outputs=x)

    return model


# Step decay schedule drops the learning rate by half every 10 epochs
def step_decay(epoch):
    initial_lr = args.lr
    drop = .5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch)/epochs_drop))

    return lr


def train_datagen(y, batch_size=8):
    indices = list(range(y.shape[0]))
    while(True):
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y[indices[i:i+batch_size]]
            # Adding noise
            noise = np.random.normal(0, args.sigma/255.0, ge_batch_y.shape)
            ge_batch_x = ge_batch_y + noise
            yield ge_batch_x, ge_batch_y


def load_data():
    imgs = scipy.io.loadmat('./Datasets/Vessel40x40.mat')
    imgs_all = imgs.get('images_all')
    train_data, test_data = train_test_split(imgs_all, test_size=.2)

    return train_data, test_data


def train():

    # Create the Model
    model = denoise_network()

    # Compiling the model
    model.compile(optimizer='adam', loss=['mse'], metrics=['acc'])

    # Using callback
    loss_history = LossHistory()
    lr = LearningRateScheduler(step_decay)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # Training the model
    history = model.fit(train_datagen(train_data, batch_size=args.batch_size),
                        steps_per_epoch=len(train_data)//args.batch_size, epochs=args.epoch,
                        callbacks=[loss_history, lr, early_stopping])

    model.save('./Trained_Models/model.h5')

    return model


# Computing the peak signal-to-noise ratio between images
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def test(model):
    # Creating the directories
    os.mkdir(test_dir)
    os.mkdir(clean_test)
    os.mkdir(noisy_test)
    os.mkdir(output_test)

    index = 1
    psnr = []
    ssim = []
    name = []

    for img in test_data:
        img_clean = img.astype('float32') / 255.0
        img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)
        img_test = img_test.astype('float32')
        # predict
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1)
        y_predict = model.predict(x_test)
        # calculate numeric metrics
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        psnr_noise, psnr_denoised = PSNR(img_clean, img_test), PSNR(img_clean, img_out)
        ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        # save images
        filename = "img" + '{}'.format(index)
        name.append(filename)
        img_clean = Image.fromarray((img_clean * 255).astype('uint8'))
        img_clean.save(clean_test + filename + '.png')
        img_test = Image.fromarray((img_test * 255).astype('uint8'))
        img_test.save(noisy_test + filename + '_sigma' + '{}_psnr{:.2f}.png'.format(args.sigma, psnr_noise))
        img_out = Image.fromarray((img_out * 255).astype('uint8'))
        img_out.save(output_test + filename + '_psnr{:.2f}.png'.format(psnr_denoised))
        index += 1

    psnr_avg = sum(psnr) / len(psnr)
    ssim_avg = sum(ssim) / len(ssim)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))

    pd.DataFrame({'name': np.array(name), 'psnr': np.array(psnr), 'ssim': np.array(ssim)}).to_csv(
        res_report_dir + '/metrics.csv', index=True)


if __name__ == '__main__':

    # Defining the directories
    images_dir = './images/'
    test_dir = images_dir + 'test/'
    clean_test = test_dir + 'clean/'
    noisy_test = test_dir + 'noisy/'
    output_test = test_dir + 'output/'
    res_report_dir = './report'

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    else:
        shutil.rmtree(images_dir)
        os.mkdir(images_dir)

    # Load the data
    train_data, test_data = load_data()
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
    train_data = train_data.astype('float32') / 255.0

    if args.test_model:
        if not os.path.exists('./Trained_Models'):
            print("No model to test!")
        else:
            model = tf.keras.models.load_model('./Trained_Models/model.h5')
            test(model)
    else:
        if not os.path.exists('./Trained_Models'):
            os.mkdir('./Trained_Models')
        else:
            shutil.rmtree('./Trained_Models')
            os.mkdir('./Trained_Models')
        model = train()
        test(model)
