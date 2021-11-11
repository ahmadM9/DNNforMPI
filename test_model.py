import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
from skimage.measure import compare_ssim
from utitlities import PSNR


class Test_model:

    def __init__(self, testing_set, model, sigma):
        self.testing_set = testing_set
        self.model = model
        self.sigma = sigma

    def test(self):

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

        # Creating the directories
        os.mkdir(test_dir)
        os.mkdir(clean_test)
        os.mkdir(noisy_test)
        os.mkdir(output_test)

        index = 1
        psnr = []
        ssim = []
        name = []

        for img in self.testing_set:
            img_clean = img.astype('float32') / 255.0
            img_test = img_clean + np.random.normal(0, self.sigma / 255.0, img_clean.shape)
            img_test = img_test.astype('float32')
            # predict
            x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1)
            y_predict = self.model.predict(x_test)
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
            img_test.save(noisy_test + filename + '_sigma' + '{}_psnr{:.2f}.png'.format(self.sigma, psnr_noise))
            img_out = Image.fromarray((img_out * 255).astype('uint8'))
            img_out.save(output_test + filename + '_psnr{:.2f}.png'.format(psnr_denoised))
            index += 1

        psnr_avg = sum(psnr) / len(psnr)
        ssim_avg = sum(ssim) / len(ssim)
        name.append('Average')
        psnr.append(format(psnr_avg, ".3f"))
        ssim.append(format(ssim_avg, ".3f"))
        print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))

        pd.DataFrame({'name': np.array(name), 'psnr': np.array(psnr), 'ssim': np.array(ssim)}).to_csv(
            res_report_dir + '/metrics.csv', index=True)






