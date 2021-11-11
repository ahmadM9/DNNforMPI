import os
from DcNN_fista import DcNN_Fista
from data import Data, get_image
from fista import Fista
import argparse
from sklearn.metrics import mean_squared_error
import shutil
import tensorflow as tf
from plotter import Plotter
from test_model import Test_model
from train_model import Train_model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_model', dest='train', type=bool, default=False, help='train the model')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=.0001, help='initial learning rate')
parser.add_argument('--sigma', default=25, type=int, help='noise_level')

parser.add_argument('--test_model', dest='test', type=bool, default=False, help='test the model')
parser.add_argument('--path_to_model', default=None, type=str, help='Path to the trained model')

parser.add_argument('--nn', dest='nn', type=bool, default=False, help='Reconstructing using DNN')
parser.add_argument('--scalar', dest='scalar', type=float, default=1.0, help='Denoisor scalar')

parser.add_argument('--lambda_', dest='lambda_', type=float, default=.0001, help='Regularization term that controls '
                                                                                 'the norm of the solution')
parser.add_argument('--iterations', dest='iterations', type=int, default=5000, help='Number of iterations')
parser.add_argument('--lip_constant', dest='lip_constant', type=float, default=None, help='Lipschitz constant')

parser.add_argument('--clean', dest='clean', type=bool, default=False, help='Cleaning the directories')

args = parser.parse_args()

if __name__ == '__main__':

    # Cleaning directories
    if args.clean is True:
        shutil.rmtree('./images')
        shutil.rmtree('./report')
        shutil.rmtree('./results')
        shutil.rmtree('./Trained_Models')

    # Initialize the data
    data = Data()

    original_img = get_image(0)

    if args.train is True:
        train_model = Train_model(data.train_data, data.val_data, args.batch_size, args.epoch, args.sigma, args.lr)
        history = train_model.train()
        loss_graph = Plotter()
        loss_graph.plot('loss', 'Training and Validation loss for {} epochs with {} level of gaussian noise'
                        .format(len(history.history['loss']), args.sigma), history, original_img=None,
                        reconstructed_img=None)

    elif args.test is True:
        if args.path_to_model is None or not os.path.exists(args.path_to_model):
            print("No model found. Please select a trained model.")
        else:
            model = tf.keras.models.load_model(args.path_to_model)
            test_model = Test_model(data.test_data, model, args.sigma)
            test_model.test()

    elif args.nn is True:
        dcnn_fista = DcNN_Fista(args.iterations, args.scalar, args.path_to_model, args.lambda_)
        x, finished_iterations = dcnn_fista.dcnn_fista(data.sys_matrix, data.coil_ges, original_img,
                                                       lip_constant=args.lip_constant)
        x = x.reshape(40, 40)
        mse = mean_squared_error(original_img.astype('float'), x.astype('float'))
        nn_graph = Plotter()
        nn_graph.plot('nn', 'Reconstructed using neural network and denoiser scaling' + '\n'
                      + 'MSE: {:.2f} in {} iterations'.format(mse, finished_iterations), _object=None,
                      original_img=original_img, reconstructed_img=x)

    else:
        fista = Fista(args.lambda_, args.iterations)
        x, finished_iterations = fista.fista(data.sys_matrix, data.coil_ges, original_img,
                                             lip_constant=args.lip_constant)
        x = x.reshape(40, 40)
        mse = mean_squared_error(original_img.astype('float'), x.astype('float'))
        fista_graph = Plotter()
        fista_graph.plot('fista', 'Reconstructed using FISTA Algorithm' + '\n'
                         + 'MSE: {:.2f} in {} iterations'.format(mse, finished_iterations), _object=None,
                         original_img=original_img, reconstructed_img=x)
