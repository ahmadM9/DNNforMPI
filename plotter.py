import matplotlib.pyplot as plt


class Plotter:

    def plot(self, graph, title, _object, original_img, reconstructed_img):

        if graph == 'loss':
            # Plotting the results of training
            train_loss = _object.history['loss']
            val_loss = _object.history['val_loss']
            plt.plot(train_loss, 'b', label='Training loss')
            plt.plot(val_loss, 'r', label='Validation loss')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('./results/loss.png')

        if graph == 'nn':
            f, axarr = plt.subplots(1, 2)
            f.suptitle(title)
            axarr[0].imshow(original_img)
            axarr[0].set_title('Original')
            axarr[1].imshow(reconstructed_img)
            axarr[1].set_title('Reconstructed')
            plt.savefig('./results/DNN.png')

        if graph == 'fista':
            plt.title(title)
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(original_img)
            axarr[0].set_title('Original')
            axarr[1].imshow(reconstructed_img)
            axarr[1].set_title('Reconstructed')
            plt.savefig('./results/fista.png')

        plt.show()
