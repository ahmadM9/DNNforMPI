import tensorflow as tf
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from model import denoise_network
from utitlities import datagen, step_decay


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


class Train_model:

    def __init__(self, training_set, validation_set, batch_size, epoch, sigma, lr):
        self.training_set = training_set
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.epoch = epoch
        self.sigma = sigma
        self.lr = lr

    def train(self):
        # Reshaping and scaling the training data
        training_set = self.training_set.reshape((self.training_set.shape[0], self.training_set.shape[1],
                                                  self.training_set.shape[2], 1))
        training_set = training_set.astype('float32') / 255.0

        # Reshaping and scaling the validation data
        validation_set = self.validation_set.reshape((self.validation_set.shape[0], self.validation_set.shape[1],
                                                      self.validation_set.shape[2], 1))
        validation_set = validation_set.astype('float32') / 255.0

        # Create the Model
        model = denoise_network()

        # Compiling the model
        model.compile(optimizer='adam', loss=['mse'], metrics=['acc'])

        # Using callback
        loss_history = LossHistory()
        lr = LearningRateScheduler(step_decay)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        # Training the model
        history = model.fit(datagen(training_set, sigma=self.sigma, batch_size=self.batch_size),
                            validation_data=datagen(validation_set, sigma=self.sigma, batch_size=self.batch_size),
                            validation_steps=len(validation_set) // self.batch_size,
                            steps_per_epoch=len(training_set) // self.batch_size, epochs=self.epoch,
                            callbacks=[loss_history, early_stopping, lr])

        model.save('./Trained_Models/model_epochs={}_lr={}_sigma={}.h5'.format(len(history.history['loss']), self.lr,
                                                                               self.sigma))

        return history
