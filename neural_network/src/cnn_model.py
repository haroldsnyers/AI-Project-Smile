from os.path import isfile, join

import attr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from keras_preprocessing.image import img_to_array
from plotly.offline import iplot
from plotly.subplots import make_subplots

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.utils.vis_utils import plot_model


@attr.s
class CNNModel:
    input_size = attr.ib(default=(48, 48, 1))
    n_classes = attr.ib(default=7)
    optimiser = attr.ib(default='adam')
    hist = attr.ib(default=None)
    _model = attr.ib(default=None, type=Sequential)
    _train_loss = attr.ib(default=None)
    _train_acc = attr.ib(default=None)
    _test_loss = attr.ib(default=None)
    _test_acc = attr.ib(default=None)

    SAVE_DIRECTORY = 'model/'

    @property
    def get_model_summary(self):
        return self._model.summary()

    @property
    def get_model(self):
        return self._model

    def __attrs_post_init__(self):
        pass

    def compute_model(self):
        self._model = self._build_model()
        self._compile_model(opt=self.optimiser)

    def _build_model(self):
        # init CNN
        model = Sequential()

        # first convolution layer
        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_size))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())

        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # Second Convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu'))

        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # Flattening
        model.add(Flatten())

        # Full Connection
        # one dimensional vector
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))

        # Output Layer
        # the activation function softmax makes sure to rescale the final values between zero and one and
        # that the sum of the values of all out layer neurons is equal to just 1
        model.add(Dense(units=self.n_classes, activation='softmax'))

        return model

    def _compile_model(self, opt):
        self._model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_model(self, training_set, test_set, steps_per_epoch=None, validation_steps=None):
        if steps_per_epoch is None:
            steps_per_epoch = training_set.n // training_set.batch_size
        if validation_steps is None:
            validation_steps = test_set.n // test_set.batch_size

        self.hist = self._model.fit(x=training_set,
                                    validation_data=test_set,
                                    epochs=60,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=validation_steps)

    def evaluate_model(self, training_set, test_set):
        self._train_loss, self._train_acc = self.compute_accuracy_and_loss(training_set)
        self._test_loss, self._test_acc = self.compute_accuracy_and_loss(test_set)

        print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(self._train_acc * 100,
                                                                                    self._test_acc * 100))
        print("final train loss = {:.2f} , validation loss = {:.2f}".format(self._train_loss * 100,
                                                                            self._test_loss * 100))

    def generate_model_plot(self, filename):
        plot_model(self._model, to_file=self.SAVE_DIRECTORY + filename + '.png', show_shapes=True, show_layer_names=True)

    def predict(self, image, emotions, emotion_indices):
        test_image_angry = load_img(image, target_size=(self.input_size[0], self.input_size[1]))
        # change image to be compatible with the prediction method
        test_image_angry = img_to_array(test_image_angry)  # convert to numpy array
        # needed as we trained our data in batches
        test_image_angry = np.expand_dims(test_image_angry,
                                          axis=0)  # axis=0 such that the dimension of the batch is first dimension

        result = self._model.predict(test_image_angry)

        # this can be fault as the prediction will most likely not be round
        result_emotion = emotions[emotion_indices.index(result[0][0])]

        print("the person emotion is :" + result_emotion)

    def save_model(self, filename):
        self._model.save(join(self.SAVE_DIRECTORY, filename))
        print('[+] Model trained and saved at ' + filename)

    def load_model(self, filename):
        filename_dir = filename + '.txt'
        if isfile(join(self.SAVE_DIRECTORY, filename_dir)):
            self._model.load(join(self.SAVE_DIRECTORY, filename_dir))
            print('[+] Model loaded from ' + filename_dir)

    def compute_accuracy_and_loss(self, eval_set):
        return self._model.evaluate(eval_set)

    def plot_accuracy_and_loss_plt(self):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 2)
        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 1)
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_accuracy_and_loss_plotly(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Model Accuracy', 'Model Loss'])
        fig.add_trace(go.Scatter(
            self.hist.history['accuracy'],
            mode='lines',
            name='train',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            self.hist.history['val_accuracy'],
            mode='lines',
            name='test',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            self.hist.history['loss'],
            mode='lines',
            name='train',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            self.hist.history['val_loss'],
            mode='lines',
            name='test',
        ), row=1, col=2)

        iplot(fig)
