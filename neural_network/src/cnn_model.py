import datetime
from os.path import isfile, join

import attr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from keras_preprocessing.image import img_to_array
from plotly.offline import iplot
from plotly.subplots import make_subplots

import tensorflow as tf
import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

from neural_network.src.models import MODELS


@attr.s
class CNNModel:
    input_size = attr.ib(default=(48, 48, 1))
    n_classes = attr.ib(default=7)
    optimiser = attr.ib(default='adam')
    epochs = attr.ib(default=60)
    hist = attr.ib(default=None)
    _model = attr.ib(default=None, type=Sequential)
    _train_loss = attr.ib(default=None)
    _train_acc = attr.ib(default=None)
    _test_loss = attr.ib(default=None)
    _test_acc = attr.ib(default=None)
    _model_type = attr.ib(default='model1')

    SAVE_DIRECTORY = '../model/'

    @property
    def get_model_summary(self):
        return self._model.summary()

    @property
    def get_model(self):
        return self._model

    def __attrs_post_init__(self):
        pass

    def compute_model(self, model_choice, activation_func):
        self._model = self._build_model(model_choice=model_choice, activation_fct=activation_func)
        self._compile_model(opt=self.optimiser)

    def _build_model(self, model_choice='model1', activation_fct='relu'):
        # init CNN
        # tf.debugging.set_log_device_placement(True)
        #
        # # Create some tensors
        # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # c = tf.matmul(a, b)
        #
        # print(c)

        # # make sure soft-placement is off
        # tf_config = tf.ConfigProto(allow_soft_placement=False)
        # tf_config.gpu_options.allow_growth = True
        # s = tf.Session(config=tf_config)
        # k.set_session(s)
        self._model_type = model_choice

        with tf.device('/gpu:0'):
            model = Sequential()

            model = MODELS[model_choice](model, self.input_size, self.n_classes, activation_fct)

        return model

    def _compile_model(self, opt):
        self._model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_model(self, training_set, test_set, steps_per_epoch=None, validation_steps=None):
        if steps_per_epoch is None:
            steps_per_epoch = training_set.n // training_set.batch_size
        if validation_steps is None:
            validation_steps = test_set.n // test_set.batch_size

        log_dir = "neural_network/logs/fit/" + self._model_type + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.hist = self._model.fit(x=training_set,
                                    validation_data=test_set,
                                    epochs=self.epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=validation_steps,
                                    callbacks=tensorboard_callback, 
                                    # use_multiprocessing=True,
                                    # workers=8
                                    )

    def evaluate_model(self, training_set, test_set):
        self._train_loss, self._train_acc = self.compute_accuracy_and_loss(training_set)
        self._test_loss, self._test_acc = self.compute_accuracy_and_loss(test_set)

        print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(self._train_acc * 100,
                                                                                    self._test_acc * 100))
        print("final train loss = {:.2f} , validation loss = {:.2f}".format(self._train_loss * 100,
                                                                            self._test_loss * 100))

    def generate_model_plot(self, filename):
        plot_model(self._model, to_file=filename + '.png', show_shapes=True, show_layer_names=True)

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
        self._model.save(filename)
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
