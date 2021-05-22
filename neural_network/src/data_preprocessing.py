import attr
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from keras_preprocessing.image import ImageDataGenerator, os
from tensorflow.python.keras.preprocessing.image import load_img


@attr.s
class DataPreProcessor:
    images_train_dir = attr.ib(default=None)
    images_test_dir = attr.ib(default=None)
    batch_size = attr.ib(default=64)
    image_shape = attr.ib(default=[48, 48])
    _train_datagen = attr.ib(default=None)
    _test_datagen = attr.ib(default=None)
    _emotion_dict = attr.ib(default=None)
    _emotions = attr.ib(default=None)
    _emotions_indices = attr.ib(default=None)
    _train_count = attr.ib(default=None)
    _test_count = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.compute_count_images()
        self.generate_train_data_gen()
        self.generate_test_data_gen()
        self._train_set = self.generate_data_set(self.images_train_dir)
        self._test_set = self.generate_data_set(self.images_test_dir)
        self._emotion_dict = self._train_set.class_indices
        self._emotions = list(self._emotion_dict.keys())
        self._emotions_indices = list(self._emotion_dict.values())

    @property
    def get_train_set(self):
        return self._train_set

    @property
    def get_test_set(self):
        return self._test_set

    @property
    def get_train_count(self):
        return self._train_count

    @property
    def get_emotions_dataset(self):
        return self._emotion_dict

    @property
    def get_emotions_indices(self):
        return self._emotions_indices

    @property
    def get_test_count(self):
        return self._test_count

    def generate_train_data_gen(self):
        self._train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # feature scaling (like normalization -> put every pixel between 0 and 1
            zoom_range=0.2,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

    def generate_test_data_gen(self):
        self._test_datagen = ImageDataGenerator(rescale=1. / 255)

    def generate_data_set(self, dir):
        return self._train_datagen.flow_from_directory(dir,
                                                       batch_size=self.batch_size,
                                                       target_size=(self.image_shape[0], self.image_shape[1]),
                                                       shuffle=True,
                                                       color_mode='grayscale',
                                                       class_mode='categorical')

    @staticmethod
    def _count_exp(path, set_):
        dict_ = {}
        for expression in os.listdir(path):
            dir_ = path + expression
            dict_[expression] = len(os.listdir(dir_))
        df = pd.DataFrame(dict_, index=[set_])
        return df

    def compute_count_images(self):
        self._train_count = self._count_exp(path=self.images_train_dir, set_='train')
        self._test_count = self._count_exp(path=self.images_test_dir, set_='test')

    def plot_image_count(self):
        df = pd.DataFrame()
        df = pd.concat([df, self._train_count, self._test_count])
        train_number, test_number = df.transpose().sum()[0], df.transpose().sum()[1]

        fig = px.bar(
            df.transpose(),
            title='Training (' + str(train_number) + ') and Test (' + str(test_number) + ') count')
        fig.show()

    def plot_sample_images(self):
        plt.figure(figsize=(14, 22))
        i = 1
        for expression in os.listdir(self.images_train_dir):
            img = load_img(
                (self.images_train_dir + expression + '/' + os.listdir(self.images_train_dir + expression)[1]))
            plt.subplot(1, 7, i)
            plt.imshow(img)
            plt.title(expression)
            plt.axis('off')
            i += 1
        plt.show()
