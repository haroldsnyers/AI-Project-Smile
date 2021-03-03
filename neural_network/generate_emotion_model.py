from datetime import datetime

from keras.optimizers import Adam
from neural_network.src.data_preprocessing import DataPreProcessor
from neural_network.src.cnn_model import CNNModel

train_dir = 'dataset/fer2013/train/'
test_dir = 'dataset/fer2013/test/'

row, col = 48, 48
classes = 7

data_processor = DataPreProcessor(images_train_dir=train_dir, images_test_dir=test_dir)

training_set = data_processor.get_train_set
test_set = data_processor.get_test_set

opt = Adam(lr=0.0001, decay=10e-6)
cnn_model = CNNModel(optimiser=opt)
cnn_model.compute_model(model_choice='xception')

date = datetime.now()
cnn_model.generate_model_plot(filename='model/' + date.strftime('%d-%m-%yT%Hh%Mm%Sd'))

# possible to modify steps per epoch and validation steps
cnn_model.train_model(training_set=training_set, test_set=test_set)

date = datetime.now()
cnn_model.save_model(date.strftime('%d-%m-%yT%H:%M:%S'))

