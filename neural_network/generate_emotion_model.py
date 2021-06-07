from datetime import datetime
import os
import sys

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)

from keras.optimizers import Adam
from neural_network.src.data_preprocessing import DataPreProcessor
from neural_network.src.cnn_model import CNNModel
from neural_network.src.models import ActivationFunction, Models

train_dir = 'neural_network/dataset/fer2013/train/'
test_dir = 'neural_network/dataset/fer2013/test/'

row, col = 48, 48
classes = 7

# train_dir = 'neural_network/dataset/affectnet8/train/'
# test_dir = 'neural_network/dataset/affectnet8/val/'

# row, col = 224, 224
# classes = 8

batch = 32
epoch = 160
model = Models.res_net_v50

activation_fct = ActivationFunction.relu

data_processor = DataPreProcessor(
    images_train_dir=train_dir, images_test_dir=test_dir, batch_size=batch, image_shape=[row, col])

training_set = data_processor.get_train_set
test_set = data_processor.get_test_set

opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
cnn_model = CNNModel(optimiser=opt, epochs=epoch, n_classes=classes, input_size=(row, col, 1))
cnn_model.compute_model(model_choice=model, activation_func=activation_fct)

print(cnn_model.get_model_summary)

print(model[:6])
if model[:6] == 'resnet':
    model = 'res_net'

date = datetime.now()
cnn_model.generate_model_plot(filename='neural_network/models/' + model + '/' + date.strftime('%d-%m-%yT%Hh%Mm%Ss'))

# possible to modify steps per epoch and validation steps
cnn_model.train_model(training_set=training_set, test_set=test_set, model_choice=model)

date = datetime.now()
cnn_model.save_model(filename='neural_network/models/' + model + '/'+ date.strftime('%d-%m-%yT%Hh%Mm%Ss'))

cnn_model.plot_accuracy_and_loss_plotly()
