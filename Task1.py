import pandas
from layers.fullyconnected import FC
from activations import *
from model import Model
from optimizers.gradientdescent import GD
from losses.meansquarederror import MeanSquaredError

train_data = pandas.read_csv("datasets/california_houses_price/california_housing_train.csv").to_numpy()
test_data = pandas.read_csv("datasets/california_houses_price/california_housing_test.csv").to_numpy()

input_train = train_data[:, 0:8].T
output_train = train_data[:, 8].T
output_train = np.expand_dims(output_train, axis=-1).T

input_train[1:8, :] /= np.max(input_train[1:8, :], axis=1, keepdims=True)
input_train[0, :] /= np.min(input_train[0, :], keepdims=True)

max = np.max(output_train)
print(max)
output_train /= max

input_test = test_data[:, :8].T
output_test = test_data[:, 8].T

input_test[1:8, :] /= np.max(input_test[1:8, :], axis=1, keepdims=True)
input_test[0, :] /= np.min(input_test[0, :], keepdims=True)

architecture = {
    'FC1': FC(8, 32, 'FC1'),
    'ACTIVE1': ReLU(),
    'FC2': FC(32, 16, 'FC2'),
    'ACTIVE2': ReLU(),
    'FC3': FC(16, 1, 'FC3'),
    'ACTIVE3': ReLU()
}

criterion = MeanSquaredError()
optimizer = GD(architecture, learning_rate=0.7)
model = Model(architecture, criterion, optimizer)

model.train(input_train, output_train, 1000, batch_size=1000, shuffling=False, verbose=50)

print(np.array(model.predict(input_test[:, :10])))
