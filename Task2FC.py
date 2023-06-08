import os
from layers.fullyconnected import FC
from activations import*
from model import Model
from optimizers.gradientdescent import GD
from PIL import Image
from losses.binarycrossentropy import BinaryCrossEntropy


pics2 = ['datasets/MNIST/2/' + fileName for fileName in os.listdir('datasets/MNIST/2/')]
pics5 = ['datasets/MNIST/5/' + fileName for fileName in os.listdir('datasets/MNIST/5/')]

data2 = []
data5 = []
for pic in pics2:
    data2.append(np.array(Image.open(pic)))

for pic in pics5:
    data5.append(np.array(Image.open(pic)))


data2 = np.array(data2)/255
data5 = np.array(data5)/255

print(data2.shape)
print(data5.shape)

data2 = data2.reshape(1000, -1).T
data5 = data5.reshape(1000, -1).T

print(data2.shape)
print(data5.shape)

input_train = np.concatenate((data2, data5), axis=1)
output_train = np.concatenate((np.zeros((1, 1000)), np.ones((1, 1000))), axis=1)

output_valid = np.concatenate((np.zeros((1, 300)), np.ones((1, 300))), axis=1)
input_valid = np.concatenate((data2, data5), axis=1)


architecture = {
    'FC1': FC(784, 32, 'FC1'),
    'ACTIVE1': ReLU(),
    'FC2': FC(32, 16, 'FC2'),
    'ACTIVE2': ReLU(),
    'FC3': FC(16, 1, 'FC3'),
    'ACTIVE3': Sigmoid()
}

criterion = BinaryCrossEntropy()
optimizer = GD(architecture, learning_rate=0.3)
model = Model(architecture, criterion, optimizer)

model.train(input_train, output_train, 1000, batch_size=30, shuffling=False, verbose=50)

print(model.predict(input_valid))


