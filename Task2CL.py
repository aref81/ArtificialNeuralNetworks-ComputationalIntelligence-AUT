import os
from layers.fullyconnected import FC
from layers.convolution2d import Conv2D
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

data2 = np.expand_dims(data2, axis=-1)
data5 = np.expand_dims(data5, axis=-1)

print(data2.shape)
print(data5.shape)

input_train = np.concatenate((data2[:1000, :, :, :], data5[:1000, :, :, :]), axis=0)
output_train = np.concatenate((np.zeros((1, 1000)), np.ones((1, 1000))), axis=1)

output_valid = np.concatenate((np.zeros((1, 300)), np.ones((1, 300))), axis=1)
input_valid = np.concatenate((data2, data5), axis=1)


architecture = {
    'CONV1': Conv2D(1, 1, 'CONV1', kernel_size=(10, 10), stride=(1, 1), padding=(1, 1)),
    'ACTIVE1': ReLU(),
    'FC1': FC(441, 16, 'FC1'),
    'ACTIVE2': ReLU(),
    'FC2': FC(16, 1, 'FC2'),
    'ACTIVE3': Sigmoid()
}

criterion = BinaryCrossEntropy()
optimizer = GD(architecture, learning_rate=0.3)
model = Model(architecture, criterion, optimizer)

model.train(input_train, output_train, 1000, batch_size=30, shuffling=False, verbose=50)


print(model.predict(input_valid))


