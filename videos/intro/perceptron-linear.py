# Neural nets care about the scale of the date. Keras and Pytorch give much 
# better results if data is normalised between 0 and 1, or -1 and 1
# Steps taken in this tutorial to IMPROVE a neural net
# 1. Set activation function to softmax
# 2. Set loss function to categorical cross entropy (which is a good idea if doing categorisation)
# 3. Normalised data (always a good idea, because Neural net software cares about the scale of data)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float")
X_train /= 255.
X_test  = X_test.astype("float")
X_test /= 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
# Adding a second dense layer with 100 hidden nodes (config.hidden_nodes = 100) - This has been added at the end of the example to see what would happen
# This has actually resulted in OVERFITTING. One can tell that there's overfitting when accuracy is better than validation accuracy or test accuracy.
# Validation accuracy should improve, as it is the accuracy of the model with data it hasn't seen yet
# To avoid overfitting, or if overfitting is already happening, add a "dropout" layer in between the perceptron layers as below
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu')) 
model.add(Dropout(0.4))
# Now takes num_classes as a parameter instead of 1. Also activation function softmax is what we use in the last layer of the neural network if doing classification
# simply to constrait the output to be between 0 and 1
model.add(Dense(num_classes,activation='softmax')) 
# Change the loss function to "categorical_crossentropy"
# Always use categorical cross entropy if categorising things. Example: if the answer you're looking is 1 and the output is zero, the loss will be infinite, while giving
# diminishing returns the closer you get to 1. The lower the loss the better in a loss function
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
				
model.summary() # Gives a summary of the model, including number of weights, etc

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(labels=labels, data_type="image")])


