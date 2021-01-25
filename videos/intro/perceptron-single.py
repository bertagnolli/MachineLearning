# Open Anaconda
# activate ml-class 
# navigate to ml-class folder
# wandb login - add api from wandb.ai website (croto, k4x)
# cc63bd6814ea995ce93de845da788169427eb8e6
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

config.epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model=Sequential() # because it is a series of steps
model.add(Flatten(input_shape=(img_width,img_height))) #flattens the image from 28x28 to a 1 dimensional array (How? Don't know... Why? Becayse that's how NN works)
# Dense because every single neuron input is connected to every output. Our network output 1 single number and that's where (1) comes from
# Changing the activation function to sigmoid, as it will scale highly negative numbers to zero and highly positive numbers to 1
model.add(Dense(1,activation='sigmoid')) 
# Loss = how much we dislike the output or how different the output is from what we want it to be (simplest loss funcion is Mean absolute error, fancier Mean Square error hence "mse")
# Optimiser: Another parameter to setup is which gradient descent algorithm to use (learning rate is the most crucial and difficult parameter to setup on a gradient descent, which is essentially
# how fast we change the weights). "adam" is the most used because it adapts the descent depending on the case
# Metric set to accuracy which doesn't change the algorithm, but makes Keras output the accuracy as it learns
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])  
			

# Fit the model
model.fit(X_train, is_five_train, epochs=config.epochs, validation_data=(X_test, is_five_test),
                    callbacks=[WandbCallback(labels=labels, data_type="image")])

#print(model.predict(X_test[:10,:,:]))


