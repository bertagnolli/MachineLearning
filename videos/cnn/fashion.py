# import numpy
# from keras.datasets import fashion_mnist
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Dropout
# from keras.utils import np_utils
# import wandb
# from wandb.keras import WandbCallback

# # logging code
# run = wandb.init()
# config = run.config
# config.epochs = 10

# # load data
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# img_width = X_train.shape[1]
# img_height = X_train.shape[2]
# labels =["T-shirt/top","Trouser","Pullover","Dress",
    # "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# num_classes = y_train.shape[1]

# # build model
# model = Sequential()
# # This model does convolution instead of flattening! That's why there's a convolutional layer
# # Notice that giving the input shape is required (28,28,1)
# model.add(Conv2D(32,
    # (config.first_layer_conv_width, config.first_layer_conv_height),
    # input_shape=(28, 28,1),
    # activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32,
    # (config.first_layer_conv_width, config.first_layer_conv_height),
    # input_shape=(13, 13,1),
    # activation='relu'))
	
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dropout(0.4))
# model.add(Dense(config.dense_layer_size, activation='relu'))
# model.add(Dropout(0.4))

# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam',
                # metrics=['accuracy'])

# model.summary()

# # Fit the model
# model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    # callbacks=[WandbCallback(data_type="image", labels=labels)])
# This is the final challenge example from https://www.youtube.com/watch?v=wzy8jI-duEQ&list=PLD80i8An1OEHSai9cf-Ip-QReOVW76PlB&index=5&pbjreload=101
# Open Anaconda
# activate ml-class 
# navigate to ml-class folder
# wandb login - add api from wandb.ai website (croto, k4x)
# cc63bd6814ea995ce93de845da788169427eb8e6
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb

run = wandb.init()
config = run.config
config.img_width = 28
config.img_height = 28
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dense_layer_size = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
# This is because Keras' 2D convolutions want 3D inputs, because images are 2D + 1D for colour (RGB)
X_train = X_train.reshape(X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#labels=range(10)
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# build model
model = Sequential()
# This model does convolution instead of flattening! That's why there's a convolutional layer
# Notice that giving the input shape is required (28,28,1)
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(28, 28,1),
    activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(13, 13,1),
    activation='relu'))
	
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(config.dense_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

model.summary()
# Model summary will give the summary of the network layers, as below:
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 32)        320         # The output of the convolution produces 32 different images
#																	 # It shrinks a 28x28 image to 26x26 because of the convolution (it loses the end pixels)
#																	 # The output size of a convolution layer is really large
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0           # Max pooling reduces the image by half
# _________________________________________________________________
# dropout (Dropout)            (None, 13, 13, 32)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 5408)              0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 5408)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               540900
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1010
# =================================================================

# This CNN produces a 98.9% accuracy, but to improve to 99% it will require another Convolution layer. The reason is
# because a convolution only acts at one scale, but if we shrink the image down and do a convolution on another scale,
# we can detect patterns at multiple scales, which is supposed to be much more efficient. 
# A typical application for an image recognition task will have multiple convolution layers and some sort of shrinking
# operation in between (such as max pooling)
model.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=config.epochs,
        callbacks=[WandbCallback(data_type="image")])

model.save('FashionMNISTtest.h5')