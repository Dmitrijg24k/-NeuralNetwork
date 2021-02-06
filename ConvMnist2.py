import tensorflow as tf
import numpy as np
from tensorflow import keras

# class myCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('acc')>=0.99):
#             print("Достигнута точность 99,8%, поэтому обучение закончено!")
#             self.model.stop_training = True

# def train_mnist():

#     # YOUR CODE SHOULD START HERE
#     # YOUR CODE SHOULD END HERE
#     import tensorflow as tf
#     mnist = tf.keras.datasets.mnist

#     (x_train, y_train),(x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0
    
#     callbacks = myCallback()

#     model = tf.keras.models.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(512, activation=tf.nn.relu),
#         keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['acc'])
    
#     model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

#     return 1

# train_mnist()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.998):
            print("Достигнута точность 99%, поэтому обучение закончено!")
            self.model.stop_training = True

def train_mnist_conv():
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # model fitting
    history = model.fit(training_images, training_labels, epochs=19, callbacks=[callbacks])

    return history.epoch, history.history['acc'][-1]

_, _ =train_mnist_conv()


# import tensorflow as tf
# print(tf.__version__)
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images.reshape(60000, 28, 28, 1)
# training_images=training_images / 255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(training_images, training_labels, epochs=7)
# test_loss = model.evaluate(test_images, test_labels)

# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(3,4)
# FIRST_IMAGE=0
# SECOND_IMAGE=23
# THIRD_IMAGE=28
# CONVOLUTION_NUMBER = 1
# from tensorflow.keras import models
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
# # f1 = activation_model.predict(test_images[FIRST_IMAGE])[0]
# # axarr[0,0].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
# # axarr[0,0].grid(False)
# for x in range(0,4):
#   f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[0,x].grid(False)
#   f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[1,x].grid(False)
#   f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[2,x].grid(False)