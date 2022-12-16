import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from keras import layers
from keras.utils import image_dataset_from_directory
import matplotlib.image as mpimg

"""
===========================================================
Animals Recognition using Neural Network
===========================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have matplotlib, tensorflow, numpy and keras packages.
Program uses neural network built with TensorFlow library to learn and recognize 
weather given picture presents cat or dog. The dataset consists of 1000 pictures
of cats and dogs. To use program, you need to download test1 folder.
"""

warnings.filterwarnings('ignore')

"""
Data preparation for training
"""
base_dir = 'test1'

train_datagen = image_dataset_from_directory(base_dir,
                                             image_size=(200, 200),
                                             subset='training',
                                             seed=1,
                                             validation_split=0.1,
                                             batch_size=32)
test_datagen = image_dataset_from_directory(base_dir,
                                            image_size=(200, 200),
                                            subset='validation',
                                            seed=1,
                                            validation_split=0.1,
                                            batch_size=32)

"""
Model contains of four convolutional layers followed by MaxPooling layers.
The flatten layer flattens output of the convolutional layer.
Then there are three fully connected layers followed by the output of the 
flattened layer.
The last layer with sigmoid function classifies the result into two classes.
"""
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

"""
Training model, using 10 iterations.
"""
history = model.fit(train_datagen,
                    epochs=10,
                    validation_data=test_datagen)

"""
Visualization the training and validation accuracy witch each iteration.
"""
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

"""
Checking model with different images.
"""
test_image = tf.keras.utils.load_img('test1/824.jpg', target_size=(200, 200))

test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

"""
Result array
"""
result = model.predict(test_image)


"""
Visualizing first tested image with its classification.
"""
i = 0
animal = ""
if (result >= 0.5):
    animal = "Dog"
else:
    animal = "Cat"

img = mpimg.imread("test1/824.jpg")
imgplot = plt.imshow(img)
plt.title(animal)
plt.show()

"""
Another image
"""
test_image = tf.keras.utils.load_img('test1/766.jpg', target_size=(200, 200))

test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

"""
Result array
"""
result = model.predict(test_image)

"""
Visualizing second tested image with its classification.
"""
i = 0
animal = ""
if (result >= 0.5):
    animal = "Dog"
else:
    animal = "Cat"

img = mpimg.imread("test1/766.jpg")
imgplot = plt.imshow(img)
plt.title(animal)
plt.show()
