import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
===========================================================
Pen-Based Recognition of Handwritten Digits Data Set using Neural Network
===========================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have tensorflow, numpy, matplotlib and sklearn packages.
Program uses neural network built with TensorFlow library to learn and recognize handwritten digits from picture.
It uses the Digits Dataset, which you can find on: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
This database was created by collecting 250 samples from 44 writers. Digits are represented as constant length vectors
with 16 input arguments which all are integers in the range 0...100.
The last attribute is the class code 0..9.
The program was created to compare precision of prediction by Neural Network and Decision Tree - using the same dataset.
"""

"""
Downloading the dataset.
"""
digits = datasets.load_digits()
X = digits.images
y = digits.target
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

"""
Splitting data into training and testing datasets
"""
train_images, test_images, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

"""
Building the model.
The first - Flatten layer transforms images to a one-diamensional array of 8 by 8 pixels.
Next come two Dense layers. First one has 128 nodes (or neurons). The second (and last) layer returns
a logits array with length of 10. Each node contains a score that indicates the current image belongs to
one of the 10 classes.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(8, 8)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

"""
Compiling the model.
There are 3 functions:
- Loss function - measures how accurate the model is during training.
- Optimizer - how the model is updated based on the data it sees and its loss function
- Metrics - is used to monitor the training and testing steps. The following example uses accuracy, the fraction
  of the images that are correctly classified.
"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
Fitting the model to the training data.
"""
model.fit(train_images, y_train, epochs=10)

"""
Evaluating accuracy.
"""
test_loss, test_acc = model.evaluate(test_images,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

"""
Mqking predictions.
"""
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

"""
Testing and ploting graph of predictions.
"""
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

"""
Ploting images with their predictions.
"""
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y)
plt.tight_layout()
plt.show()
