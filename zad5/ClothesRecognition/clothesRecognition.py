import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

"""
===========================================================
Clothes Recognition using Neural Network
===========================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have matplotlib, tensorflow, sklearn and numpy packages. Program uses neural network built with
TensorFlow library to learn and recognize clothes from picture. Model uses the Fashion MNIST dataset which contains
70000 images of clothing articles in 10 categories. 60000 images are used for training the network and 10000 images
are used to test it.
"""

"""
Downloading the dataset.
"""
clothes_dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = clothes_dataset.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
Scaling the images.
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Building the model.
The first - Flatten layer transforms images to a one-diamensional array of 28 by 28 pixels.
Next come two Dense layers. First one has 128 nodes (or neurons). The second (and last) layer returns
a logits array with length of 10. Each node contains a score that indicates the current image belongs to
one of the 10 classes.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
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
model.fit(train_images, train_labels, epochs=10)

"""
Evaluating accuracy.
"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

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
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

"""
Creating the Confusion Matrix
"""
y_probs = model.predict(test_images)
y_preds = y_probs.argmax(axis=1)
cm=confusion_matrix(y_preds,test_labels)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax);
plt.show()
