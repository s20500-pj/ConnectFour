import numpy as np
import warnings
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree, datasets
import matplotlib.pyplot as plt

"""
=====================================================================
Pen-Based Recognition of Handwritten Digits Data Set - Decision Tree
=====================================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have numpy, matplotlib and sklearn packages.
Program predicts handwritten digits using Decision Tree Algorithm.
It uses the Digits Dataset, which you can find on: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
This database was created by collecting 250 samples from 44 writers. Digits are represented as constant length vectors
with 16 input arguments which all are integers in the range 0...100.
The last attribute is the class code 0..9.
"""

"""
Load input data
"""
digits = datasets.load_digits()

"""
Visualisation of sample data
"""
plt.title("Sample data - digit '1'")
plt.imshow(digits.images[1])
plt.show()

warnings.filterwarnings("ignore")

"""
Changing shape of image data from 3D to 2D
"""
X = digits.images.reshape(len(digits.images), -1)
y = digits.target

"""
Split input data into ten classes based on labels
"""
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])
class_3 = np.array(X[y == 3])
class_4 = np.array(X[y == 4])
class_5 = np.array(X[y == 5])
class_6 = np.array(X[y == 6])
class_7 = np.array(X[y == 7])
class_8 = np.array(X[y == 8])
class_9 = np.array(X[y == 9])

"""
Splitting data into training and testing datasets
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

"""
To fit the data , a Machine Learning model called Decision Tree Classifier is being used
"""
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

"""
The output was predicted by passing X_test and also stored real target in expected_y
"""
expected_y = y_test
predicted_y = model.predict(X_test)

"""
Printing the classification report
"""
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
               'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
print(metrics.classification_report(expected_y, predicted_y, target_names=class_names))
