import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

"""
===========================================================
Pen-Based Recognition of Handwritten Digits Data Set - SVM
===========================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have matplotlib and sklearn packages.
Program predicts handwritten digits using SVM Algorithm.
It uses the Digits Dataset, which you can find on: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
This database was created by collecting 250 samples from 44 writers. Digits are represented as constant lenght vectors
with 16 input arguments which all are integers in the range 0...100.
The last atribute is the class code 0..9.
Some scientists claims that it predicts the digit accurately more than 95% of the times.

"""

"""
Importing required data
"""
digits = datasets.load_digits()

"""
Visualisation of sample data
"""
plt.title("Sample data - digit '1'")
plt.imshow(digits.images[1])
plt.show()

"""
Printing data shapes:
"""
print("Data shapes:")
print(digits.images.shape)
print(digits.target.shape)

"""
Changing shape of image data from 3D to 2D
"""

X = digits.images.reshape(len(digits.images), -1)
y = digits.target

"""
Printing data shapes:
"""

print("Data shapes after reshaping:")
print(X.shape)
print(y.shape)

"""
Splitting the data into 2 parts - training and testing data.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y)

"""
Creating an instance of SVM
"""
svc = svm.SVC(kernel='rbf', C=1, gamma=0.001)

"""
Checking how algorithm works by feeding it with testing data
"""
svc.fit(X_train, y_train)

"""
The accuracy of the algorithm.
"""
print("Prediction accuracy:")
print((svc.score(X_test, y_test).__round__(3)))
