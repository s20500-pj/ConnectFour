import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

"""
==========================================
Wine Quality - Decision Tree
==========================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have numpy, matplotlib, and sklearn packages.
Program predicts quality of white wine on scale given chemical measures of wine, using Decision Tree Algorithm.
It uses The Wine Quality Dataset, which you can find on: https://machinelearningmastery.com/standard-machine-learning-datasets/
It is a multi-class classification problem. There are 11 inputs and one output variable.
The output is quality of wine on scale from 0 to 10.

The variable names are as follows:
1. Fixed acidity.
2. Volatile acidity.
3. Citric acid.
4. Residual sugar.
5. Chlorides.
6. Free sulfur dioxide.
7. Total sulfur dioxide.
8. Density.
9. pH.
10. Sulphates.
11. Alcohol.
12. Quality (score between 0 and 10).
"""

warnings.filterwarnings("ignore")

"""
Load input data
"""
input_file = 'winequality-white.csv'
dataset = np.loadtxt(input_file, delimiter=';')
X, y = dataset[:, :11], dataset[:, -1]

"""
Split input data into eleven classes based on labels
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
class_10 = np.array(X[y == 10])

"""
Visualization of input data
"""

plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
            edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='green',
            edgecolors='black', linewidth=1, marker='p')
plt.scatter(class_3[:, 0], class_3[:, 1], s=75, facecolors='red',
            edgecolors='black', linewidth=1, marker='X')
plt.scatter(class_4[:, 0], class_4[:, 1], s=75, facecolors='blue',
            edgecolors='black', linewidth=1, marker='+')
plt.scatter(class_5[:, 0], class_5[:, 1], s=75, facecolors='orange',
            edgecolors='black', linewidth=1, marker='*')
plt.scatter(class_6[:, 0], class_6[:, 1], s=75, facecolors='purple',
            edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_7[:, 0], class_7[:, 1], s=75, facecolors='brown',
            edgecolors='black', linewidth=1, marker='P')
plt.scatter(class_8[:, 0], class_8[:, 1], s=75, facecolors='yellow',
            edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_9[:, 0], class_9[:, 1], s=75, facecolors='green',
            edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_10[:, 0], class_10[:, 1], s=75, facecolors='pink',
            edgecolors='black', linewidth=1, marker='X')

plt.title('Input data for wine quality predicting')
plt.show()

"""
Splitting data into training and testing datasets
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

"""
To fit the data , a Machine Learning model called Decision Tree Classifier is beeing used
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
class_names = ['Quality Score: 0', 'Quality Score: 1', 'Quality Score: 2', 'Quality Score: 3', 'Quality Score: 4',
               'Quality Score: 5', 'Quality Score: 6', 'Quality Score: 7', 'Quality Score: 8', 'Quality Score: 9',
               'Quality Score: 10']
print(metrics.classification_report(expected_y, predicted_y, target_names=class_names))
