import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

"""
==========================================
Wine Quality - SVM
==========================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have numpy, matplotlib, and sklearn packages.
Program predicts quality of white wine on scale given chemical measures of wine, using SVM Algorithm.
It uses The Wine Quality Dataset, which was imported from sklearn datasets library.
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

"""
Importing required data
"""
wine = datasets.load_wine()
X = wine.data[:, :2]
y = wine.target

"""
Splitting the data into 2 parts - training and testing data.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y)

"""
Creating an instance of SVM and fitting out data
"""
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

"""
The accuracy of the algorithm.
"""
print("Prediction accuracy:")
print((svc.score(X_test, y_test).__round__(4)))

"""
Creting a mesh to plot in
"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min) / 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Fixed acidity')
plt.ylabel('Volatile acidity')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
