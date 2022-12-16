import tensorflow as tf
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from sklearn.metrics import r2_score

"""
===========================================================
Concrete Strength prediction using Neural Network
===========================================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have matplotlib, tensorflow, sklearn, pandas and keras packages.
Program uses neural network built with TensorFlow library to learn and predict concrete
strength based on materials used to make it. 
The program uses concrete_strength dataset which can be found here:
https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
It consists of 1030 observations and has 8 input attributes and 1 output attribute.
"""

"""
Downloading the dataset.
"""
df = pd.read_csv('concrete_data.csv')
x_org = df.drop('csMPa', axis=1).values
y_org = df['csMPa'].values

"""
Splitting data into training and testing datasets.
"""
X_train, X_test, y_train, y_test = train_test_split(x_org, y_org, test_size=0.3)

"""
Scaling data.
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Defining Root Mean Square Error As our Metric Function.
"""


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


"""
Building the model.
"""
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(1, activation='linear'))

"""
Optimizing, compiling and training the model
"""
opt = keras.optimizers.Adam(lr=0.0015)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[rmse])
history = model.fit(X_train, y_train, epochs=35, batch_size=32, validation_split=0.1)
print(model.summary())

"""
Making predictions.
"""
y_predict = model.predict(X_test)
print('Accuracy: ', round(r2_score(y_test, y_predict), 4))
