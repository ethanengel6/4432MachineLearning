import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import reciprocal


def sklearn_to_df(sklearn_dataset):
    data = sklearn_dataset['data']
    cols = sklearn_dataset['feature_names']
    target = sklearn_dataset['target']
    df = pd.DataFrame(data, columns=cols)
    df['target'] = pd.Series(target)
    return df

diab_bunch =load_diabetes()
diab_df = sklearn_to_df(diab_bunch)
diab_train, diab_test= train_test_split(diab_df, test_size=0.2, random_state=42)
diab_target_train = diab_train['target']
diab_train.drop(columns=['target'],inplace=True)
diab_target_test=diab_test['target']
diab_test.drop(columns=['target'],inplace=True)

mlp = MLPRegressor(solver='lbfgs', max_iter=10000, activation='relu',
random_state=42, learning_rate_init=0.0001,tol=.0001)
mlp.fit(diab_train, diab_target_train)
pred1 = mlp.predict(diab_test)


#Part2
titanic_df = pd.read_csv('titanic_clean.csv')
titanic_label_df = pd.read_csv('titanic_label.csv')
titanic_df.drop(titanic_df.columns[0],axis=1,inplace=True)
titanic_label_df.drop(titanic_label_df.columns[0],axis=1,inplace=True)
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10
titanic_train, titanic_test, titanic_label_train, titanic_label_test = train_test_split(titanic_df, titanic_label_df, test_size=1 - train_ratio)
titanic_val,titanic_test, titanic_label_val, titanic_label_test = train_test_split(titanic_test, titanic_label_test, test_size=test_ratio/(test_ratio + validation_ratio))
print(titanic_label_train.head())
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(12,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(titanic_train, titanic_label_train, epochs=30, validation_data=(titanic_val, titanic_label_val))
model.evaluate(titanic_test, titanic_label_test)

#Part3
bike_df = pd.read_csv('bike_data.csv')
bike_label_df = pd.read_csv('bike_label.csv')
bike_df.drop(bike_df.columns[0],axis=1,inplace=True)
bike_label_df.drop(bike_label_df.columns[0],axis=1,inplace=True)
print(bike_df.head())
bike_train, bike_test, bike_label_train, bike_label_test = train_test_split(bike_df, bike_label_df, test_size=1 - train_ratio)
bike_val,bike_test, bike_label_val, bike_label_test = train_test_split(bike_test, bike_label_test, test_size=test_ratio/(test_ratio + validation_ratio))
model2 = keras.models.Sequential()
model2.add(keras.layers.Flatten(input_shape=(61,)))
model2.add(keras.layers.Dense(30, activation='relu'))
print(model2.summary())
model2.compile(loss='mean_squared_error', optimizer='sgd',metrics=['mean_squared_error'])
history = model2.fit(bike_train, bike_label_train, batch_size=50,epochs=30, validation_data=(bike_val, bike_label_val))
model2.evaluate(bike_test, bike_label_test)

#Part4
def build_model(optimizer='adam',n_hidden=1,n_neurons=30,learning_rate=.003,input_shape=(61,)):
    model3=keras.models.Sequential()
    model3.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model3.add(keras.layers.Dense(n_neurons, activation='relu'))
        model3.add(keras.layers.Dense(1))
        model3.compile(loss="mse",optimizer=optimizer)
    return model3

keras_reg=KerasRegressor(build_model)
keras_reg.fit(bike_train,bike_label_train,epochs=25,
validation_data=(bike_val,bike_label_val),callbacks=[keras.callbacks.EarlyStopping(patience=10)])
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam','adam']
param_grid={"optimizer":optimizer}
search_cv=RandomizedSearchCV(keras_reg,param_grid,n_iter=10,cv=3)
search_cv.fit(bike_train,bike_label_train,epochs=25,
validation_data=(bike_val,bike_label_val),callbacks=keras.callbacks.EarlyStopping(patience=10))
print(search_cv.best_params_,"RMSE=",((search_cv.best_score_)*-1)**.5)
