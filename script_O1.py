# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 08:14:35 2021

@author: HP-Pro
"""

from keras import backend as K
from tensorflow.python.keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.initializers import Initializer
from sklearn.cluster import KMeans


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_

# Commented out IPython magic to ensure Python compatibility.
import numpy as np, pandas as pd
from keras.models import Sequential 
from keras.layers.core import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
da=pd.read_csv('Classeur1.csv')
data1=pd.read_csv('yield_data.csv', )
dt = pd.merge(da, data1)
#### preprocess de mes données
from pandas import get_dummies
dt =get_dummies(dt)
dt = dt.assign(REND_AGGR = dt['REND'] / 7)
dt.replace('?',-99999, inplace=True)
dt.drop(['REND'], 1, inplace=True)
dt.head(10) #Return 10 rows of data

dttrans=np.transpose(dt)
print(dttrans[0].value_counts())
dttrans[0].value_counts()[:].plot(kind='bar', alpha=0.5)
plt.xlabel('\n Figure 1: Répartition selon classes \n', fontsize='17', horizontalalignment='center')
plt.tick_params(axis='x',  direction='out', length=10, width=3)

plt.show() #2300
#data spliting
X=dt.iloc[:,:-1].values
y = dt.iloc[:,-1].values
#standarizing
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('resulats de scalling')
print(X,y)
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)#80% train et 20% test

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# sc_y = StandardScaler()
# y_train = y_train.reshape((len(y_train), 1))
# y_train = sc_y.fit_transform(y_train)
# y_train = y_train.ravel()
model = Sequential()
rbflayer = RBFLayer(34,initializer=InitCentersKMeans(X_train),
                        betas=2.0,input_shape=(875,))
model.add(rbflayer)
model.add(Dense(1))
model.add(Activation('linear'))
opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='mean_squared_error',
                  optimizer=opt, metrics=['accuracy'])

print(model.summary())
