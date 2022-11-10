import tensorflow
print(tensorflow.__version__)
import keras
print(keras.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense, Activation 
from keras.layers.recurrent import SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
from keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from keras.models import load_model
from keras.utils.vis_utils import plot_model

headers = ['Timestamp', 'Signal']

df = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFCombline\\reflected.csv',names= headers, sep= '\t'
                 ,skiprows=range(0,3216),float_precision=None)

df.head(5)
print(df[:5])
df.plot(x ='Timestamp', y = 'Signal' )
plt.show()

df_test = df['Signal'].to_numpy()

def _load_data(data, n_prev = 100):  

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev])
        docY.append(data[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

(X_test, Y_test) = _load_data(df_test, n_prev = 20)

print(X_test.shape, Y_test.shape)

model = load_model('Transmitted_model0.h5')
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
plot_model(model, to_file='Sine_model.png', show_shapes=True, show_layer_names=True)
X_test = np.asarray(X_test).astype(float)
y_pred = model.predict(X_test)

val_loss = 0
for i in range(len(Y_test)):
    val_loss = val_loss + ((Y_test[i] - y_pred[i])**2)
result = val_loss / len(Y_test)
print("validation loss is:", result)

#plt.ylim(-0.11,0.11)

plt.plot(Y_test,label="true", linewidth = 3)
plt.plot(y_pred,label="predicted")
plt.legend()
plt.show()
