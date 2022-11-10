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


df01 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df02 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df03 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df04 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df05 = pd.DataFrame(columns = ['Timestamp', 'Signal'])

for t in range(8000):

    #Y = (sin(10 * 0.01 * t) * exp(-0.1 * 0.01 * t))
    y = np.sin(18*0.01*t) * np.exp(-0.1*0.01*t)
    #y = np.sin(18*0.01*t) 
    
    #Z = (sin(8 * 0.01 * t) * exp(-0.2 * 0.01 * t))
    z = np.sin(15*0.01*t) * np.exp(-0.1*0.01*t)
    #z = np.sin(15*0.01*t) 
    
    #X = (sin(6 * 0.01 * t) * exp(-0.2 * 0.01 * t))
    x = np.sin(12*0.01*t) * np.exp(-0.1*0.01*t)

    #m = np.sin(9*0.01*t) * np.exp(-0.05*0.01*t)

    
    p = y + z + x
    #q = p * np.exp(0.1*0.01*t)
    
    #df01 = pd.concat([df01, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':y }])], ignore_index=True)
    #df02 = pd.concat([df02, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':z }])], ignore_index=True)
    #df03 = pd.concat([df03, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':x }])], ignore_index=True)
    df01 = pd.concat([df01, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':p }])], ignore_index=True)
    #df05 = pd.concat([df05, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':q }])], ignore_index=True)

#my_array = [ df01, df02, df03, df04]
#my_array = [ df01]

    #plt.rcParams["figure.figsize"] = (15,5.5)

#plt.subplot(1, 4, 1)

#plt.plot( df01['Timestamp'].iloc[1:300], df01['Signal'].iloc[1:300], 'g' )
    
#plt.subplot(1, 4, 2)
#plt.plot(df02['Timestamp'].iloc[1:300], df02['Signal'].iloc[1:300], 'r' )
    
#plt.subplot(1, 4, 3)
#plt.plot(df03['Timestamp'].iloc[1:300], df03['Signal'].iloc[1:300], 'b' )
    
#plt.subplot(1, 4, 4)
#plt.plot(df04['Timestamp'].iloc[1:300], df04['Signal'].iloc[1:300], 'y' )
    
    
# space between the plots
#plt.tight_layout()
 
# show plot
#plt.show()


def _load_data(data, n_prev = 100):  

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def _divide_data(dataset, length_of_sequences, test_size = 0.25):
    ntr = int(len(dataset) * (1 - test_size))
    df_train = dataset[["Signal"]].iloc[:ntr]
    df_test  = dataset[["Signal"]].iloc[ntr:]
    (X_train, Y_train) = _load_data(df_train, n_prev = length_of_sequences)
    (X_test, Y_test)   = _load_data(df_test, n_prev = length_of_sequences)

    return X_train, Y_train, X_test, Y_test

(X_train, Y_train, X_test, Y_test) = _divide_data(df01, 2)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

model = load_model('Sine_wave0.h5')
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
plot_model(model, to_file='Sine_model.png', show_shapes=True, show_layer_names=True)
X_test = np.asarray(X_test).astype(float)
y_pred = model.predict(X_test)

            #plt.ylim(-0.11,0.11)

plt.plot(Y_test,label="true")
plt.plot(y_pred,label="predicted")
plt.show()
