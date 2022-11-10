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
#from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
from keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json

headers = ['Timestamp', 'Signal']
df01 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\Ring Resonator\\reflected.csv',
                 names= headers, sep= '\t', float_precision=None)   


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

(X_train, Y_train, X_test, Y_test) = _divide_data(df01, 20)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

def build_model(hp):
    model = Sequential()
    model.add(SimpleRNN(1,return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
    num_layers = hp.Int('num_layers', 0, 4)
    print("layers are : " ,num_layers)
    for i in range(num_layers):
        with hp.conditional_scope('num_layers', list(range(i+1, 4+1))):
            model.add(SimpleRNN(hp.Int(f'layer{i}_units',min_value=1,max_value=50,step=10),return_sequences=True))
    model.add(SimpleRNN(hp.Int('last_layer_neurons',min_value=1,max_value=50,step=10)))
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        print(hp.Boolean("dropout"))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(1))
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=hp_learning_rate),metrics = ['mse'])
    return model

tuner= RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=3 #change
        )


X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
Y_test = np.asarray(Y_test).astype(np.float32)
tuner.search(
        x=X_train,
        y=Y_train,
        epochs=50,
        batch_size=50,
        validation_data=(X_test,Y_test),
)
tuner.search_space_summary()
tuner.results_summary()
#best_hps=tuner.get_best_hyperparameters()[0]

best_model = tuner.get_best_models(num_models=1)[0]
Y_pred=best_model.predict(X_test)
print(Y_pred.shape)
plt.plot(Y_test,label="true")
plt.plot(Y_pred,label="predicted")
plt.show()
save_model(best_model,'SynDataModel.h5')
#dump(t_trans, open('target_scaler.pkl', 'wb'))
#dump(f_trans, open('feature_scaler.pkl', 'wb'))

