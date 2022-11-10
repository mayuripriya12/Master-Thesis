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
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.layers import Dropout
from keras.utils.vis_utils import plot_model


headers = ['Timestamp', 'Signal']
#df = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\Ring Resonator\\i1.txt',skiprows=range(0,3),
                 #names= headers, sep= '\t', float_precision=None)
df = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFCombline\\reflected.csv',skiprows=range(0,1608),
                 names= headers, sep= '\t', float_precision=None, nrows = 2*804)
df.head(5)
print(len(df))
print(df[:5])
df.plot(y = 'Signal', use_index = True )
plt.show()

time_col = df['Timestamp'].to_numpy()
signal_col = df['Signal'].to_numpy()
n = len(df)
test_size = 0.25
n_prev = 20


def _load_data(data, n_prev = 100):  

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].to_numpy())
        docY.append(data.iloc[i+n_prev].to_numpy())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def _divide_data(dataset, length_of_sequences, test_size = 0.25):
    ntr = int(len(dataset) * (1 - test_size))
    df_train = dataset[["Signal"]].iloc[:ntr]
    df_test  = dataset[["Signal"]].iloc[ntr:]
    (X_train, y_train) = _load_data(df_train, n_prev = length_of_sequences)
    (X_test, y_test)   = _load_data(df_test, n_prev = length_of_sequences)

    return X_train, y_train, X_test, y_test

def define_model(length_of_sequences, batch_size = None, stateful = False):
    in_out_neurons = 1
    hidden_neurons = 20
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons))  

    rnn = SimpleRNN(41, 
                   return_sequences=True,
                    stateful = stateful,
                    name="RNN")(inp)
    rnn = SimpleRNN(11, 
                    stateful = stateful,
                    name="RNN2")(rnn)

    dens = Dense(in_out_neurons,name="dense")(rnn)
    model = Model(inputs=[inp],outputs=[dens])

    #opt = RMSprop(
    #learning_rate=0.00001,
    #rho=0.9)
    opt = Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=opt)
    return(model,(inp,rnn,dens))

my_array = [df]
ind = 0
#for x in my_array:

while ind < len(my_array):
    x = my_array[ind]
    (X_train, y_train, X_test, y_test) = _divide_data(x, 20)
    print(X_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model, (inp,rnn,dens) = define_model(length_of_sequences = X_train.shape[1])
    plot_model(model, to_file='Sine_model.png', show_shapes=True, show_layer_names=True)
    model.summary()
    X_train=np.asarray(X_train).astype(float)
    y_train=np.asarray(y_train).astype(float)
    hist = model.fit(X_train, y_train, batch_size=600, epochs=1000, 
                 verbose=True,validation_split=0.25)
    model.save("Transmitted_model"+str(ind)+".h5")
    for label in ["loss","val_loss"]:
        plt.plot(hist.history[label],label=label)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
    plt.legend()
    plt.yscale('log') 
    plt.savefig("validation"+str(ind)+".png")
    plt.clf()
    plt.cla()

    X_test = np.asarray(X_test).astype(float)
    
    y_pred = model.predict(X_test)

    
    val_loss = 0
    for i in range(len(y_test)):
        val_loss = val_loss + ((y_test[i] - y_pred[i])**2)

    result = val_loss / len(y_test)
    print("validation loss is:", result)

    plt.plot(y_test,label="true")
    plt.plot(y_pred,label="predicted")
   
    #t_range = list(range(x_range + n_prev, n))
    #t_rang1 = map(float, t_range)


    

    #plt.ylim(-0.11,0.11)

    
    plt.legend()
    plt.savefig("prediction"+str(ind)+".png")
    plt.clf()
    plt.cla()
    ind += 1
