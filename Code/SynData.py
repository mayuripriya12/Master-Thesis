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
from keras.models import Sequential
from tensorflow.keras.layers import Dropout
from keras.utils.vis_utils import plot_model


df01 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
df02 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df03 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df04 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
#df05 = pd.DataFrame(columns = ['Timestamp', 'Signal'])
n = 8000
test_size = 0.25
n_prev = 20

for t in range(n):

    #Y = (sin(10 * 0.01 * t) * exp(-0.1 * 0.01 * t))
    y = np.sin(18*0.01*t) * np.exp(-0.1 * 0.01 * t)
    #y = np.sin(18*0.01*t) 
    
    #Z = (sin(8 * 0.01 * t) * exp(-0.2 * 0.01 * t))
    z = np.sin(15*0.01*t) * np.exp(-0.1*0.01*t)
    #z = np.sin(15*0.01*t) 
    
    #X = (sin(6 * 0.01 * t) * exp(-0.2 * 0.01 * t))
    x = np.sin(12*0.01*t) * np.exp(-0.1*0.01*t)

    #m = np.sin(9*0.01*t) * np.exp(-0.05*0.01*t)

    
    p = y + z + x
     
    q = p * 1.33013790e+00 * np.exp(1.07538112e-03*t)
    
    df01 = pd.concat([df01, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':p }])], ignore_index=True)
    df02 = pd.concat([df02, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':q }])], ignore_index=True)
    #df03 = pd.concat([df03, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':x }])], ignore_index=True)
    #df04 = pd.concat([df04, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':p }])], ignore_index=True)
    #df05 = pd.concat([df05, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':q }])], ignore_index=True)

plt.plot(df01['Timestamp'].iloc[1:3000], df01['Signal'].iloc[1:3000], 'b')
plt.show()
plt.plot(df02['Timestamp'].iloc[1:3000], df02['Signal'].iloc[1:3000], 'r')
plt.show()
#my_array = [ df01, df02, df03, df04, df05]
my_array = [df01, df02]

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

#def define_model():
    #model = Sequential()
    #model.add(SimpleRNN(1,return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
    #model.add(SimpleRNN(11,return_sequences=True))
    #model.add(SimpleRNN(31))
    #model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer="adam")
    #return model

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
   
    #t_range = list(range(x_range + n_prev, n))
    #t_rang1 = map(float, t_range)

    if(ind != 0):
        y_test1=[]
        y_pred1=[]
        i=0
        x_range = int((n * (1-test_size)))
        print(x_range)
        for t in range(x_range + n_prev, n):
            y_pred1.append( (1/1.33013790e+00) * np.exp(-1.07538112e-03*t) * y_pred[i])
            y_test1.append((1/1.33013790e+00) * np.exp(-1.07538112e-03*t) * y_test[i])
            i = i+1
        print(y_pred.shape)
        plt.plot(y_test1,label="true")
        plt.plot(y_pred1,label="predicted")

    else:

        plt.plot(y_test,label="true")
        plt.plot(y_pred,label="predicted")

    #plt.ylim(-0.11,0.11)

    
    plt.legend()
    plt.savefig("prediction"+str(ind)+".png")
    plt.clf()
    plt.cla()
    ind += 1

