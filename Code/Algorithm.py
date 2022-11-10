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
df01 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFWaveguide\\i1.txt',skiprows=range(0,3),
                 names= headers, sep= '\t', float_precision=None)
maxValueIndexObj = df01['Signal'].idxmax()

print("Max values of column is at row index position :")
print(maxValueIndexObj)

#The pulse width is twice the max value index
pulseWidth = maxValueIndexObj * 2
print(pulseWidth)

#The window length is twice the pulse width
windowLength = 2* pulseWidth

df02 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFWaveguide\\reflected.csv',
                 names= headers, sep= '\t', float_precision=None, nrows = windowLength)

#time_col = df['Timestamp'].to_numpy()
#signal_col = df['Signal'].to_numpy()
#n = len(df)
test_size = 0.25
n_prev = 20

def _arrange_data(data, prev_step = 10):  

    dataX, dataY = [], []
    for i in range(len(data)-prev_step):
        dataX.append(data.iloc[i:i+prev_step].to_numpy())
        dataY.append(data.iloc[i+prev_step].to_numpy())
    arrX = np.array(dataX)
    arrY = np.array(dataY)

    return arrX, arrY

def _arrange_data2(data, prev_step = 10):  

    docX, docY = [], []
    for i in range(len(data)-prev_step):
        docX.append(data[i:i+prev_step])
        docY.append(data[i+prev_step])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def _divide_data(dataset, length_of_sequences, test_size = 0.25):
    ntr = int(len(dataset) * (1 - test_size))
    df_train = dataset[["Signal"]].iloc[:ntr]
    df_test  = dataset[["Signal"]].iloc[ntr:]
    (X_train, y_train) = _arrange_data(df_train, prev_step = length_of_sequences)
    (X_test, y_test)   = _arrange_data(df_test, prev_step = length_of_sequences)

    return X_train, y_train, X_test, y_test

def define_model(length_of_timesteps, batch_size = None, stateful = False):
    input_features = 1
    inp = Input(batch_shape=(batch_size, 
                length_of_timesteps, 
                input_features))  

    rnn = SimpleRNN(41, 
                   return_sequences=True,
                    stateful = stateful,
                    name="RNN")(inp)
    rnn = SimpleRNN(11, 
                    stateful = stateful,
                    name="RNN2")(rnn)

    dens = Dense(input_features,name="dense")(rnn)
    model = Model(inputs=[inp],outputs=[dens])

    opt = Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=opt)
    return(model,(inp,rnn,dens))

(X_train, y_train, X_test, y_test) = _divide_data(df02, 20)
print(X_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
model, (inp,rnn,dens) = define_model(length_of_timesteps = X_train.shape[1])
model.summary()
X_train=np.asarray(X_train).astype(float)
y_train=np.asarray(y_train).astype(float)
hist = model.fit(X_train, y_train, batch_size=600, epochs=1000, 
                 verbose=True,validation_split=0.25)
validation_loss_test  = hist.history["val_loss"][-1]
print("val loss from test data is")
print(validation_loss_test) 
wl = 0
threshold = 0.000001
while(1):
    if(validation_loss_test >= threshold):
        wl = wl + windowLength
        df03 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFWaveguide\\reflected.csv',skiprows=range(0,wl),
                     names= headers, sep= '\t', float_precision=None, nrows = windowLength)
        (X_train, y_train, X_test, y_test) = _divide_data(df03, 20)
        model, (inp,rnn,dens) = define_model(length_of_timesteps = X_train.shape[1])
        model.summary()
        X_train=np.asarray(X_train).astype(float)
        y_train=np.asarray(y_train).astype(float)
        hist = model.fit(X_train, y_train, batch_size=600, epochs=1000, 
                     verbose=True,validation_split=0.25)
        validation_loss_test = hist.history["val_loss"][-1]
        print("val loss from test data from first if condn is")
        print(validation_loss_test) 
    else:
    
        for label in ["loss","val_loss"]:
            plt.plot(hist.history[label],label=label)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
        plt.legend()
        plt.yscale('log') 
        plt.savefig("validation.png")
        plt.clf()
        plt.cla()

        X_test = np.asarray(X_test).astype(float)
        
        y_pred = model.predict(X_test)

        plt.plot(y_test,label="true")
        plt.plot(y_pred,label="predicted")
        plt.legend()
        plt.savefig("prediction.png")
        plt.clf()
        plt.cla()
       
        df03 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFWaveguide\\reflected.csv',skiprows=range(0,wl),
                     names= headers, sep= '\t', float_precision=None, nrows = 0.25 * windowLength)
        df_test = df03['Signal'].to_numpy()
        (X_test2, Y_test2) = _arrange_data2(df_test, prev_step = 20)


        #model = load_model('Transmitted_model0.h5')
        #model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        X_test2 = np.asarray(X_test2).astype(float)
        y_pred2 = model.predict(X_test2)

        val_loss = 0
        for i in range(len(Y_test2)):
            val_loss = val_loss + ((Y_test2[i] - y_pred2[i])**2)
        result = val_loss / len(Y_test2)
        print("validation loss is:", result)

        if(result <= validation_loss_test ):
            model.save("Transmitted_model.h5")
            df04 = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\BPFWaveguide\\reflected.csv',skiprows=range(0,wl + len(df03)),
                     names= headers, sep= '\t', float_precision=None)
            df_test2 = df04['Signal'].to_numpy()
            (X_test3, Y_test3) = _arrange_data2(df_test2, prev_step = 20)


            #model = load_model('Transmitted_model0.h5')
            #model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            X_test3 = np.asarray(X_test3).astype(float)
            y_pred3 = model.predict(X_test3)
            plt.plot(Y_test3,label="true", linewidth = 3)
            plt.plot(y_pred3,label="predicted")
            plt.legend()
            plt.savefig("predictionEntireSignal.png")
            plt.clf()
            plt.cla()
            exit()
        else:
            validation_loss_test = threshold
            print("val loss from test data from else condn is")
            print(validation_loss_test) 
    
            
        




    
    
