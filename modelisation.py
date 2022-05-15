import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation
from keras.layers import concatenate, add, Lambda
from keras.models import Model, Sequential
import keras.backend as K

import datetime
import numpy as np

from datasets import create_test_data


#LSTNet model avec des portes GRU
def LSTNet(history_weeks, horizon_weeks, hidC = 30,Ck = 3,hidR = 20,hidS = 30,skip = 15,loss = "mse",pt =3):

    x = Input(shape=(history_weeks, 3))

    # CNN
    c = Conv1D(hidC, Ck, activation='relu', use_bias = True, kernel_regularizer = keras.regularizers.L1(0.005))(x)

    # RNN
    r = GRU(hidR, kernel_regularizer = keras.regularizers.L1(0.005))(c)
    r = Lambda(lambda k: K.reshape(k, (-1, hidR)))(r)

    # skip-RNN
    if skip > 0:
        # Reshape the data to easily setup the skip-RNN
        s = Lambda(lambda k: k[:, int(-pt*skip):, :])(c)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, skip, hidC)))(s)
        s = Lambda(lambda k: K.permute_dimensions(k, (0,2,1,3)))(s)
        s = Lambda(lambda k: K.reshape(k, (-1, pt, hidC)))(s)

        s = GRU(hidS, kernel_regularizer = keras.regularizers.L1(0.005))(s)
        s = Lambda(lambda k: K.reshape(k, (-1, skip*hidS)))(s)
        r = concatenate([r,s])
    #Sortie
    res = Dense(horizon_weeks, kernel_regularizer = keras.regularizers.L1(0.005))(r)
    res = Activation("relu")(res)
    
    model = Model(inputs=x, outputs=res)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics = [keras.metrics.mse])
    return model


#Evaluation du modèle
import datetime
def simulate(strategy, df, horizon_days, history_days, datestart, model,gap = 7):
    last_date = list(df.Date)[-1]
    date = datestart
    gain = 0
    #Simule les gains en suivant une stratégie, semaine après semaine
    while (date + datetime.timedelta(days = horizon_days)) < last_date:
        #Récupère et formate les données
        X_test, y_test = create_test_data(df, history_days=history_days, horizon_days=horizon_days, date = date)
        y_test = y_test[0,0]
        
        #Incrémente le temps
        date += datetime.timedelta(days = gap)
        
        #Récupère les prédictions du modèle
        y_pred = model(X_test)[0][0].numpy()
        
        #Evalue la décision prise par la stratégie
        bet =  strategy(y_pred, X_test)
        last_VIX = X_test[0,int(history_days/7 - 1),1]
        
        #Evalue les gains
        if bet == -1:
            #print("Vous avez parié à la baisse et gagné :", last_VIX - y_test)
            gain += last_VIX - y_test
        if bet == 1:
            #print("Vous avez parié à la hausse et gagné :", y_test - last_VIX)
            gain += y_test - last_VIX
    return gain

