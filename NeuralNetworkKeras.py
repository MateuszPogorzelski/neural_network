'''
GUT Gdansk University of Technology 2023
-------------------------------------------------------------------------------
"Research and analysis of deep learning architectures and their impact 
 on the radiolocation system effectiveness in indoor environment."
     
@author: Pogorzelski Mateusz 176809
'''

from keras import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
from lib import data, metrics, pred_plot
import matplotlib
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Normalization


n_inputs = 3
n_outputs = 2
epochs = 500
patience = 50

def build_model(trial, param, norm, tuning):
    '''Return DNN model'''
    
    model = Sequential(name='MLP_model')
    model.add(norm)
        
    for i in range(param['n_layers']):
            if tuning == False: 
                units = param['n_units_{}'.format(i+1)] 
            else: 
                units = trial.suggest_int("n_units_{}".format(i+1),param['n_units_min'], param['n_units_max'])
                
            model.add(Dense(units=units, activation=param['activation'], name='dense_0{}'.format(i))) 
            model.add(Dropout(param['dropout']))
            
    model.add(Dense(n_outputs, activation='linear', name='dense_0{}'.format(i+1)))
    
    # complile model
    optimizer = getattr(tf.keras.optimizers, param['optimizer'])
    model.compile(loss='mae',
                  optimizer=optimizer(param['learning_rate']),
                  metrics=['mae', 'mse']
                  )
    
    return model


def objective(trial):
    '''Tuning hyperparameters in trails and return best evaluation score'''
    
    params = {
            'n_layers': trial.suggest_int("n_layers", 3, 10),
            'n_units_min': 8,
            'n_units_max': 128,
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'test_size': trial.suggest_categorical('test_size', [0.2]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'optimizer': trial.suggest_categorical('optimizer', ['Adam','RMSprop','SGD']),
            'activation': trial.suggest_categorical('activation', ['relu','tanh','sigmoid'])
            }
    
    # prepared data
    xtrain, xval, xtest, ytrain, yval, ytest = data(params['test_size'])
    # normalization
    xtrain_norm = normalization_NN(xtrain)
    # build model
    model = build_model(trial, 
                        params, 
                        xtrain_norm, 
                        tuning=True)
    # define callback
    callback_es = EarlyStopping(monitor='val_loss',
                                patience=patience,
                                mode='min')
    # fit model
    model.fit(
            xtrain,
            ytrain,
            validation_data=(xval, yval),
            batch_size=params['batch_size'],
            epochs=epochs,
            verbose=0,
            callbacks=[callback_es],
            workers = 2
            )
    
    # evaluate performance
    _,score,_ = model.evaluate(xtest, ytest, verbose=0, workers = 2)
    
    return score


def normalization_NN(xtrain):
    '''Return normalized data for NN'''
    xtrain_norm = Normalization(input_shape=[n_inputs,], axis=None)
    xtrain_norm.adapt(xtrain)
    
    return xtrain_norm


def neural_network_tunning(n_trials):
    '''Tuning hyperparameters and return best trial'''
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=20,), 
                                direction='minimize')
    study.optimize(lambda trial:objective(trial), n_trials=n_trials)
    best_trial = study.best_trial
    
    return best_trial, study
    

def neural_network_best_model(best_trial, epochs):
    '''Return value of evaluation, Y-parameters history of training''' 
    
    xtrain, xval, xtest, ytrain, yval, ytest = data(best_trial.params['test_size'])
    best_model = build_model(best_trial, best_trial.params, normalization_NN(xtrain), tuning=False)  
    best_model.save('models/MLP_model')
    best_model.summary()
    
    callback_es = EarlyStopping(monitor='val_loss',
                       patience=patience,
                       mode='min',
                       )
    
    callback_ms = ModelCheckpoint(monitor='val_loss',
                         filepath='best_model/best_model_ChP',
                         save_best_only=True,
                         save_weights_only=True,
                         mode='min',
                         verbose=0
                         )
    
    history = best_model.fit(xtrain,
                             ytrain,
                             validation_data=(xval, yval),
                             shuffle=True,
                             batch_size=best_trial.params['batch_size'],
                             epochs=epochs,
                             verbose=2,
                             callbacks=[callback_es, callback_ms]
                            )
    
    results = best_model.evaluate(xtest, ytest, verbose=0)
    pred = best_model.predict(xtest, verbose=0)
    Y = [ytrain, ytest, pred]
    
    weights = best_model.get_weights()
    best_model.save_weights('weights/MLP_weights', overwrite=True, 
                            save_format=None, options=None)
    
    return results, Y, history, weights
