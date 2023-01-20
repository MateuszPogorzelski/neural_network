'''
GUT Gdansk University of Technology 2022
-------------------------------------------------------------------------------
"Research and analysis of deep learning architectures and their impact 
 on the radiolocation system effectiveness in indoor environment."
     
@author: Pogorzelski Mateusz 176809
'''

from NeuralNetworkKeras import neural_network_tunning, neural_network_best_model
from lib import metrics, plot_loss, pred_plot, EstimateRxOnTheFloor
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import time

epochs = 500
n_trials = 50
random_state = 42



# %% MLP tuning and training
start1 = time.time()
best_trial, study = neural_network_tunning(n_trials)
end1 = time.time()

start2 = time.time()

results, Y, history, weights = neural_network_best_model(best_trial, epochs)

end2 = time.time()
total_time1 = end1 - start1
total_time2 = end2 - start2

print('\nCzas optymizacji MLP: {:.2f} sec'.format(total_time1))
print('\nCzas treningu MLP: {:.2f} sec'.format(total_time2))

print('\nNumber of finished trials: {}'.format(len(study.trials)))
print('Loss: {:.2f}'.format(best_trial.value))
print('\nParams: ')
for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
    
# %% compute metrics and histograms
# [ytrain, ytest, ypred] = Y[1, 2, 3]
Y[2] = pd.DataFrame(Y[2])
Y[1] = pd.DataFrame(Y[1])
x_hist, y_hist, d_hist = metrics(Y[1], Y[2], make_hist = True)

# %%
history_df = pd.DataFrame(history.history)
print(len(history_df))
# history_df.to_csv('history_df_MLP.csv', header=True, index=False)

#%% create predictions' and losses' plots
plot_loss(history_df, 1.7, 3, 15, criterion='mae')
plot_loss(history_df, 7, 13, 15, criterion='mse')
pred_plot(Y[1], Y[2], 'x')
pred_plot(Y[1], Y[2], 'y')

# %% show estimate localizations on the floor
x = [12, 14, 7, 2, 6]
y = [3, 8, 3, 4, 8]
EstimateRxOnTheFloor(Y[2], Y[1], x, y)

# %% change hiperparameters and research their impact on MAE
epochs = len(history_df)
prepared_trial_H = best_trial

# %% Reaserch impact of HP

# manual way

# prepared_trial_H = { 'n_layers': 3,
#                       'batch_size': 64,
#                       'test_size': 0.2,
#                       'learning_rate': 0.002
#                       'dropout': 0.23,
#                       'optimizer': 'RMSprop',
#                       'activation': 'relu',
#                       'n_units_1': 92,
#                       'n_units_2': 78,
#                       'n_units_3': 48

#                     }

# semi-automat way
# uncomment one-by-one to reaserch impact of HP

batch_size = [16, 48, 80, 128]
# learning_rate = [0.00002, 0.0002, 0.02, 0.2]
# dropout = [0.05, 0.1, 0.3, 0.5]
# optimizer = ['Adam','SGD']
# activation = ['tanh','sigmoid']
# units = [[61, 51, 34],[92, 78, 48],[153, 129, 78], [244, 206, 136]]
# layers = [[122, 103], [68, 92], [78, 48, 34]]
variable = batch_size

fig, ax = plt.subplots(1,2, figsize = (15,5), dpi=900, sharey = True)

for i in range(2):
    ax[i].set_ylabel('MAE [m]')
    ax[i].set_xlabel('numer iteracji')
    ax[i].grid()
    ax[i].set_xlim([0, epochs])
    ax[i].set_ylim([1.6, 2.7])

ax[0].plot(history_df['mae'].rolling(15).mean(), 
           label="prepared_trial_H.params['batch_size']", c ='r')
ax[1].plot(history_df['val_mae'].rolling(15).mean(), 
           label="prepared_trial_H.params['batch_size']", c = 'r')

for i, data in enumerate(variable):
    
    # uncomment to reaserch impact of layers
    
    # if i==0:
    #     prepared_trial_H.params['n_units_4'] = layers[i][0]
    #     prepared_trial_H.params['n_units_5'] = layers[i][1]
        
    # if i==1:
    #     prepared_trial_H.params['n_units_6'] = layers[i][0]
    #     prepared_trial_H.params['n_units_7'] = layers[i][1]
        
    # if i==2:
    #     prepared_trial_H.params['n_units_8'] = layers[i][0]
    #     prepared_trial_H.params['n_units_9'] = layers[i][1]
    #     prepared_trial_H.params['n_units_10'] = layers[i][2]
    
    resultsH, YH, historyH, weightsH = neural_network_best_model(prepared_trial_H, epochs)
    
    YH[2] = pd.DataFrame(YH[2])
    YH[1] = pd.DataFrame(YH[1])
    x_histH, y_histH, d_histH = metrics(YH[1], YH[2], make_hist = True)
    
    historyH_df = pd.DataFrame(historyH.history)
    historyH_df.to_csv('history_H5.csv', header=True, index=False)
    ax[0].plot(historyH_df['mae'].rolling(15).mean(), linestyle = '--')
    ax[1].plot(historyH_df['val_mae'].rolling(15).mean(), linestyle = '--')
    ax[0].legend()
    ax[1].legend()              