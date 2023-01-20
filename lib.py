'''
GUT Gdansk University of Technology 2022
-------------------------------------------------------------------------------
"Research and analysis of deep learning architectures and their impact 
 on the radiolocation system effectiveness in indoor environment."
     
@author: Pogorzelski Mateusz 176809
'''

from prepared_data import prepare_set
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import locale

locale.setlocale(locale.LC_NUMERIC, "de_DE")
plt.rcdefaults()
plt.rcParams['axes.formatter.use_locale'] = True
matplotlib.rcParams.update({'font.size': 9, 'font.family': 'Arial', 
                              'mathtext.fontset': 'stix'})


def data(test_size):
    '''Return train, valid and test sets'''
    df_set = prepare_set()
    X = df_set.iloc[:,2:5]
    Y = df_set.iloc[:,0:2]
    
    # histograms
    fig, ax = plt.subplots(1,3, figsize=(8,3), dpi=900,sharey=True)
    sns.histplot(X.iloc[:,0], ax = ax[0], label='AP1')
    ax[0].grid()
    sns.histplot(X.iloc[:,1], ax = ax[1], label='AP2')
    ax[1].grid()
    sns.histplot(X.iloc[:,2], ax = ax[2], label='AP3')
    ax[2].grid()
    
    #box plot
    fig = plt.figure(figsize =(10, 6), dpi = 900)
    plt.boxplot(X.iloc[:,:])
    plt.xticks([1,2,3], ['AP1', 'AP2','AP3'])
    plt.xlabel('Punkty dostępowe')
    plt.ylabel('RSSI [dBm]')
    plt.grid()
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=2*test_size, 
                                                    random_state=42)
    xval, xtest, yval, ytest = train_test_split(xtest, ytest, test_size=0.5, 
                                                    random_state=42)
    
    return xtrain, xval, xtest, ytrain, yval, ytest
    

def EstimateRxOnTheFloor(pred, ytest, X, Y):
    '''Return predicted and real position plot on the floor plot'''

    arr = np.concatenate((pred, ytest), axis=1)
    
    fig, ax = plt.subplots(1,1, figsize = (8,4), dpi=900)
    linewidth = 3
    color_wall = 'k'
    color_glass = '#AAAAAA'
    
    # y lines
    plt.plot([0.5, 0.5], [2.5, 9.5],   color = color_wall, linewidth = linewidth)
    plt.plot([17.5, 17.5], [9.5, 7.5], color = color_wall, linewidth = linewidth)
    plt.plot([18.5, 18.5], [4.5, 7.5], color = color_wall, linewidth = linewidth)
    plt.plot([17.5, 17.5], [1.5, 4.5], color = color_wall, linewidth = linewidth)
    plt.plot([14.5, 14.5], [0.5, 1.5], color = color_wall, linewidth = linewidth)
    plt.plot([10.5, 10.5], [0.5, 9.5], color = color_wall, linewidth = linewidth)
    plt.plot([12, 12], [6.5, 9.5],     color = color_wall, linewidth = linewidth)
    plt.plot([13.5, 13.5], [6.5, 9.5], color = color_wall, linewidth = linewidth)
    plt.plot([6.5, 6.5], [2.5, 6.5],   color = color_wall, linewidth = linewidth)
    plt.plot([14.5, 14.5], [4.5, 8.5], color = color_glass,linewidth = linewidth)
    
    # x lines
    plt.plot([0.5, 17.5], [9.5, 9.5], color = color_wall, linewidth = linewidth)
    plt.plot([17.5, 18.5], [7.5, 7.5], color = color_wall, linewidth = linewidth)
    plt.plot([14.5, 17.5], [8.5, 8.5], color = color_wall, linewidth = linewidth)
    plt.plot([14.5, 18.5], [4.5, 4.5], color = color_wall, linewidth = linewidth)
    plt.plot([14.5, 17.5], [1.5, 1.5], color = color_wall, linewidth = linewidth)
    plt.plot([10.5, 17.5], [4.5, 4.5], color = color_wall, linewidth = linewidth)
    plt.plot([10.5, 14.5], [0.5, 0.5], color = color_wall, linewidth = linewidth)
    plt.plot([0.5, 10.5], [2.5, 2.5], color = color_wall, linewidth = linewidth)
    plt.plot([0.5, 13.5], [6.5, 6.5], color = color_wall, linewidth = linewidth)
    
    # cell
    # plt.plot([1.5, 2.5], [3.5, 3.5], color = 'b', linewidth = 2, ls = '--')
    # plt.plot([1.5, 2.5], [4.5, 4.5], color = 'b', linewidth = 2, ls = '--')
    # plt.plot([1.5, 1.5], [3.5, 4.5], color = 'b', linewidth = 2, ls = '--')
    # plt.plot([2.5, 2.5], [3.5, 4.5], color = 'b', linewidth = 2, ls = '--')
    # plt.text(3, 4, 'obszar (2,4)')
    
    # APs localization
    AP = pd.DataFrame([[5.5,12],[5,8],[8,8]])
    plt.scatter(AP.iloc[:2,1], AP.iloc[:2,0], label="Punkt dostępowy AP piętro I", 
                linewidth = linewidth-1, marker='D', c='r' )
    plt.scatter(AP.iloc[2,1], AP.iloc[2,0], label="Punkt dostępowy AP parter", 
                linewidth = linewidth-1, marker='D', c='b' )
    plt.text(12.25, 5.25, 'AP1')
    plt.text(8.25, 4.75, 'AP2')
    plt.text(8.25, 7.75, 'AP3')
    
    ax.set_ylabel('y [m]', rotation = 'horizontal', labelpad = 10)
    ax.set_xlabel('x [m]', rotation = 'horizontal', labelpad = 10)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top') 
    plt.xticks(np.arange(0, 19, 1))
    plt.yticks(np.arange(0, 10, 1))
    
    plt.grid()
    ax.invert_yaxis()
    
    # stairs
    for i in range(9):
        plt.plot([13.5, 14.4], [6.5+i*0.25, 6.5+i*0.25], color = color_wall, linewidth = 1)
        plt.plot([14.5+i*0.25, 14.5+i*0.25], [8.5, 9.4], color = color_wall, linewidth = 1)
    
    plt.arrow(14,6.5,0,1.75,width=0.05)
    plt.arrow(14.5,9,1.75,0,width=0.05)
    
    # draw plot
    for i in range(len(X)):
        newArrY = arr[arr[:,2] == Y[i]]
        newArr = newArrY[newArrY[:,3] == X[i]]
    
        plt.scatter(newArr[:,1], newArr[:,0], label=f'Przewidywana lokalizacja terminala ruchomego ({X[i]}, {Y[i]})', 
                    linewidth = linewidth-1 , marker='o')
        plt.scatter(newArr[:,3], newArr[:,2], label=f'Rzeczywista lokalizacja terminala ruchomego ({X[i]}, {Y[i]})', 
                    linewidth = linewidth-1, marker='D', )
    
    plt.legend(bbox_to_anchor=(0.57, 0))    
            
    
def pred_plot(ytest, pred, wsp):
    '''Return predicted and real position plot'''
    
    plt.figure(figsize=(5,5), dpi = 900)
    
    if wsp == 'x':
        N = ytest.iloc[:,1].max()
        plt.scatter(ytest.iloc[:,1], pred.iloc[:,1], marker='x', c='orange', linewidth=1)
    elif wsp =='y':
        N = ytest.iloc[:,0].max()
        plt.scatter(ytest.iloc[:,0], pred.iloc[:,0], marker='x', c='g', linewidth=1)
    plt.xlim(0,N+1)
    plt.ylim(0,N+1)
    plt.xlabel('Rzeczywista lokalizacja terminala ruchomego [m]')
    plt.ylabel('Przewidywana lokalizacja terminala ruchomego [m]')
    plt.xticks(np.arange(0, N+1, 1))
    plt.yticks(np.arange(0, N+1, 1))
    plt.plot([0, N], [0, N])
    

def metrics(ytest, pred, make_hist):
    '''Return statisctics, histograms and distribuant'''
    d = (ytest.iloc[:,1].to_numpy() - pred.iloc[:,1].to_numpy())**2 + ((ytest.iloc[:,0].to_numpy() - pred.iloc[:,0].to_numpy()))**2
    d_hist = d**(1/2) 
    x_hist = np.abs((ytest.iloc[:,1].to_numpy() - pred.iloc[:,1].to_numpy()))
    y_hist = np.abs((ytest.iloc[:,0].to_numpy() - pred.iloc[:,0].to_numpy()))
    
    d_mse = ((d_hist - d_hist.mean())**2)
    x_mse = ((x_hist - x_hist.mean())**2)
    y_mse = ((y_hist - y_hist.mean())**2)
    
    RupX = (ytest.iloc[:,1].to_numpy() - pred.iloc[:,1].to_numpy())**2
    RdownX = (ytest.iloc[:,1].to_numpy() - ytest.iloc[:,1].to_numpy().mean())**2
    
    R = 1 - RupX.sum()/RdownX.sum()
    R_real_x = R - 2/(len(ytest)-2-1)*(1-R)
    
    print(f'x = {x_hist.mean():.2f} +- {x_hist.std():.2f}')
    print(f'y = { y_hist.mean():.2f} +- { y_hist.std():.2f}')
    print(f'd = {d_hist.mean():.2f} +- {d_hist.std():.2f}\n')
    
    print(f'x_mse = {x_mse.mean():.2f} +- {x_mse.std():.2f}')
    print(f'y_mse = {y_mse.mean():.2f} +- {y_mse.std():.2f}')
    print(f'd_mse = {d_mse.mean():.2f} +- {d_mse.std():.2f}\n')

    if make_hist:
        fig, ax = plt.subplots(1,3, figsize = (10,5), dpi=900, sharey=True)
        
        ax[0].hist(x_hist, 50, facecolor='orange', alpha = 0.75, edgecolor='k')
        ax[0].set_xlabel('MAE x [m]')
        ax[0].set_ylabel('Częstoć')
        ax[0].grid()
        
        ax[1].hist(y_hist, 50, facecolor='g', alpha = 0.75, edgecolor='k')
        ax[1].set_xlabel('MAE y [m]')
        ax[1].grid()
        
        ax[2].hist(d_hist, 50, facecolor='b', alpha = 0.75, edgecolor='k')
        ax[2].grid()
        ax[2].set_xlabel('MAE [m]')
    
    # distribuant
    plt.figure(figsize=(8,5), dpi = 900)
    
    from matplotlib import pyplot
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf_d = ECDF(d_hist)
    ecdf_x = ECDF(x_hist)
    ecdf_y = ECDF(y_hist)
    
    for i in range(1,9):
        print(f'P(x<{i}): %.3f' % ecdf_y(i))

    plt.plot(ecdf_d.x, ecdf_d.y*100)
    plt.plot(ecdf_x.x, ecdf_x.y*100)
    plt.plot(ecdf_y.x, ecdf_y.y*100)
    plt.xlabel('Błąd bezwzględny położenia [m]')
    plt.ylabel('Estymata dystrybuanty [%]')
    plt.grid()
    plt.legend(labels=['Estymata dystrybuanty d', 'Estymata dystrybuanty w osi x', 'Estymata dystrybuanty w osi y'])
        
    return x_hist, y_hist, d_hist


def plot_loss(history_df, N, M, R, criterion):
    '''Return plot of training and valid losses'''
    xstep = 50
    
    if criterion == 'mae':
        ystep = 0.1
        ylabel = 'MAE [m]'

    elif criterion == 'mse':
        ystep = 0.5
        ylabel = 'MSE [m$^2$]'
    
    running_median = history_df[criterion].rolling(R).mean()
    running_median2 = history_df[f'val_{criterion}'].rolling(R).mean()
    matplotlib.rcParams.update({'font.size': 9, 'font.family': 'Arial', 
                              'mathtext.fontset': 'stix'})
    plt.figure(figsize=(8,5), dpi=900)     
    
    plt.plot(history_df[criterion], label='zbiór treningowy', c='#AAAAAA')
    plt.plot(history_df[f'val_{criterion}'], label='zbiór walidacyjny', c='#DDDDDD')
    plt.plot(running_median, label='mediana zbiór treningowy', linestyle='-')
    plt.plot(running_median2, label='mediana zbiór walidacyjny', linestyle='-')

    plt.ylim([N, M])
    plt.xlim([0, len(history_df)])
    plt.xticks(np.arange(0, len(history_df)+xstep, xstep))
    plt.yticks(np.arange(N, M+ystep, ystep))
    plt.xlabel('numer iteracji')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    
