'''
GUT Gdansk University of Technology 2023
-------------------------------------------------------------------------------
"Research and analysis of deep learning architectures and their impact 
 on the radiolocation system effectiveness in indoor environment."
     
@author: Pogorzelski Mateusz 176809
'''

import pandas as pd
import re

row = ['01','02','03','04','05','06','07','08','09']
col = ['01','02','03','04','05','06','07','08','09',
        '10','11','12','13','14','15','16','17','18']

mac = ['fa:92:bf:21:46:4d', 'fa:92:bf:21:46:58', 'fa:92:bf:21:46:c6']
path = 'pomiary/'


def prepare_one(nr):
    '''Return set for one location'''
    # read file, select only valid MAC, drop least important data
    df = pd.read_csv(path+str(nr)+'.csv', sep=';')
    df = df[df['mac'].isin(mac)]
    df_rssi = df['last 100 signal strength'].str.split(',', expand=True)
    df = df.drop(['ssid','frequency','frequency center 1', 'frequency center 2',
                  'signal strength','last 100 signal strength','Unnamed: 11',
                  'channelwidth_mhz','type','capabilities'], axis=1)
    
    df_rssi = df_rssi.transpose()
    
    # connect measurements form all APs to one 'df_set' file
    dfn = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    df_set = pd.DataFrame()
    
    for i in range(df_rssi.shape[1]):
        # replace '-' by mean of meadured values RSSI at the same point
        mean = 0
        counter = 0
        for j in range(df_rssi.shape[0]):
            if df_rssi.iloc[j,i] != '-' : 
                mean += int(df_rssi.iloc[j,i])
                counter += 1
        mean = round(mean / counter, 2)
        dfn[i] = df_rssi.iloc[:,i].replace('-', mean)
        dfn[i] = pd.to_numeric(dfn[i])
        
        df_set = pd.concat([df_set, dfn[i]], ignore_index=True, axis=1)
        
    # add coordinates from file's name
    match = re.match(r'([0-9]{2})([0-9]{2})', str(nr), re.I)
    if match: items = match.groups()
    df_set.insert(0,'y',int(items[0]))
    df_set.insert(1,'x',int(items[1]))
    df_set.columns = ['y', 'x', 'AP1', 'AP2', 'AP3']  

    return df_set


def prepare_set():
    '''Return prepared set'''
    df2 = pd.DataFrame()
    
    # connect all measurements to one 'set.csv' file
    for i in range(len(row)):
        for j in range (len(col)):
            try:
                df1 = prepare_one('{}{}'.format(row[i],col[j]))
                df2 = pd.concat([df2, df1], ignore_index=True)
            except: 
                pass
    
    df2.to_csv('set.csv', header=True, index=False)
    
    return df2
