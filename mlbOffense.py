#!/usr/bin/python
import knn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def procYear(player, kNeighbors=4, year=''):
    """ takes as input dataframe from year specific FanGraphs Leaderboard, 
    year (blank = all), player, and the number of neighbors in cluster
    returns cluster of players
    """
    ## load the knn clustering class object
    knnObj = knn.knn()
    knnObj.k = kNeighbors

    knnObj.procList = ['CPU'] ## Choose CPU or GPU distance calculations

    ## read fangraphs csv into dataframe
    ## downloaded from https://www.fangraphs.com/
    df = pd.read_csv('FanGraphsLeaderboard'+str(year)+'.csv')

    ## remove name artifact
    df = df.rename(columns={'\xef\xbb\xbf"Name"':'name'})

    ## handle % in floating point values
    for col in df.columns:
        if "%" in col:
            ## chop off ' %'
            df[col] = df.apply(lambda x: float(x[col][:-2]), axis=1)

    ## use player name as id
    dataCl = np.array(df.name)
    
    ## columns to remove from clustering
    drop_cols = ['Team','playerid','name']

    data = np.column_stack([df[col] for col in df.columns if col not in drop_cols])

    ## handle wide data edge case
    if len(data)<=knnObj.k:
        return [0]

    ## feature scaling
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    nPts = len(data)

    if player not in dataCl:
        return [0]

    point = list(dataCl).index(player)
    dataPoint = data[point]

    ## subtract off test point
    knnObj.data = np.append(data[:][0:point],data[:][point+1:nPts],axis=0)
    knnObj.dataCl = np.append(dataCl[:][0:point],dataCl[:][point+1:nPts],axis=0)

    knnObj.testPt = dataPoint 

    result = knnObj.getCluster()

    return result



kNeighbors = 4

print "# MLB offensive statistics cluster"

print "### Enter season from 2000-2018 (enter -1 to run over all seasons)"
## which year to run on
year = int(raw_input().strip())

print "### Enter player name (enter -1 to run over all players)"
player = raw_input().strip()

if player != '-1':

    result = procYear(player, kNeighbors, year)

    print " "

    print " "

    print "The", kNeighbors, "nearest neighbors in cluster are:"

    print " "

    for neighbor in result:
        print neighbor+", "

else:
    ## could export this to csv for lookup without compilation
    for player in np.array(df.name):
        result = procYear(player, kNeighbors)
    
