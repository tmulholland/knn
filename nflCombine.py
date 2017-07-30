#!/usr/bin/python
import knn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def procDraft(df, year, player, kNeighbors):
    """ takes as input dataframe from combine.csv, 
    year (-1 = all), player, and the number of neighbors in cluster
    returns cluster of players
    """
    ## load the knn clustering class object
    knnObj = knn.knn()
    knnObj.k = kNeighbors

    knnObj.procList = ['GPU'] ## Choose CPU or GPU distance calculations

    pos = df[df.name==player].position.values[0]

    ## select on position as slice
    if year>0:
        df = df[df.year==year]

    ## select on position as slice
    df = df[df.position==pos]

    ## remove entries with zero value
    df = df[(df.fortyyd>0) & (df.broad>0) & (df.vertical>0) & (df.twentyss>0)]

    
    ## not many QBs or WRs were tested on bench
    if (pos!='QB' or pos!='WR'):
        df = df[df.bench>0]

    ## use player name as id
    dataCl = np.array(df.name)
    
    data = np.column_stack((df.heightinchestotal, df.weight, 
                            df.fortyyd, df.broad, df.vertical, df.twentyss))

    if (pos!='QB' or pos!='WR'):
        data = np.column_stack((data, df.bench))

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



## pandas dataframe 
## imported from csv file downloaded from http://nflsavant.com/about.php
df = pd.read_csv('combine.csv', error_bad_lines=False,  warn_bad_lines=False)

kNeighbors = 4

print "# NFL draft cluster"

print "### Enter draft year from 1999-2015 (enter -1 to run over all drafts)"
## which draft year to run on
year = int(raw_input().strip())

print "### Enter player name (enter -1 to run over all players)"
player = raw_input().strip()

if player != '-1':

    result = procDraft(df, year, player, kNeighbors)

    print "The", kNeighbors, "nearest neighbors in cluster are:"
    for neighbor in result:
        print neighbor

else:
    ## could export this to csv for lookup without compilation
    for player in np.array(df.name):
        result = procDraft(df, -1, player, kNeighbors)
    
