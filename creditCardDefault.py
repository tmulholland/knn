import numpy as np
import knn
import time

def doFeatureScaling(data):
    """ Simple scaling to ensure each feature is between 0 and 1
    Get about a 5-6% increase in classification success rate
    """
    ## rescale features 
    for feature in range(len(data[0][:-1])):
        for datapoint in range(len(data[:,1])):
            Mean = data[:,feature].mean()
            Std = data[:,1].std()
            data[datapoint,feature] = (data[datapoint,feature]-Mean)/Std

## main script
## Data (were modified) downloaded from here:
## https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

## decide which processors to use (GPU, CPU, or both)
procList = ['CPU','GPU']
rateDict = dict.fromkeys(procList)

## number of training points (max=30000)
nPoints = 3000

doScaling = False ## scaling already done in preprocessing of data

## last column is classification, others are attributes
data = np.genfromtxt('CreditDefaultData.csv', delimiter=',',dtype='f')

## first 2 lines are descriptions (not if using preprocessed data)
#data = data[2:nPoints]
data = data[:nPoints]

if doScaling:
    doFeatureScaling(data)

#np.savetxt("CreditDefaultData.csv", data, delimiter=",")

## classification
dataCl = data[:,-1]

## attributes (skim off index and classification)
data = data[:, 1:-1]


nPts = len(data)

knn = knn.knn()
knn.k = 4

pretxt = str(nPoints)+" training data examples with "+str(len(data[0]))+" features \n"
print "# Credit card default prediction \n"
print pretxt

print "| Processor | Runtime | Correctly predicted | "
print "| ------------- |:----------------:| ---------------:|"
for proc in procList:
    errorRate = 0
    knn.procList = [proc]
    start_time = time.time()

    for point in range(1,nPts-1):

        dataPoint = data[point]

        ## subtract off test point
        knn.data = np.append(data[:][0:point],data[:][point+1:nPts],axis=0)
        knn.dataCl = np.append(dataCl[:][0:point],dataCl[:][point+1:nPts],axis=0)
        knn.testPt = dataPoint 
    
        result = knn.getPred()

        errorRate+=float(result==dataCl[point])

    timeToRun = time.time() - start_time 
    rateDict[proc] = timeToRun   
    timeUnit = ' seconds'
    if timeToRun>60:
        timeToRun*=1/60.
        timeUnit = ' minutes'
    if timeToRun>60:
        timeToRun*=1/60.
        timeUnit = ' hours'
        if timeToRun>24:
            timeToRun*=1/24.
            timeUnit = ' days'
    timeToRun = round(timeToRun,3)

    errorRate = errorRate/(nPts-2)
    print "| "+proc+" | "+str(timeToRun) +timeUnit+" | "+ str( round(errorRate*100,2)) + " percent |"


if ( 'GPU' in procList and 'CPU' in procList):
    perfIncrease = round(rateDict['CPU']/rateDict['GPU'],1)
    st = '\n### GPU processing was '+str(perfIncrease)+' times faster'

    print st
