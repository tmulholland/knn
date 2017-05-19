import numpy as np
import knn
import time

def doFeatureScaling(data):
    """ Simple scaling to ensure each feature is between 0 and 1
    Get about a 5-6% increase in classification success rate
    """
    ## rescale features 
    for feature in range(len(data[0][:])):
        for datapoint in range(len(data[:,1])):
            theMax = max(data[:,feature])
            data[datapoint,feature]*=1./theMax

## main script
## Data (were modified) downloaded from here:
## https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

## decide which processors to use (GPU, CPU, or both)
procList = ['CPU','GPU']
rateDict = dict.fromkeys(procList)

## number of training points (max=30000)
nPoints = 1000

doScaling = False ## scaling already done in  preprocessing of data

## last column is classification, others are attributes
data = np.genfromtxt('CreditDefaultData.csv', delimiter=',',dtype='f')

## first 2 lines are descriptions
data = data[2:nPoints]

if doScaling:
    doFeatureScaling(data)

#np.savetxt("CreditDefaultData.csv", data, delimiter=",")

## classification
dataCl = data[:,-1]

## attributes
data = data[:, 1:-1]


nPts = len(data)

## for large datasets,
## generally seem to get better classification if k is larger
knn = knn.knn()
knn.k = 10

pretxt = str(nPoints)+" training data examples with "+str(len(data[0]))+" features"
print len(pretxt)*'*'
print len(pretxt)*'*'
print "Credit card default prediction"
print pretxt

for proc in procList:
    errorRate = 0
    knn.procList = [proc]
    start_time = time.time()
    print '********** Result of using '+proc+' **********'
    for point in range(1,nPts-1):

        dataPoint = data[point]

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

    print '--- '+str(timeToRun) +timeUnit+' to run  ---'
    errorRate = errorRate/(nPts-2)
    print '--- '+ str( round(errorRate*100,3)) + ' percent correctly predicted ---' 

if ( 'GPU' in procList and 'CPU' in procList):
    perfIncrease = round(rateDict['CPU']/rateDict['GPU'],1)
    st = '****** GPU processing was '+str(perfIncrease)+' times faster ******'

    print len(st)*'*'
    print st
    print len(st)*'*'
