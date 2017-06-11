import numpy as np
import knn
import time


## Datadownloaded from here:
## http://http://archive.ics.uci.edu/ml/datasets/Phishing+Websites

## last column is classification, others are attributes
data = np.genfromtxt('Phishing.txt', delimiter=',',dtype='f')

## classification
dataCl = data[:,-1]

## attributes
data = data[:, :-1]

procList = ['CPU','GPU']
rateDict = dict.fromkeys(procList)

## number of training points (max=11055)
nPoints = 5000
#nPoints = len(data)

knn = knn.knn()

pretxt = str(nPoints)+" training data examples with "+str(len(data[0]))+" features \n"
print "# Phishing prediction \n"
print pretxt

print "| Processor | Runtime | Correctly predicted | "
print "| ------------- |:----------------:| ---------------:|"
for proc in procList:
    errorRate = 0
    knn.procList = [proc]
    start_time = time.time()

    for point in range(1,nPoints-1):

        dataPoint = data[point]

        knn.data = np.append(data[:][0:point],data[:][point+1:nPoints],axis=0)
        knn.dataCl = np.append(dataCl[:][0:point],dataCl[:][point+1:nPoints],axis=0)
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

    errorRate = errorRate/(nPoints-2)
    print "| "+proc+" | "+str(timeToRun) +timeUnit+" | "+ str( round(errorRate*100,2)) + " percent |"

if ( 'GPU' in procList and 'CPU' in procList):
    perfIncrease = round(rateDict['CPU']/rateDict['GPU'],1)
    st = '\n### GPU processing was '+str(perfIncrease)+' times faster'

    print st

