import numpy as np
import knn
import time

## Data (were modified) downloaded from here:
## https://archive.ics.uci.edu/ml/datasets/Qualitative_Bankruptcy

## last column is classification, others are attributes
data = np.genfromtxt('Qualitative_Bankruptcy.txt', delimiter=',',dtype='f')

## classification
dataCl = data[:,-1]

## attributes
data = data[:, :-1]

nPts = len(data)

knn = knn.knn()

for proc in ['CPU','GPU']:
    errorRate = 0
    knn.procList = [proc]
    start_time = time.time()
    for point in range(1,nPts-1):

        dataPoint = data[point]

        knn.data = np.append(data[:][0:point],data[:][point+1:nPts],axis=0)
        knn.dataCl = np.append(dataCl[:][0:point],dataCl[:][point+1:nPts],axis=0)
        knn.testPt = dataPoint 
    
        result = knn.getPred()
        
        errorRate+=float(result==dataCl[point])

    print '******* Result of using '+proc+' *******'
    print("--- %s seconds to run  ---" % round(time.time() - start_time,3))
    errorRate = errorRate/(nPts-2)
    print("--- %s percent correct ---") % round(errorRate*100,3)
