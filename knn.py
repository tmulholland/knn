import numpy as np

class knn(object):
    """ k-nearest-neighbor classifier to test CUDA performace increase over CPU

    """

    def __init__(self):

        ## decide which algorithm to compute distance
        self.doCPU = True
        self.doGPU = True

        ## dictionary of distances [CPU array and GPU array]
        self.distDict = dict.fromkeys(['CPU','GPU'])

    def cpuDist():
        
        self.distDict['CPU'] = [np.sum((self.testPt-self.inData[n,:])**2) 
                                for n in range(self.nData)]

    def getPred(k,testPt,inData,inClass):
        """ knn class prediction from set of data with classes and a testPoint
        k = number of nearest neighbors to consider
        testPt = features of data you want to classify
        inData = set of data to compare testPoint to
        inClass = classes that correspond to inData """

        self.nData = len(self.inClass)
        
        ## compute distance array with CPU (brute force)
        if self.doCPU:
            self.cpuDist()

        for Key in distDict.Keys():
            distances = distDict[Key]
            indices = np.argsort(distances)

            classes = [inClass[indices[index]] for index in range(k)]
        
            ## very simple prediction 
            ## takes most frequent class of kth nearst neighbor
            classPred = np.argmax(np.bincount(classes))
        
            print classPred
