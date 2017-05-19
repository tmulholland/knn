import numpy as np

class knn(object):
    """ k-nearest-neighbor classifier to test CUDA performace increase over CPU

    """

    def __init__(self, ):

        ## The 'k' in k-nearest-neighbors
        self.k = 3

        ## decide which processing unit(s) to compute distance
        self.procList = ['CPU',] #'GPU',]
                         

        ## dictionary of distances [CPU array and GPU array]
        self.distDict = dict.fromkeys(self.procList)

    def cpuDist(self):
        
        self.distDict['CPU'] = [np.sum((self.testPt-self.data[n,:])**2) 
                                for n in range(self.nData)]

    def getPred(self):
        """ knn class prediction from set of data with classes and a testPoint
        k = number of nearest neighbors to consider
        testPt = features of data you want to classify
        inData = set of data to compare testPoint to
        inClass = classes that correspond to inData """

        self.nData = len(self.dataCl)
        
        ## compute distance array with CPU (brute force)
        if 'CPU' in self.procList:
            self.cpuDist()

        for Key in self.procList:
            distances = self.distDict[Key]
            indices = np.argsort(distances)

            classes = [self.dataCl[indices[index]] for index in range(self.k)]
        
            ## very simple prediction 
            ## takes most frequent class of kth nearst neighbor
            classPred = np.argmax(np.bincount(classes))
        
        return classPred
