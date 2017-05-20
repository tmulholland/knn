import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class knn(object):
    """ k-nearest-neighbor classifier to test CUDA performace increase over CPU
    
    Could make this a much better classifier.
    However, purpose was to compare GPU and CPU speed performance

    """

    def __init__(self, ):

        ## The 'k' in k-nearest-neighbors
        self.k = 3

        ## decide which processing unit(s) to compute distance
        self.procList = ['GPU']
                         
        ## dictionary of distances [CPU array and GPU array]
        self.distDict = dict.fromkeys(self.procList)

    def cpuDist(self):
        """ simple NN brute force distance calculator
        """

        self.distDict['CPU'] = [np.sum((self.testPt-self.data[n,:])*
                                       (self.testPt-self.data[n,:])) 
                                for n in range(self.nData)]

    def gpuDist(self):
        """ CUDA GPU distance calculator 
        """

        # define launch configuration for GPU computing
        tpb = (320,1,1)
        nbks = (int(self.nData/320)+1, 1)

        # CUDA source module
        mod = SourceModule("""
        __global__ void dist(float* dataPt, float* data, float* distance, int* nData)
        {
          int idx = threadIdx.x + blockIdx.x * blockDim.x;

          if(idx < nData[0]) {
          int nfeat = nData[1];

          for(int feat = 0; feat<nfeat; feat++) {
            distance[idx]+=(data[idx*nfeat+feat]-dataPt[feat])*(data[idx*nfeat+feat]-dataPt[feat]);
          }    
         }
        }""")

        ## initialize distance and N(features) array
        distances = np.zeros(self.nData,'f')
        size = np.array([self.nData,len(self.testPt)],'i')

        ## allocate memory on device 
        data_d = cuda.mem_alloc(self.data.nbytes)
        testPt_d = cuda.mem_alloc(self.testPt.nbytes)
        distances_d = cuda.mem_alloc(distances.nbytes)
        size_d = cuda.mem_alloc(size.nbytes)

        # copy arrays to device
        cuda.memcpy_htod(data_d, self.data)
        cuda.memcpy_htod(testPt_d, self.testPt)
        cuda.memcpy_htod(size_d, size)
        cuda.memcpy_htod(distances_d, distances)


        # define and execute (gpu) function
        func = mod.get_function("dist")
        func(testPt_d, data_d, distances_d, size_d, block=tpb, grid=nbks)

        # copy output back to host
        cuda.memcpy_dtoh(distances, distances_d)

        self.distDict['GPU'] = distances

    def getPred(self):
        """ knn class prediction from set of data with classes and a testPoint
        k = number of nearest neighbors to consider
        testPt = features of data you want to classify
        data = set of data to compare testPt to
        dataCl = classes that correspond to data """

        self.nData = np.array(len(self.dataCl),'i')
        
        ## compute distance array with CPU (brute force)
        if 'CPU' in self.procList:
            self.cpuDist()

        ## compute distance array with GPU
        if 'GPU' in self.procList:
            self.gpuDist()

        for Key in self.procList:
            distances = self.distDict[Key]
            indices = np.argsort(distances)

            classes = [self.dataCl[indices[index]] for index in range(self.k)]
        
            ## very simple prediction 
            ## takes most frequent class of kth nearst neighbor
            classPred = np.argmax(np.bincount(classes))
        
        return classPred
