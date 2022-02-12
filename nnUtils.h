typedef struct imageTrainingSamples
{
    int numItems;
    float *inputSamples;
    float *trueOutput;
} imageTrainingSamples;

float *createNetwork(int numLayers, int *layerSizes);

void printNetwork(float *weights, int numLayers, int *layerSizes);
__device__ void d_printNetwork(float *weights, int numLayers, int *layerSizes);

int getIndex(int layerIndex, int nodeIndex, int weightIndex, int *layerSizes);

int getNumNetworkWeights(int numLayers, int *layerSizes);

void initNetworkWeights(float *weights, int numLayers, int *layerSizes);

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate,
    float *trainingTime
);

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate,
    float *d_nodeErrors, float *d_nodeValues, float *scratchWeights,
    int useSubkernels
);

__global__ void k_updateNodeValues(
    int myWeightsIndex, int nodeDataValuesInOffset, int nodeDataValuesOutOffset,
    float *scratchWeights, float *nodeValuesIn, float *nodeValuesOut,
    int *layerSizes, int layerIndex
);

__device__ void updateNodeValues(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesInOffset, int nodeDataValuesOutOffset,
    float *scratchWeights, float *nodeValuesIn, float *nodeValuesOut,
    int *layerSizes, int layerIndex
);

__global__ void k_updateNodeErrors(
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset, int trueValueStartIndex,
    float *scratchWeights, float *nodeValues, float *nodeErrors, float *trueValues,
    int *layerSizes, int numLayers, int layerIndex
);

__device__ void updateNodeErrors(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset, int trueValueStartIndex,
    float *scratchWeights, float *nodeValues, float *nodeErrors, float *trueValues,
    int *layerSizes, int numLayers, int layerIndex
);

__global__ void k_updateWeights(
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset,
    float *scratchWeights, float *nodeValues, float *nodeErrors,
    int *layerSizes, int layerIndex, float learnRate
);

__device__ void updateWeights(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset,
    float *scratchWeights, float *nodeValues, float *nodeErrors,
    int *layerSizes, int layerIndex, float learnRate
);

__global__ void sumVectors(float *vectors, int numVectors, int vectorLength);

void batchTrainNetworkGpu(
    float *weights, int numLayers, int *layerSizes,
    float *trainData, int trainDataCount, int internalIterations,
    float *trueValues, float learnRate, int batchSize,
    int numEpochs, imageTrainingSamples *testCases,
    int useSubkernels, int threadsPerBlock
);

float activationFunction(float x);
__device__ float d_activationFunction(float x);

float *classify(float *weights, int numLayers, int *layerSizes, float *sample, int sampleIndex);

int listMax(int numValues, int *values);
__device__ int d_listMax(int numValues, int *values);

imageTrainingSamples *getImageData(char *filename, int numItems, int startingIndex);

void printSampleSketch(float *pixelValues, int sampleIndex);

int imageSampleTrueValue(float * trueValues, int sampleIndex);

int imageSampleTestResult(float *trueValues, int sampleIndex, float *result);

int imageSampleResultToInt(float *result);

void testNetwork(float *weights, int numLayers, int *layerSizes, imageTrainingSamples *testCases);

int getNumValueNodes(int numLayers, int *layerSizes);
int getValueIndex(int *layerSizes, int layerIndex, int nodeIndex);
int getNumErrorNodes(int numLayers, int *layerSizes);
int getErrorIndex(int *layerSizes, int layerIndex, int nodeIndex);

float testNetworkGpu(float *weights, int numLayers, int *layerSizes, imageTrainingSamples *testCases);

__global__ void classifyAndCheckSample(
    float *weights, int numLayers, int *layerSizes,
    float *testData, int thisBatchNumSamples,
    float *trueValues, float *nodeValues,
    int *results
);

__global__ void applyBatchWeightUpdate(float *weights, float *deltas, int numWeights);
