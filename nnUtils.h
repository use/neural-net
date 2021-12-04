typedef struct imageTrainingSamples
{
    int numItems;
    float *inputSamples;
    float *trueOutput;
} imageTrainingSamples;

float *createNetwork(int numLayers, int *layerSizes);

void printNetwork(float *weights, int numLayers, int *layerSizes);
__device__ void d_printNetwork(float *weights, int numLayers, int *layerSizes);

int getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize);
__device__ int d_getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize);

void initNetworkWeights(float *weights, int numLayers, int *layerSizes);

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate);

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate, float *weightDeltas,
    float *d_nodeErrors, float *d_nodeValues
);

void batchTrainNetworkGpu(
    float *weights, int numLayers, int *layerSizes,
    float *trainData, int trainDataCount, int internalIterations,
    float *trueValues, float learnRate, int batchSize,
    int numEpochs
);

float activationFunction(float x);
__device__ float d_activationFunction(float x);

float *classify(float *weights, int numLayers, int *layerSizes, float *sample);

int listMax(int numValues, int *values);
__device__ int d_listMax(int numValues, int *values);

imageTrainingSamples *getImageData(char *filename, int numItems, int startingIndex);

void printSampleSketch(float *pixelValues, int sampleIndex);

int imageSampleTrueValue(float * trueValues, int sampleIndex);
