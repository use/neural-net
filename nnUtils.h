float *createNetwork(int numLayers, int *layerSizes);

__host__ __device__ void printNetwork(float *weights, int numLayers, int *layerSizes);

__host__ __device__ int getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize);

void initNetworkWeights(float *weights, int numLayers, int *layerSizes);

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate);

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate, float *weightDeltas,
    float *d_nodeErrors, float *d_nodeValues
);

__host__ __device__ float activationFunction(float x);

float *classify(float *weights, int numLayers, int *layerSizes, float *sample);

__host__ __device__ int listMax(int numValues, int *values);
