float *createNetwork(int numLayers, int *layerSizes);

void printNetwork(float *weights, int numLayers, int *layerSizes);

int getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize);

void initNetworkWeights(float *weights, int numLayers, int *layerSizes);

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate);

float activationFunction(float x);

float *classify(float *weights, int numLayers, int *layerSizes, float *sample);

int max(int numValues, int *values);
