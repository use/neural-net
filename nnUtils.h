typedef struct neuralNetwork
{
    struct networkLayer **layers;
    int numLayers;
    int *layerSizes;
    int maxLayerSize;
} neuralNetwork;

typedef struct networkLayer
{
    struct networkNode **nodes;
} networkLayer;

typedef struct networkNode
{
    float *inWeights;
} networkNode;

neuralNetwork *createNetwork(int numLayers, int *layerSizes);

void printNetwork(neuralNetwork *);

void initNetworkWeights(neuralNetwork *);

void trainNetwork(neuralNetwork *net, float **trainingData, int numTrainingData, int numIterations, float **trueValues, float learnRate);

int maxLayerSize(neuralNetwork *net);

float activationFunction(float x);

float *classify(neuralNetwork *net, float *sample);
