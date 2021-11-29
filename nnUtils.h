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
    float bias;
} networkLayer;

typedef struct networkNode
{
    float *inWeights;
} networkNode;

neuralNetwork *createNetwork(int numLayers, int *layerSizes);

void printNetwork(neuralNetwork *);

void initNetworkWeights(neuralNetwork *);

void trainNetwork(neuralNetwork *net, float **trainingData, int numTrainingData, float **trueValues);

int maxLayerSize(neuralNetwork *net);
