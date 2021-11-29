typedef struct neuralNetwork
{
    struct networkLayer **layers;
    int numLayers;
    int *layerSizes;
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
