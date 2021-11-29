#include "nnUtils.h"
#include <stdio.h>

int main(void)
{
    int layerSizes[4] = {2, 5, 3, 6};
    neuralNetwork *net = createNetwork(4, layerSizes);
    printNetwork(net);
    initNetworkWeights(net);
    printNetwork(net);
}
