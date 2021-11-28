#include "nnUtils.h"

int main(void)
{
    int layerSizes[3] = {2, 3, 2};
    neuralNetwork *net = createNetwork(3, layerSizes);
    printNetwork(net);
}
