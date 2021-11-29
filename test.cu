#include "nnUtils.h"
#include <stdio.h>

int main(void)
{
    int layerSizes[3] = {2, 3, 2};
    neuralNetwork *net = createNetwork(3, layerSizes);
    printNetwork(net);
    initNetworkWeights(net);
    printNetwork(net);

    float tmpTrainData[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
    };

    float tmpTrueValues[4][2] = {
        {1.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
    };

    float **trainData = (float **)malloc(sizeof(int *) * 4);
    for (int i = 0; i<4; i++) {
        trainData[i] = (float *)malloc(sizeof(int) * 2);
        for (int j = 0; j<2; j++) {
            trainData[i][j] = tmpTrainData[i][j];
        }
    }
    float **trueValues = (float **)malloc(sizeof(int *) * 4);
    for (int i = 0; i<4; i++) {
        trueValues[i] = (float *)malloc(sizeof(int) * 2);
        for (int j = 0; j<2; j++) {
            trueValues[i][j] = tmpTrueValues[i][j];
        }
    }

    trainNetwork(net, trainData, 4, 100001, trueValues, .02);

    printNetwork(net);
}
