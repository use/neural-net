#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

    trainNetwork(net, trainData, 4, 1000001, trueValues, .05);

    printNetwork(net);

    for (int i = 0; i < 4; i++)
    {
        float *sample = (float *)malloc(2 * sizeof(float));
        sample[0] = trainData[i][0];
        sample[1] = trainData[i][1];
        float *result = classify(net, sample);

        printf("classification input:\n");
        for (int i = 0; i < 2; i++)
        {
            printf("(%d) %f  ", i, sample[i]);
        }
        printf("\n");
        printf("classification result:\n");
        for (int i = 0; i < 2; i++)
        {
            printf("(%d) %f  ", i, result[i]);
        }
        printf("\n");
        assert(
            result[0] - trueValues[i][0] < .01 &&
            result[1] - trueValues[i][1] < .01
        );
    }
}
