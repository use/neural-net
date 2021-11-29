#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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


    int layerSizes2[3] = {2, 3, 2};
    neuralNetwork *net2 = createNetwork(3, layerSizes2);
    initNetworkWeights(net2);

    int indata[8][8] = {
        { 1,1,1,1, 1,1,1,1},
        { 1,1,1,1, 1,1,1,1},
        { 1,1,0,1, 1,1,1,1},
        { 1,0,0,0, 1,1,1,1},
        { 1,0,0,0, 0,0,1,1},
        { 0,0,0,0, 0,0,1,1},
        { 0,0,0,0, 0,1,1,1},
        { 0,0,0,1, 1,1,1,1}
    };

    float **input = (float **)malloc(sizeof(float *) * 8 * 8);
    float **trueOut = (float **)malloc(sizeof(float *) * 8 * 8);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j<8; j++)
        {
            float *inValues = (float*)malloc(sizeof(float) * 2);
            inValues[0] = i;
            inValues[1] = j;
            input[i * 8 + j] = inValues;

            float *trueOutValues = (float*)malloc(sizeof(float) * 2);
            if (indata[i][j]) {
                trueOutValues[0] = 0;
                trueOutValues[1] = 1;
            } else {
                trueOutValues[0] = 1;
                trueOutValues[1] = 0;
            }
            trueOut[i * 8 + j] = trueOutValues;
        }
    }
    trainNetwork(net2, input, 64, 100001, trueOut, .05);

    for (int i = 0; i < 64; i++)
    {
        float *sample = (float *)malloc(2 * sizeof(float));
        sample[0] = input[i][0];
        sample[1] = input[i][1];
        float *result = classify(net2, sample);

        printf("input: ");
        for (int i = 0; i < 2; i++)
        {
            printf("(%d) %.3f  ", i, sample[i]);
        }
        printf("\n");
        printf("result: ");
        for (int i = 0; i < 2; i++)
        {
            printf("(%d) %.3f  ", i, result[i]);
        }
        if (
            fabsf(result[0] - trueOut[i][0]) > .1 ||
            fabsf(result[1] - trueOut[i][1]) > .1
        )
        {
            printf(" (DOH)");
        }
        printf(" expected: (%.3f, %.3f)", trueOut[i][0], trueOut[i][1]);
        printf("\n");
        assert(
            fabsf(result[0] - trueOut[i][0]) < .1 &&
            fabsf(result[1] - trueOut[i][1]) < .1
        );
    }
}
