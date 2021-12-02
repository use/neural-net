#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

void testAndFunction();
void testAndFunctionGpu();
void testTonyFunction();

int main(void)
{
    testAndFunctionGpu();
    // testTonyFunction();
}

void testAndFunctionGpu()
{
    int numLayers = 3;
    int layerSizes[3] = {2, 3, 2};
    float *weights = createNetwork(numLayers, layerSizes);
    printNetwork(weights, numLayers, layerSizes);
    initNetworkWeights(weights, numLayers, layerSizes);
    printNetwork(weights, numLayers, layerSizes);

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

    int inDataWidth = 2;
    int inDataCount = 4;
    int outDataWidth = 2;
    float *trainData = (float *)malloc(sizeof(int *) * inDataCount);
    for (int i = 0; i<inDataCount; i++) {
        for (int j = 0; j<inDataWidth; j++) {
            trainData[i * inDataWidth + j] = tmpTrainData[i][j];
        }
    }
    float *trueValues = (float *)malloc(sizeof(int *) * inDataCount * outDataWidth);
    for (int i = 0; i<4; i++) {
        for (int j = 0; j<2; j++) {
            trueValues[i * inDataWidth + j] = tmpTrueValues[i][j];
        }
    }

    int maxLayerSize = listMax(numLayers, layerSizes);
    float *weightDeltas = (float*) malloc(sizeof(weights));

    int threadsPerBlock = 2;
    int numBlocks = (int)ceil(inDataCount / threadsPerBlock); // need to check this math

    float *d_weights = 0;
    float *d_trainData = 0;
    float *d_trueValues = 0;
    float *d_weightDeltas = 0;
    float *d_nodeErrors = 0;
    float *d_nodeValues = 0;

    cudaMalloc(&d_weights, sizeof(weights));
    cudaMalloc(&d_trainData, sizeof(trainData));
    cudaMalloc(&d_trueValues, sizeof(trueValues));
    cudaMalloc(&d_weightDeltas, sizeof(weights));
    cudaMalloc(&d_nodeErrors, sizeof(float) * numLayers * maxLayerSize * numBlocks * threadsPerBlock);
    cudaMalloc(&d_nodeValues, sizeof(float) * numLayers * maxLayerSize * numBlocks * threadsPerBlock);

    cudaMemcpy(d_weights, weights, sizeof(weights), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainData, trainData, sizeof(trainData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trueValues, trueValues, sizeof(trueValues), cudaMemcpyHostToDevice);

    trainNetworkGpu<<<numBlocks, threadsPerBlock>>>(weights, numLayers, layerSizes, trainData, 4, 1, trueValues, .05, d_weightDeltas, d_nodeErrors, d_nodeValues);

    cudaMemcpy(weightDeltas, d_weightDeltas, sizeof(weightDeltas), cudaMemcpyDeviceToHost);

    for (int i = 0; i < sizeof(weights) / sizeof(float); i++)
    {
        weights[i] = weightDeltas[i];
    }

    printNetwork(weights, numLayers, layerSizes);

    for (int i = 0; i < 4; i++)
    {
        float *sample = (float *)malloc(2 * sizeof(float));
        sample[0] = trainData[i * inDataWidth + 0];
        sample[1] = trainData[i * inDataWidth + 1];
        float *result = classify(weights, numLayers, layerSizes, sample);

        printf("classification input:\n");
        for (int j = 0; j < 2; j++)
        {
            printf("(%d) %f  ", j, sample[j]);
        }
        printf("\n");
        printf("classification result:\n");
        for (int j = 0; j < 2; j++)
        {
            printf("(%d) %f  ", j, result[j]);
        }
        printf("\n");
        assert(
            fabsf(result[0] - trueValues[i * outDataWidth + 0]) < .1 &&
            fabsf(result[1] - trueValues[i * outDataWidth + 1]) < .1
        );
    }
}

void testAndFunction()
{
    int numLayers = 3;
    int layerSizes[3] = {2, 3, 2};
    float *weights = createNetwork(numLayers, layerSizes);
    printNetwork(weights, numLayers, layerSizes);
    initNetworkWeights(weights, numLayers, layerSizes);
    printNetwork(weights, numLayers, layerSizes);

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

    int inDataWidth = 2;
    int inDataCount = 4;
    int outDataWidth = 2;
    float *trainData = (float *)malloc(sizeof(int *) * inDataCount);
    for (int i = 0; i<inDataCount; i++) {
        for (int j = 0; j<inDataWidth; j++) {
            trainData[i * inDataWidth + j] = tmpTrainData[i][j];
        }
    }
    float *trueValues = (float *)malloc(sizeof(int *) * inDataCount * outDataWidth);
    for (int i = 0; i<4; i++) {
        for (int j = 0; j<2; j++) {
            trueValues[i * inDataWidth + j] = tmpTrueValues[i][j];
        }
    }

    trainNetwork(weights, numLayers, layerSizes, trainData, 4, 100001, trueValues, .05);

    printNetwork(weights, numLayers, layerSizes);

    for (int i = 0; i < 4; i++)
    {
        float *sample = (float *)malloc(2 * sizeof(float));
        sample[0] = trainData[i * inDataWidth + 0];
        sample[1] = trainData[i * inDataWidth + 1];
        float *result = classify(weights, numLayers, layerSizes, sample);

        printf("classification input:\n");
        for (int j = 0; j < 2; j++)
        {
            printf("(%d) %f  ", j, sample[i]);
        }
        printf("\n");
        printf("classification result:\n");
        for (int j = 0; j < 2; j++)
        {
            printf("(%d) %f  ", j, result[i]);
        }
        printf("\n");
        assert(
            fabsf(result[0] - trueValues[i * outDataWidth + 0]) < .1 &&
            fabsf(result[1] - trueValues[i * outDataWidth + 1]) < .1
        );
    }
}

void testTonyFunction()
{
    int numLayers = 3;
    int layerSizes[3] = {2, 3, 2};
    float *net = createNetwork(3, layerSizes);
    initNetworkWeights(net, numLayers, layerSizes);

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

    int inDataWidth = 2;
    int inDataLength = 64;
    int outDataWidth = 2;
    float *input = (float *)malloc(sizeof(float *) * inDataLength * inDataWidth);
    float *trueOut = (float *)malloc(sizeof(float *) * inDataLength * outDataWidth);
    int dataIndex = 0;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j<8; j++)
        {
            input[dataIndex * inDataWidth + 0] = i;
            input[dataIndex * inDataWidth + 1] = j;

            if (indata[i][j]) {
                trueOut[dataIndex * inDataWidth + 0] = 0;
                trueOut[dataIndex * inDataWidth + 1] = 1;
            } else {
                trueOut[dataIndex * inDataWidth + 0] = 1;
                trueOut[dataIndex * inDataWidth + 1] = 0;
            }
            dataIndex ++;
        }
    }
    trainNetwork(net, numLayers, layerSizes, input, 64, 100001, trueOut, .05);

    for (int i = 0; i < 64; i++)
    {
        int dataIndex = i * inDataWidth;
        float *sample = (float *)malloc(inDataWidth * sizeof(float));
        sample[0] = input[dataIndex + 0];
        sample[1] = input[dataIndex + 1];
        float *result = classify(net, numLayers, layerSizes, sample);

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
            fabsf(result[0] - trueOut[dataIndex + 0]) > .1 ||
            fabsf(result[1] - trueOut[dataIndex + 1]) > .1
        )
        {
            printf(" (DOH)");
        }
        printf(" expected: (%.3f, %.3f)", trueOut[dataIndex + 0], trueOut[dataIndex + 1]);
        printf("\n");
        assert(
            fabsf(result[0] - trueOut[dataIndex + 0]) < .1 &&
            fabsf(result[1] - trueOut[dataIndex + 1]) < .1
        );
    }
}
