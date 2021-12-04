#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

void testAndFunction();
void testAndFunctionGpu();
void testTonyFunction();
void testTonyFunctionGpu();
void testImageDataParsing();
void testImageTrainingGpu();

int main(void)
{
    // testAndFunctionGpu();
    // testTonyFunction();
    // testTonyFunctionGpu();

    testImageDataParsing();

    testImageTrainingGpu();
}

void testImageTrainingGpu()
{
    int numLayers = 3;
    int layerSizes[3] = {28 * 28, 100, 10};
    float *weights = createNetwork(numLayers, layerSizes);
    initNetworkWeights(weights, numLayers, layerSizes);
    printf("Initialized weights\n");

    int numSamples = 1;
    char filePath[] = "data/mnist_train.csv";
    imageTrainingSamples *samples = getImageData(filePath, numSamples, 0);
    assert(imageSampleTrueValue(samples->trueOutput, 0) == 5);
    printf("Got training data\n");
    printSampleSketch(samples->inputSamples, 0);

    int internalIterations = 1;
    batchTrainNetworkGpu(
        weights, numLayers, layerSizes,
        samples->inputSamples, numSamples, internalIterations,
        samples->trueOutput, .05, 1,
        1
    );
    printf("Done training\n");
}

void testImageDataParsing()
{
    imageTrainingSamples *samples = NULL;
    char filePath[] = "data/mnist_code_verification.csv";
    samples = getImageData(filePath, 1000, 0);
    printSampleSketch(samples->inputSamples, 0);
    assert(imageSampleTrueValue(samples->trueOutput, 0) == 7);

    char filePath2[] = "data/mnist_code_verification.csv";
    samples = getImageData(filePath2, 10, 5);

    assert(imageSampleTrueValue(samples->trueOutput, 0) == 1);
    assert(imageSampleTrueValue(samples->trueOutput, 1) == 4);
    assert(imageSampleTrueValue(samples->trueOutput, 2) == 9);
    assert(imageSampleTrueValue(samples->trueOutput, 3) == 5);
    assert(imageSampleTrueValue(samples->trueOutput, 4) == 9);
    assert(imageSampleTrueValue(samples->trueOutput, 5) == 0);
    assert(imageSampleTrueValue(samples->trueOutput, 6) == 6);

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

    int inDataWidth = layerSizes[0];
    int inDataCount = 4;
    int outDataWidth = layerSizes[numLayers - 1];
    float *trainData = (float *)malloc(sizeof(float) * inDataCount * inDataWidth);
    for (int i = 0; i<inDataCount; i++) {
        for (int j = 0; j<inDataWidth; j++) {
            trainData[i * inDataWidth + j] = tmpTrainData[i][j];
        }
    }
    float *trueValues = (float *)malloc(sizeof(float) * inDataCount * outDataWidth);
    for (int i = 0; i<inDataCount; i++) {
        for (int j = 0; j<outDataWidth; j++) {
            trueValues[i * outDataWidth + j] = tmpTrueValues[i][j];
        }
    }

    int internalIterations = 1;
    batchTrainNetworkGpu(
        weights, numLayers, layerSizes,
        trainData, inDataCount, internalIterations,
        trueValues, .05, 2,
        10000
    );

    printNetwork(weights, numLayers, layerSizes);

    for (int i = 0; i < inDataCount; i++)
    {
        float *sample = (float *)malloc(inDataWidth * sizeof(float));
        sample[0] = trainData[i * inDataWidth + 0];
        sample[1] = trainData[i * inDataWidth + 1];
        float *result = classify(weights, numLayers, layerSizes, sample);

        printf("classification input:\n");
        for (int j = 0; j < inDataWidth; j++)
        {
            printf("(%d) %f  ", j, sample[j]);
        }
        printf("\n");
        printf("classification result:\n");
        for (int j = 0; j < outDataWidth; j++)
        {
            printf("(%d) %f  ", j, result[j]);
        }
        printf("\n");
        assert(
            fabsf(result[0] - trueValues[i * outDataWidth + 0]) < .1 &&
            fabsf(result[1] - trueValues[i * outDataWidth + 1]) < .1
        );
    }
    printf("got here\n");
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

void testTonyFunctionGpu()
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
    batchTrainNetworkGpu(
        net, numLayers, layerSizes,
        input, inDataLength, 1,
        trueOut, .05, 64,
        100000
    );

    printNetwork(net, numLayers, layerSizes);

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
