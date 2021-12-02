#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int max(int numValues, int *values)
{
    int max = 0;
    for (int i = 0; i < numValues; i++)
    {
        if (values[i] > max)
        {
            max = values[i];
        }
    }
    return max;
}

float *createNetwork(int numLayers, int *layerSizes)
{

    int maxLayerSize = max(numLayers, layerSizes);

    float *weights = (float *)malloc(sizeof(float) * numLayers * maxLayerSize * (maxLayerSize + 1));

    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
    {
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex++)
        {
            if (layerIndex == 0)
            {
                // don't do anything because layer 0 is inputs and has no incoming weights
            }
            else
            {
                int numWeights = 1 + layerSizes[layerIndex - 1]; // +1 for bias
                for (int weightIndex = 0; weightIndex < numWeights; weightIndex++)
                {
                    int index = getIndex(
                        layerIndex, nodeIndex, weightIndex,
                        maxLayerSize
                    );
                    weights[index] = 0.5f;
                }
            }
        }
    }
    return weights;
}

void printNetwork(float *weights, int numLayers, int *layerSizes)
{
    int maxLayerSize = max(numLayers, layerSizes);
    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
    {
        printf("\n---<Layer %d>\n", layerIndex);
        printf("Layer size: %d\n", layerSizes[layerIndex]);
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex++)
        {
            printf("[%d] ", nodeIndex);
            if (layerIndex > 0)
            {
                // print weights pointing to this node
                int numWeights = layerSizes[layerIndex - 1] + 1;
                for (int weightIndex = 0; weightIndex < numWeights; weightIndex ++)
                {
                    int index = getIndex(
                        layerIndex, nodeIndex, weightIndex,
                        maxLayerSize
                    );
                    printf("%.6f, ", weights[index]);
                    if (weightIndex == numWeights - 1)
                    {
                        printf("(bias)");
                    }
                }
                printf("   ");
            }
        }
        printf("\n");
    }
}

void initNetworkWeights(float *weights, int numLayers, int *layerSizes)
{
    int maxLayerSize = max(numLayers, layerSizes);
    srand(time(NULL));
    for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
    {
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
        {
            int numWeights = 1 + layerSizes[layerIndex - 1];
            for (int weightIndex = 0; weightIndex < numWeights; weightIndex ++)
            {
                int index = getIndex(
                    layerIndex, nodeIndex, weightIndex,
                    maxLayerSize
                );
                weights[index] = (float)((rand() % 10000 + 1 - 5000)) / 10000.0f;
            }
        }
    }
}

int getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize)
{
    return layerIndex * (maxLayerSize + 1) * maxLayerSize +
        nodeIndex * (maxLayerSize + 1) +
        weightIndex;
}

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate, int *weightDeltas)
{
    int maxLayerSize = max(numLayers, layerSizes);

    // make a local copy of weights so they can be adjusted
    float *myWeights = (float *)malloc(sizeof(weights));
    for (int i = 0; i < sizeof(myWeights) / sizeof(float); i ++)
    {
        myWeights[i] = weights[i];
    }

    // node delta
    float errors[numLayers][maxLayerSize];
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < maxLayerSize; j++)
        {
            errors[i][j] = 0;
        }
    }

    // activation values
    float values[numLayers][maxLayerSize];
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < maxLayerSize; j++)
        {
            values[i][j] = 0;
        }
    }

    int dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int dataStartIndex = dataIndex * layerSizes[0];
    int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        // load training sample
        for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
        {
            values[0][nodeIndex] = trainingData[dataStartIndex + nodeIndex];
        }
        if (iterationIndex == 0 && dataIndex == 1)
        {
            printf("Training Data\n");
            for (int i = 0; i < numTrainingData; i++)
            {
                printf("[%d] ", i);
                for (int j = 0; j < layerSizes[0]; j++)
                {
                    printf("%.4f ", trainingData[dataStartIndex + j]);
                }
                printf("(");
                for (int j = 0; j < layerSizes[numLayers - 1]; j++)
                {
                    printf("%.4f ", trueValues[trueValueStartIndex + j]);
                }
                printf(")\n");
            }
            printf("Values\n");
            for (int i = 0; i < numLayers; i++)
            {
                printf("[%d] ", i);
                for (int j = 0; j < maxLayerSize; j++)
                {
                    printf("%.4f ", values[i][j]);
                }
                printf("\n");
            }
        }
        // forward compute
        // start with first hidden layer
        for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
        {
            for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
            {
                float sum = 0;
                for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
                {
                    float prevLayerValue = values[layerIndex - 1][weightIndex];
                    int index = getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                    sum += prevLayerValue * myWeights[index];
                }
                // add bias
                int biasIndex = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                sum += myWeights[biasIndex];
                values[layerIndex][nodeIndex] = activationFunction(sum);
            }
        }
        // find error of layers
        for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex --)
        {
            for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
            {
                if (layerIndex == numLayers - 1)
                {
                    // special case for output layer
                    float value = values[layerIndex][nodeIndex];
                    float actual = trueValues[trueValueStartIndex + nodeIndex];
                    errors[layerIndex][nodeIndex] =
                        value *
                        (1 - value) *
                        (value - actual);
                }
                else
                {
                    float sum = 0;
                    for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                    {
                        int index = getIndex(layerIndex + 1, nextLayerNodeIndex, nodeIndex, maxLayerSize);
                        sum += myWeights[index] *
                            errors[layerIndex + 1][nextLayerNodeIndex];
                    }
                    float value = values[layerIndex][nodeIndex];
                    errors[layerIndex][nodeIndex] = sum * value * (1 - value);
                }
            }
        }

        // update weights
        for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
        {
            for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
            {
                for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
                {
                    int index = getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                    myWeights[index] -=
                        learnRate *
                        errors[layerIndex][nodeIndex] *
                        values[layerIndex - 1][weightIndex];
                }
                // update bias
                int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                myWeights[index] -=
                    learnRate *
                    errors[layerIndex][nodeIndex];
            }
        }
        if (
            iterationIndex < 11 ||
            iterationIndex == 100 ||
            iterationIndex == 1000 ||
            iterationIndex == 10000 ||
            iterationIndex == numIterations - 1
        )
        {
            printf("\nIteration %d\n", iterationIndex);
            printf("(Training sample)\n");
            for (int dataNodeIndex = 0; dataNodeIndex < layerSizes[0]; dataNodeIndex ++)
            {
                printf("%.6f ", trainingData[dataStartIndex + dataNodeIndex]);
            }
            printf("\n");
            printf("(Value data below)\n");
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex ++)
            {
                printf("[%d] ", layerIndex);
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    printf("%.6f ", values[layerIndex][nodeIndex]);
                }
                printf("\n");
            }
            printf("(Error data below)\n");
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex ++)
            {
                printf("[%d] ", layerIndex);
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    printf("%.6f ", errors[layerIndex][nodeIndex]);
                }
                printf("\n");
            }
            printf("Printing network for iteration %d\n", iterationIndex);
            printNetwork(myWeights, numLayers, layerSizes);
        }
    }
}

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate)
{
    int maxLayerSize = max(numLayers, layerSizes);
    // node delta
    float errors[numLayers][maxLayerSize];
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < maxLayerSize; j++)
        {
            errors[i][j] = 0;
        }
    }

    // activation values
    float values[numLayers][maxLayerSize];
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < maxLayerSize; j++)
        {
            values[i][j] = 0;
        }
    }

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        for (int dataIndex = 0; dataIndex < numTrainingData; dataIndex ++)
        {
            int dataStartIndex = dataIndex * layerSizes[0];
            int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];
            // load training sample
            for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
            {
                values[0][nodeIndex] = trainingData[dataStartIndex + nodeIndex];
            }
            if (iterationIndex == 0 && dataIndex == 1)
            {
                printf("Training Data\n");
                for (int i = 0; i < numTrainingData; i++)
                {
                    printf("[%d] ", i);
                    for (int j = 0; j < layerSizes[0]; j++)
                    {
                        printf("%.4f ", trainingData[dataStartIndex + j]);
                    }
                    printf("(");
                    for (int j = 0; j < layerSizes[numLayers - 1]; j++)
                    {
                        printf("%.4f ", trueValues[trueValueStartIndex + j]);
                    }
                    printf(")\n");
                }
                printf("Values\n");
                for (int i = 0; i < numLayers; i++)
                {
                    printf("[%d] ", i);
                    for (int j = 0; j < maxLayerSize; j++)
                    {
                        printf("%.4f ", values[i][j]);
                    }
                    printf("\n");
                }
            }
            // forward compute
            // start with first hidden layer
            for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    float sum = 0;
                    for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
                    {
                        float prevLayerValue = values[layerIndex - 1][weightIndex];
                        int index = getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                        sum += prevLayerValue * weights[index];
                    }
                    // add bias
                    int biasIndex = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                    sum += weights[biasIndex];
                    values[layerIndex][nodeIndex] = activationFunction(sum);
                }
            }
            // find error of layers
            for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex --)
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    if (layerIndex == numLayers - 1)
                    {
                        // special case for output layer
                        float value = values[layerIndex][nodeIndex];
                        float actual = trueValues[trueValueStartIndex + nodeIndex];
                        errors[layerIndex][nodeIndex] =
                            value *
                            (1 - value) *
                            (value - actual);
                    }
                    else
                    {
                        float sum = 0;
                        for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                        {
                            int index = getIndex(layerIndex + 1, nextLayerNodeIndex, nodeIndex, maxLayerSize);
                            sum += weights[index] *
                                errors[layerIndex + 1][nextLayerNodeIndex];
                        }
                        float value = values[layerIndex][nodeIndex];
                        errors[layerIndex][nodeIndex] = sum * value * (1 - value);
                    }
                }
            }

            // update weights
            for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
                    {
                        int index = getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                        weights[index] -=
                            learnRate *
                            errors[layerIndex][nodeIndex] *
                            values[layerIndex - 1][weightIndex];
                    }
                    // update bias
                    int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                    weights[index] -=
                        learnRate *
                        errors[layerIndex][nodeIndex];
                }
            }
            if (
                iterationIndex < 11 ||
                iterationIndex == 100 ||
                iterationIndex == 1000 ||
                iterationIndex == 10000 ||
                iterationIndex == numIterations - 1
            )
            {
                printf("\nIteration %d\n", iterationIndex);
                printf("(Training sample)\n");
                for (int dataNodeIndex = 0; dataNodeIndex < layerSizes[0]; dataNodeIndex ++)
                {
                    printf("%.6f ", trainingData[dataStartIndex + dataNodeIndex]);
                }
                printf("\n");
                printf("(Value data below)\n");
                for (int layerIndex = 0; layerIndex < numLayers; layerIndex ++)
                {
                    printf("[%d] ", layerIndex);
                    for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                    {
                        printf("%.6f ", values[layerIndex][nodeIndex]);
                    }
                    printf("\n");
                }
                printf("(Error data below)\n");
                for (int layerIndex = 0; layerIndex < numLayers; layerIndex ++)
                {
                    printf("[%d] ", layerIndex);
                    for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                    {
                        printf("%.6f ", errors[layerIndex][nodeIndex]);
                    }
                    printf("\n");
                }
                printf("Printing network for iteration %d\n", iterationIndex);
                printNetwork(weights, numLayers, layerSizes);
            }
        }
    }
}

float *classify(float *weights, int numLayers, int *layerSizes, float *sample)
{
    int maxLayerSize = max(numLayers, layerSizes);
    float values[numLayers][maxLayerSize];
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < maxLayerSize; j++)
        {
            values[i][j] = 0;
        }
    }
    // load input layer from provided sample
    for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
    {
        values[0][nodeIndex] = sample[nodeIndex];
    }
    for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
    {
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
        {
            float sum = 0;
            for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
            {
                float prevLayerValue = values[layerIndex - 1][weightIndex];
                int index = getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                sum += prevLayerValue * weights[index];
            }
            // add bias
            int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
            sum += weights[index];
            values[layerIndex][nodeIndex] = activationFunction(sum);
        }
    }
    float *out = (float *)malloc(sizeof(int) * layerSizes[numLayers - 1]);
    for (int nodeIndex = 0; nodeIndex < layerSizes[numLayers - 1]; nodeIndex ++)
    {
        out[nodeIndex] = values[numLayers - 1][nodeIndex];
    }
    return out;
}

float activationFunction(float x)
{
    return (float)(1.0f / (1 + exp(x * (-1))));
}
