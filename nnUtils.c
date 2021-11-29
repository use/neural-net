#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

neuralNetwork *createNetwork(int numLayers, int *layerSizes)
{
    neuralNetwork *network = (neuralNetwork *)malloc(sizeof(neuralNetwork));
    network->numLayers = numLayers;
    network->layerSizes = (int *)malloc(sizeof(int) * numLayers);
    network->maxLayerSize = 0;
    for (int i = 0; i < numLayers; i++)
    {
        network->layerSizes[i] = layerSizes[i];
        if (i > 0 && layerSizes[i] > network->maxLayerSize)
        {
            network->maxLayerSize = layerSizes[i];
        }
    }

    network->layers = (networkLayer **)malloc(sizeof(networkLayer *) * numLayers);

    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
    {
        networkLayer *layer = (networkLayer *)malloc(sizeof(networkLayer));
        layer->nodes = (networkNode **)malloc(sizeof(networkNode *) * layerSizes[layerIndex]);
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex++)
        {
            networkNode *node = (networkNode *)malloc(sizeof(networkNode));
            if (layerIndex == 0)
            {
                node->inWeights = NULL;
            }
            else
            {
                int numWeights = 1 + layerSizes[layerIndex - 1]; // +1 for bias
                node->inWeights = (float *)malloc(sizeof(float) * numWeights);
                for (int weightIndex = 0; weightIndex < numWeights; weightIndex++)
                {
                    node->inWeights[weightIndex] = 0.5f;
                }
            }
            layer->nodes[nodeIndex] = node;
        }
        network->layers[layerIndex] = layer;
    }
    return network;
}

void printNetwork(neuralNetwork *net)
{
    for (int layerIndex = 0; layerIndex < net->numLayers; layerIndex++)
    {
        printf("\n---<Layer %d>\n", layerIndex);
        printf("Layer size: %d\n", net->layerSizes[layerIndex]);
        for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex++)
        {
            printf("[%d] ", nodeIndex);
            if (layerIndex > 0)
            {
                // print weights pointing to this node
                int numWeights = net->layerSizes[layerIndex - 1] + 1;
                for (int weightIndex = 0; weightIndex < numWeights; weightIndex ++)
                {
                    printf("%.6f, ", net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex]);
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

void initNetworkWeights(neuralNetwork *net)
{
    srand(time(NULL));
    for (int layerIndex = 1; layerIndex < net->numLayers; layerIndex ++)
    {
        for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
        {
            int numWeights = 1 + net->layerSizes[layerIndex - 1];
            for (int weightIndex = 0; weightIndex < numWeights; weightIndex ++)
            {
                net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex] = (float)((rand() % 10000 + 1 - 5000)) / 10000.0f;
            }
        }
    }
}

void trainNetwork(neuralNetwork *net, float **trainingData, int numTrainingData, int numIterations, float **trueValues, float learnRate)
{
    // node delta
    float errors[net->numLayers][net->maxLayerSize];
    for (int i = 0; i < net->numLayers; i++)
    {
        for (int j = 0; j < net->maxLayerSize; j++)
        {
            errors[i][j] = 0;
        }
    }

    // activation values
    float values[net->numLayers][net->maxLayerSize];
    for (int i = 0; i < net->numLayers; i++)
    {
        for (int j = 0; j < net->maxLayerSize; j++)
        {
            values[i][j] = 0;
        }
    }

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        for (int dataIndex = 0; dataIndex < numTrainingData; dataIndex ++)
        {
            // load training sample
            for (int nodeIndex = 0; nodeIndex < net->layerSizes[0]; nodeIndex ++)
            {
                values[0][nodeIndex] = trainingData[dataIndex][nodeIndex];
            }
            if (iterationIndex == 0 && dataIndex == 1)
            {
                printf("Training Data\n");
                for (int i = 0; i < numTrainingData; i++)
                {
                    printf("[%d] ", i);
                    for (int j = 0; j < net->layerSizes[0]; j++)
                    {
                        printf("%.4f ", trainingData[i][j]);
                    }
                    printf("(");
                    for (int j = 0; j < net->layerSizes[net->numLayers - 1]; j++)
                    {
                        printf("%.4f ", trueValues[i][j]);
                    }
                    printf(")\n");
                }
                printf("Values\n");
                for (int i = 0; i < net->numLayers; i++)
                {
                    printf("[%d] ", i);
                    for (int j = 0; j < net->maxLayerSize; j++)
                    {
                        printf("%.4f ", values[i][j]);
                    }
                    printf("\n");
                }
            }
            // forward compute
            // start with first hidden layer
            for (int layerIndex = 1; layerIndex < net->numLayers; layerIndex ++)
            {
                for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
                {
                    float sum = 0;
                    for (int weightIndex = 0; weightIndex < net->layerSizes[layerIndex - 1]; weightIndex ++)
                    {
                        float prevLayerValue = values[layerIndex - 1][weightIndex];
                        float weight = net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex];
                        sum += prevLayerValue * weight;
                    }
                    // add bias
                    sum += net->layers[layerIndex]->nodes[nodeIndex]->inWeights[net->layerSizes[layerIndex - 1]];
                    values[layerIndex][nodeIndex] = activationFunction(sum);
                }
            }
            // find error of layers
            for (int layerIndex = net->numLayers - 1; layerIndex > 0; layerIndex --)
            {
                for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
                {
                    if (layerIndex == net->numLayers - 1)
                    {
                        // special case for output layer
                        float value = values[layerIndex][nodeIndex];
                        float actual = trueValues[dataIndex][nodeIndex];
                        errors[layerIndex][nodeIndex] =
                            value *
                            (1 - value) *
                            (value - actual);
                    }
                    else
                    {
                        float sum = 0;
                        for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < net->layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                        {
                            sum += net->layers[layerIndex + 1]->nodes[nextLayerNodeIndex]->inWeights[nodeIndex] *
                                errors[layerIndex + 1][nextLayerNodeIndex];
                        }
                        float value = values[layerIndex][nodeIndex];
                        errors[layerIndex][nodeIndex] = sum * value * (1 - value);
                    }
                }
            }

            // update weights
            for (int layerIndex = 1; layerIndex < net->numLayers; layerIndex ++)
            {
                for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
                {
                    for (int weightIndex = 0; weightIndex < net->layerSizes[layerIndex - 1]; weightIndex ++)
                    {
                        net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex] -=
                            learnRate *
                            errors[layerIndex][nodeIndex] *
                            values[layerIndex - 1][weightIndex];
                    }
                    // update bias
                    net->layers[layerIndex]->nodes[nodeIndex]->inWeights[net->layerSizes[layerIndex - 1]] -=
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
                for (int dataNodeIndex = 0; dataNodeIndex < net->layerSizes[0]; dataNodeIndex ++)
                {
                    printf("%.6f ", trainingData[dataIndex][dataNodeIndex]);
                }
                printf("\n");
                printf("(Value data below)\n");
                for (int layerIndex = 0; layerIndex < net->numLayers; layerIndex ++)
                {
                    printf("[%d] ", layerIndex);
                    for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
                    {
                        printf("%.6f ", values[layerIndex][nodeIndex]);
                    }
                    printf("\n");
                }
                printf("(Error data below)\n");
                for (int layerIndex = 0; layerIndex < net->numLayers; layerIndex ++)
                {
                    printf("[%d] ", layerIndex);
                    for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
                    {
                        printf("%.6f ", errors[layerIndex][nodeIndex]);
                    }
                    printf("\n");
                }
                printf("Printing network for iteration %d\n", iterationIndex);
                printNetwork(net);
            }
        }
    }
}

float *classify(neuralNetwork *net, float *sample)
{
    float values[net->numLayers][net->maxLayerSize];
    for (int i = 0; i < net->numLayers; i++)
    {
        for (int j = 0; j < net->maxLayerSize; j++)
        {
            values[i][j] = 0;
        }
    }
    // load input layer from provided sample
    for (int nodeIndex = 0; nodeIndex < net->layerSizes[0]; nodeIndex ++)
    {
        values[0][nodeIndex] = sample[nodeIndex];
    }
    for (int layerIndex = 1; layerIndex < net->numLayers; layerIndex ++)
    {
        for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
        {
            float sum = 0;
            for (int weightIndex = 0; weightIndex < net->layerSizes[layerIndex - 1]; weightIndex ++)
            {
                float prevLayerValue = values[layerIndex - 1][weightIndex];
                float weight = net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex];
                sum += prevLayerValue * weight;
            }
            // add bias
            sum += net->layers[layerIndex]->nodes[nodeIndex]->inWeights[net->layerSizes[layerIndex - 1]];
            values[layerIndex][nodeIndex] = activationFunction(sum);
        }
    }
    float *out = (float *)malloc(sizeof(int) * net->layerSizes[net->numLayers - 1]);
    for (int nodeIndex = 0; nodeIndex < net->layerSizes[net->numLayers - 1]; nodeIndex ++)
    {
        out[nodeIndex] = values[net->numLayers - 1][nodeIndex];
    }
    return out;
}

float activationFunction(float x)
{
    return (float)(1.0f / (1 + exp(x * (-1))));
}
