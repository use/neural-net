#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

neuralNetwork *createNetwork(int numLayers, int *layerSizes)
{
    neuralNetwork *network = (neuralNetwork *)malloc(sizeof(neuralNetwork));
    network->numLayers = numLayers;
    network->layerSizes = (int *)malloc(sizeof(int) * numLayers);
    for (int i = 0; i < numLayers; i++)
    {
        network->layerSizes[i] = layerSizes[i];
    }
    network->layers = (networkLayer **)malloc(sizeof(networkLayer *) * numLayers);

    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
    {
        networkLayer *layer = (networkLayer *)malloc(sizeof(networkLayer));
        layer->bias = 0.75f;
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
                node->inWeights = (float *)malloc(sizeof(float) * layerSizes[layerIndex - 1]);
                for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex++)
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
        if (layerIndex > 0)
        {
            printf("Bias: %.2f\n", net->layers[layerIndex]->bias);
        }
        for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex++)
        {
            printf("[%d] ", nodeIndex);
            if (layerIndex > 0)
            {
                // print weights pointing to this node
                for (int weightIndex = 0; weightIndex < net->layerSizes[layerIndex - 1]; weightIndex ++)
                {
                    printf("%.2f, ", net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex]);
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
        net->layers[layerIndex]->bias = (float)((rand() % 10000 + 1 - 5000)) / 10000.0f;
        for (int nodeIndex = 0; nodeIndex < net->layerSizes[layerIndex]; nodeIndex ++)
        {
            for (int weightIndex = 0; weightIndex < net->layerSizes[layerIndex - 1]; weightIndex ++)
            {
                net->layers[layerIndex]->nodes[nodeIndex]->inWeights[weightIndex] = (float)((rand() % 10000 + 1 - 5000)) / 10000.0f;
            }
        }
    }
}
