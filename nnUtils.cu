#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int listMax(int numValues, int *values)
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

__device__ int d_listMax(int numValues, int *values)
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

    int maxLayerSize = listMax(numLayers, layerSizes);

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
    int maxLayerSize = listMax(numLayers, layerSizes);
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
    int maxLayerSize = listMax(numLayers, layerSizes);
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

__device__ int d_getIndex(int layerIndex, int nodeIndex, int weightIndex, int maxLayerSize)
{
    return layerIndex * (maxLayerSize + 1) * maxLayerSize +
        nodeIndex * (maxLayerSize + 1) +
        weightIndex;
}

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate, float *weightDeltas,
    float *nodeErrors, float *nodeValues)
{
    // int maxLayerSize = 0;
    // for (int i = 0; i < sizeof(layerSizes) / sizeof(int); i++)
    // {
    //     if (layerSizes[i] > maxLayerSize)
    //     {
    //         maxLayerSize = layerSizes[i];
    //     }
    // }

    int dataIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (dataIndex >= numTrainingData)
    {
        return;
    }

    int maxLayerSize = d_listMax(numLayers, layerSizes);
    int numWeights = numLayers * maxLayerSize * (maxLayerSize + 1);
    // make a local copy of weights so they can be adjusted
    float *myWeights = (float *)malloc(sizeof(float) * numWeights);
    for (int i = 0; i < numWeights; i ++)
    {
        myWeights[i] = weights[i];
    }

    int nodeDataOffset = numLayers * maxLayerSize * (blockIdx.x * blockDim.x + threadIdx.x);

    int dataStartIndex = dataIndex * layerSizes[0];
    int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        // load training sample
        for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
        {
            nodeValues[nodeDataOffset + nodeIndex] = trainingData[dataStartIndex + nodeIndex];
        }
        if (0 && iterationIndex == 0 && dataIndex == 0)
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
                    printf("%.4f ", nodeValues[nodeDataOffset + i * maxLayerSize + j]);
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
                    float prevLayerValue = nodeValues[nodeDataOffset + (layerIndex - 1) * maxLayerSize + weightIndex];
                    int index = d_getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                    sum += prevLayerValue * myWeights[index];
                }
                // add bias
                int biasIndex = d_getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                sum += myWeights[biasIndex];
                nodeValues[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex] = d_activationFunction(sum);
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
                    float value = nodeValues[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex];
                    float actual = trueValues[trueValueStartIndex + nodeIndex];
                    nodeErrors[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex] =
                        value *
                        (1 - value) *
                        (value - actual);
                }
                else
                {
                    float sum = 0;
                    for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                    {
                        int index = d_getIndex(layerIndex + 1, nextLayerNodeIndex, nodeIndex, maxLayerSize);
                        sum += myWeights[index] *
                            nodeErrors[nodeDataOffset + (layerIndex + 1) * maxLayerSize + nextLayerNodeIndex];
                    }
                    float value = nodeValues[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex];
                    nodeErrors[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex] = sum * value * (1 - value);
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
                    int index = d_getIndex(layerIndex, nodeIndex, weightIndex, maxLayerSize);
                    myWeights[index] -=
                        learnRate *
                        nodeErrors[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex] *
                        nodeValues[nodeDataOffset + (layerIndex - 1) * maxLayerSize + weightIndex];
                }
                // update bias
                int index = d_getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], maxLayerSize);
                myWeights[index] -=
                    learnRate *
                    nodeErrors[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex];
            }
        }
        if (
            0 && (
                iterationIndex < 11 ||
                iterationIndex == 100 ||
                iterationIndex == 1000 ||
                iterationIndex == 10000 ||
                iterationIndex == numIterations - 1
            )
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
                    printf("%.6f ", nodeValues[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex]);
                }
                printf("\n");
            }
            printf("(Error data below)\n");
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex ++)
            {
                printf("[%d] ", layerIndex);
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    printf("%.6f ", nodeErrors[nodeDataOffset + layerIndex * maxLayerSize + nodeIndex]);
                }
                printf("\n");
            }
            // printf("Printing network for iteration %d\n", iterationIndex);
            // printNetwork(myWeights, numLayers, layerSizes);
        }
    }
    for (int i = 0; i < numWeights; i++)
    {
        // int weightIndex = (blockIdx.x * blockDim.x + threadIdx.x) % numWeights
        atomicAdd(&weightDeltas[i], myWeights[i] - weights[i]);
    }
    free(myWeights);
}

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate)
{
    int maxLayerSize = listMax(numLayers, layerSizes);
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
    int maxLayerSize = listMax(numLayers, layerSizes);
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

__device__ float d_activationFunction(float x)
{
    return (float)(1.0f / (1 + exp(x * (-1))));
}

void batchTrainNetworkGpu(
    float *weights, int numLayers, int *layerSizes,
    float *trainData, int trainDataCount, int internalIterations,
    float *trueValues, float learnRate, int batchSize,
    int numEpochs)
{
    int maxLayerSize = listMax(numLayers, layerSizes);
    int numWeights = numLayers * maxLayerSize * (maxLayerSize + 1);
    float *weightDeltas = (float *) malloc(sizeof(float) * numWeights);
    int inDataWidth = layerSizes[0];

    int threadsPerBlock = 8;

    float *d_weights = 0;
    int *d_layerSizes = 0;
    float *d_trainData = 0;
    float *d_trueValues = 0;
    float *d_weightDeltas = 0;
    float *d_nodeErrors = 0;
    float *d_nodeValues = 0;
    int numBatches = (int)ceil((float)trainDataCount / (float)batchSize);
    int numBlocks = (int)ceil((float)batchSize / (float)threadsPerBlock); // need to check this math

    printf("batchSize: %d\n", batchSize);
    printf("numBatches: %d\n", numBatches);
    printf("numBlocks: %d\n", numBlocks);

    cudaMalloc(&d_weights, sizeof(float) * numWeights);
    cudaMalloc(&d_layerSizes, sizeof(int) * numLayers);
    cudaMalloc(&d_trainData, sizeof(float) * batchSize * inDataWidth);
    cudaMalloc(&d_trueValues, sizeof(float) * batchSize * layerSizes[numLayers - 1]);
    cudaMalloc(&d_weightDeltas, sizeof(float) * numWeights);
    cudaMalloc(&d_nodeErrors, sizeof(float) * numLayers * maxLayerSize * numBlocks * threadsPerBlock);
    cudaMalloc(&d_nodeValues, sizeof(float) * numLayers * maxLayerSize * numBlocks * threadsPerBlock);

    cudaMemcpy(d_layerSizes, layerSizes, sizeof(int) * numLayers, cudaMemcpyHostToDevice);

    for (int i = 0; i < numEpochs; i++)
    {

        for (int batchNumber = 0; batchNumber < numBatches; batchNumber ++)
        {
            cudaMemcpy(d_weights, weights, sizeof(float) * numWeights, cudaMemcpyHostToDevice);

            int trainDataStartIndex = batchNumber * batchSize * inDataWidth;
            int trueValuesStartIndex = batchNumber * batchSize * layerSizes[numLayers - 1];
            int thisBatchNumSamples = batchSize;
            if ((batchNumber + 1) * batchSize > trainDataCount)
            {
                // in this case our final batch has more capacity than the number of remaining samples
                // need to copy less data in
                thisBatchNumSamples = batchSize - ((batchNumber + 1) * batchSize - trainDataCount);
            }
            int trainDataBytesToCopy = sizeof(float) * thisBatchNumSamples * inDataWidth;
            int trueValuesBytesToCopy = sizeof(float) * thisBatchNumSamples * layerSizes[numLayers - 1];
            // copy in the samples of this batch
            cudaMemcpy(d_trainData, trainData + trainDataStartIndex, trainDataBytesToCopy, cudaMemcpyHostToDevice);
            cudaMemcpy(d_trueValues, trueValues + trueValuesStartIndex, trueValuesBytesToCopy, cudaMemcpyHostToDevice);

            trainNetworkGpu<<<numBlocks, threadsPerBlock>>>(
                d_weights, numLayers, d_layerSizes,
                d_trainData, thisBatchNumSamples, internalIterations,
                d_trueValues, learnRate, d_weightDeltas,
                d_nodeErrors, d_nodeValues
            );
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            cudaMemcpy(weightDeltas, d_weightDeltas, sizeof(float) * numWeights, cudaMemcpyDeviceToHost);

            if (i < 10 || i % 1000 == 0)
            {
                printf("weightDeltas (iteration %d)\n", i);
                for (int j = 0; j < numWeights; j++)
                {
                    printf("%.3f ", weightDeltas[j]);
                }
                printf("\n");
            }
            for (int i = 0; i < numWeights; i++)
            {
                weights[i] += weightDeltas[i];
            }
            for (int i = 0; i < numWeights; i++)
            {
                weightDeltas[i] = 0;
            }
        }
    }
}

imageTrainingSamples *getImageData(char *filename, int numItems, int startingIndex)
{
    int numInputNodes = 28 * 28;
    int numOutputNodes = 10;

    imageTrainingSamples *samples = (imageTrainingSamples *)malloc(sizeof(imageTrainingSamples));
    samples->inputSamples = (float *)malloc(sizeof(float) * numItems * numInputNodes);
    samples->trueOutput = (float *)malloc(sizeof(float) * numItems * numOutputNodes);

    FILE *fileStream = fopen(filename, "r");

    char *token;
    char line[8192];
    char sep[] = ",";
    int isHeader = 1;
    int itemIndex = 0;
    int numItemsTaken = 0;
    while (numItemsTaken < numItems && fgets(line, 8192, fileStream))
    {
        if (isHeader)
        {
            // skip CSV header
            isHeader = false;
            continue;
        }

        if (itemIndex < startingIndex)
        {
            itemIndex ++;
            continue;
        }

        // first get the label / true value
        token = strtok(line, sep);
        int trueValue = strtol(token, NULL, 10);
        for (int i = 0; i < numOutputNodes; i ++)
        {
            int index = numItemsTaken * numOutputNodes + i;
            if (i == trueValue)
            {
                samples->trueOutput[index] = 1.0f;
            }
            else
            {
                samples->trueOutput[index] = 0.0f;
            }
        }

        // now get the inputs - pixels with value from 0-255
        int tokenIndex = 0;
        while (token != NULL)
        {
            samples->inputSamples[numItemsTaken * numInputNodes + tokenIndex] = (float)strtol(token, NULL, 10);
            token = strtok(NULL, sep);
            tokenIndex ++;
        }
        numItemsTaken ++;
        itemIndex ++;
    }

    samples->numItems = itemIndex;

    return samples;
}

void printSampleSketch(float *pixelValues, int sampleIndex)
{
    int width = 28;
    int height = 28;

    char ch = 0;
    float value = 0;

    int sampleOffset = sampleIndex * width * height;

    for (int y = 0; y < height; y ++)
    {
        for (int x = 0; x < width; x ++)
        {
            value = pixelValues[sampleOffset + y * width + x];
            if (value > 192) {
                ch = '#';
            }
            else if (value > 96)
            {
                ch = '=';
            }
            else if (value > 48)
            {
                ch = '-';
            }
            else
            {
                ch = ' ';
            }
            printf("%c", ch);
        }
        printf("\n");
    }
}
