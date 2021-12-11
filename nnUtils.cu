#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
void printVec(int** a, int n);

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
            file, line );
    exit( EXIT_FAILURE );
    }
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                              cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
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

    int numWeights = getNumNetworkWeights(numLayers, layerSizes);
    float *weights = (float *)malloc(sizeof(float) * numWeights);
    int flatIndex = 0;
    for (int layerIndex = 1; layerIndex < numLayers; layerIndex++)
    {
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex++)
        {
            for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1] + 1; weightIndex++)
            {
                weights[flatIndex] = (float)flatIndex;
                flatIndex ++;
            }
        }
    }
    return weights;
}

void printNetwork(float *weights, int numLayers, int *layerSizes)
{
    for (int layerIndex = 1; layerIndex < numLayers; layerIndex++)
    {
        printf("\n---<Layer %d>\n", layerIndex);
        printf("Layer size: %d\n", layerSizes[layerIndex]);
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex++)
        {
            printf("[node %d] ", nodeIndex);
            if (layerIndex > 0)
            {
                // print weights pointing to this node
                int numWeights = layerSizes[layerIndex - 1] + 1;
                for (int weightIndex = 0; weightIndex < numWeights; weightIndex ++)
                {
                    int index = getIndex(
                        layerIndex, nodeIndex, weightIndex,
                        layerSizes
                    );
                    printf("%.3f, ", weights[index]);
                    if (weightIndex == numWeights - 1)
                    {
                        printf("(bias)");
                    }
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

void initNetworkWeights(float *weights, int numLayers, int *layerSizes)
{
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
                    layerSizes
                );
                weights[index] = (float)((rand() % 10000 + 1 - 5000)) / 10000.0f;
            }
        }
    }
}

int __device__ __host__ getIndex(int layerIndex, int nodeIndex, int weightIndex, int *layerSizes)
{
    assert(layerIndex > 0);

    int prevWeights = 0;
    for (int l = 1; l < layerIndex; l++)
    {
        prevWeights = prevWeights + layerSizes[l] * (layerSizes[l-1] + 1);
    }
    return prevWeights + nodeIndex * (layerSizes[layerIndex - 1] + 1) + weightIndex;
}

__device__ __host__ int getNumNetworkWeights(int numLayers, int *layerSizes)
{
    int w = 0;
    for (int l = 1; l < numLayers; l ++)
    {
        w = w + layerSizes[l] * (layerSizes[l - 1] + 1);
    }
    return w;
}

__device__ __host__ int getNumValueNodes(int numLayers, int *layerSizes)
{
    int numValues = 0;
    for (int l = 0; l < numLayers; l ++)
    {
        numValues = numValues + layerSizes[l];
    }
    return numValues;
}

__device__ __host__ int getValueIndex(int *layerSizes, int layerIndex, int nodeIndex)
{
    int numPrev = 0;
    for (int l = 1; l <= layerIndex; l++)
    {
        numPrev = numPrev + layerSizes[l-1];
    }
    return numPrev + nodeIndex;
}

__device__ __host__ int getNumErrorNodes(int numLayers, int *layerSizes)
{
    int numErrors = 0;
    for (int l = 1; l < numLayers; l ++) // don't count input layer because it can't have errors
    {
        numErrors = numErrors + layerSizes[l];
    }
    return numErrors;
}

__device__ __host__ int getErrorIndex(int *layerSizes, int layerIndex, int nodeIndex)
{
    int numPrev = 0;
    for (int l = 2; l <= layerIndex; l++) // don't count input layer because it can't have errors
    {
        numPrev = numPrev + layerSizes[l-1];
    }
    return numPrev + nodeIndex;
}

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate, float *weightDeltas,
    float *nodeErrors, float *nodeValues, float *scratchWeights)
{

    int dataIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (dataIndex >= numTrainingData)
    {
        return;
    }

    int debug = 0;
    int numWeights = getNumNetworkWeights(numLayers, layerSizes);

    int numValueNodes = getNumValueNodes(numLayers, layerSizes);
    int numErrorNodes = getNumErrorNodes(numLayers, layerSizes);

    int myWeightsIndex = (blockIdx.x * blockDim.x + threadIdx.x) * numWeights;

    int nodeDataValuesOffset = (blockIdx.x * blockDim.x + threadIdx.x) * numValueNodes;
    int nodeDataErrorsOffset = (blockIdx.x * blockDim.x + threadIdx.x) * numErrorNodes;

    int dataStartIndex = dataIndex * layerSizes[0];
    int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        if (debug)
        {
            printf("start loading training sample\n");
        }
        // load training sample
        for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
        {
            nodeValues[nodeDataValuesOffset + nodeIndex] = trainingData[dataStartIndex + nodeIndex];
        }

        if (debug)
        {
            printf("loaded training sample\n");
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
                    float prevLayerValue = nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex - 1, weightIndex)];
                    int index = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                    sum += prevLayerValue * scratchWeights[myWeightsIndex + index];
                }
                // add bias
                int biasIndex = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
                sum += scratchWeights[myWeightsIndex + biasIndex];
                nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex, nodeIndex)] = d_activationFunction(sum);
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
                    float value = nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex, nodeIndex)];
                    float actual = trueValues[trueValueStartIndex + nodeIndex];
                    nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)] =
                        value *
                        (1 - value) *
                        (value - actual);
                }
                else
                {
                    float sum = 0;
                    for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                    {
                        int index = getIndex(layerIndex + 1, nextLayerNodeIndex, nodeIndex, layerSizes);
                        sum += scratchWeights[myWeightsIndex + index] *
                            nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex + 1, nextLayerNodeIndex)];
                    }
                    float value = nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex, nodeIndex)];
                    nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)] = sum * value * (1 - value);
                }
            }
        }
        if (debug)
        {
            printf("finished finding errors\n");
        }
        // update weights
        for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
        {
            for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
            {
                for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
                {
                    int index = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                    scratchWeights[myWeightsIndex + index] -=
                        learnRate *
                        nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)] *
                        nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex - 1, weightIndex)];
                }
                // update bias
                int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
                scratchWeights[myWeightsIndex + index] -=
                    learnRate *
                    nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)];
            }
        }
        if (debug)
        {
            printf("finished updating weights\n");
        }
    }
    if (debug)
    {
        printf("finished internal iterations\n");
    }
}

void trainNetwork(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate)
{
    // node delta
    float *errors = (float *)malloc(sizeof(float) * getNumErrorNodes(numLayers, layerSizes));

    // activation values
    float *values = (float *)malloc(sizeof(float) * getNumValueNodes(numLayers, layerSizes));

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        for (int dataIndex = 0; dataIndex < numTrainingData; dataIndex ++)
        {
            if (dataIndex % 1000 == 0)
            {
                printf("sample %d\n", dataIndex);
            }

            int dataStartIndex = dataIndex * layerSizes[0];
            int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];

            // load training sample
            for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
            {
                values[nodeIndex] = trainingData[dataStartIndex + nodeIndex];
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
                        float prevLayerValue = values[getValueIndex(layerSizes, layerIndex - 1, weightIndex)];
                        int index = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                        sum += prevLayerValue * weights[index];
                    }
                    // add bias
                    int biasIndex = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
                    sum += weights[biasIndex];
                    values[getValueIndex(layerSizes, layerIndex, nodeIndex)] = activationFunction(sum);
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
                        float value = values[getValueIndex(layerSizes, layerIndex, nodeIndex)];
                        float actual = trueValues[trueValueStartIndex + nodeIndex];
                        errors[getErrorIndex(layerSizes, layerIndex, nodeIndex)] =
                            value *
                            (1 - value) *
                            (value - actual);
                    }
                    else
                    {
                        float sum = 0;
                        for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
                        {
                            int index = getIndex(layerIndex + 1, nextLayerNodeIndex, nodeIndex, layerSizes);
                            sum += weights[index] *
                                errors[getErrorIndex(layerSizes, layerIndex + 1, nextLayerNodeIndex)];
                        }
                        float value = values[getValueIndex(layerSizes, layerIndex, nodeIndex)];
                        errors[getErrorIndex(layerSizes, layerIndex, nodeIndex)] = sum * value * (1 - value);
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
                        int index = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                        weights[index] -=
                            learnRate *
                            errors[getErrorIndex(layerSizes, layerIndex, nodeIndex)] *
                            values[getValueIndex(layerSizes, layerIndex - 1, weightIndex)];
                    }
                    // update bias
                    int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
                    weights[index] -=
                        learnRate *
                        errors[getErrorIndex(layerSizes, layerIndex, nodeIndex)];
                }
            }
        }
    }
}

float *classify(float *weights, int numLayers, int *layerSizes, float *samples, int sampleIndex)
{
    int maxLayerSize = listMax(numLayers, layerSizes);
    float *values = (float *)malloc(sizeof(float) * getNumValueNodes(numLayers, layerSizes));
    int sampleOffset = layerSizes[0] * sampleIndex;
    // load input layer from provided sample
    for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
    {
        values[nodeIndex] = samples[sampleOffset + nodeIndex];
    }
    for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
    {
        for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
        {
            float sum = 0;
            for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
            {
                float prevLayerValue = values[getValueIndex(layerSizes, layerIndex - 1, weightIndex)];
                int index = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                sum += prevLayerValue * weights[index];
            }
            // add bias
            int index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
            sum += weights[index];
            values[getValueIndex(layerSizes, layerIndex, nodeIndex)] = activationFunction(sum);
        }
    }
    float *out = (float *)malloc(sizeof(float) * layerSizes[numLayers - 1]);
    for (int nodeIndex = 0; nodeIndex < layerSizes[numLayers - 1]; nodeIndex ++)
    {
        out[nodeIndex] = values[getValueIndex(layerSizes, numLayers - 1, nodeIndex)];
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
    int numEpochs, imageTrainingSamples *testCases)
{
    int numWeights = getNumNetworkWeights(numLayers, layerSizes);
    float *weightDeltas = (float *) malloc(sizeof(float) * numWeights);
    float *scratchWeights = (float *) malloc(sizeof(float) * batchSize * numWeights);
    int inDataWidth = layerSizes[0];

    int threadsPerBlock = 8;

    float *d_weights = 0;
    int *d_layerSizes = 0;
    float *d_trainData = 0;
    float *d_trueValues = 0;
    float *d_weightDeltas = 0;
    float *d_nodeErrors = 0;
    float *d_nodeValues = 0;
    float *d_scratchWeights = 0;
    int numBatches = (int)ceil((float)trainDataCount / (float)batchSize);
    int numBlocks = (int)ceil((float)batchSize / (float)threadsPerBlock); // need to check this math

    int debug = 0;
    printf("Num Batches: %d\n", numBatches);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Num Blocks: %d\n", numBlocks);

    cudaMalloc(&d_weights, sizeof(float) * numWeights);
    cudaMalloc(&d_layerSizes, sizeof(int) * numLayers);
    cudaMalloc(&d_trainData, sizeof(float) * batchSize * inDataWidth);
    cudaMalloc(&d_trueValues, sizeof(float) * batchSize * layerSizes[numLayers - 1]);
    cudaMalloc(&d_weightDeltas, sizeof(float) * numWeights);
    cudaMalloc(&d_nodeErrors, sizeof(float) * getNumErrorNodes(numLayers, layerSizes) * numBlocks * threadsPerBlock);
    cudaMalloc(&d_nodeValues, sizeof(float) * getNumValueNodes(numLayers, layerSizes) * numBlocks * threadsPerBlock);
    cudaMalloc(&d_scratchWeights, sizeof(float) * batchSize * numWeights);

    cudaMemcpy(d_layerSizes, layerSizes, sizeof(int) * numLayers, cudaMemcpyHostToDevice);

    for (int epochIndex = 0; epochIndex < numEpochs; epochIndex++)
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

            if (debug)
            {
                printf("start copying scratch weights\n");
            }
            for (int sampleIndex = 0; sampleIndex < thisBatchNumSamples; sampleIndex ++)
            {
                cudaMemcpy(d_scratchWeights + sampleIndex * numWeights, d_weights, numWeights * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            if (debug)
            {
                printf("done copying scratch weights\n");
            }
            trainNetworkGpu<<<numBlocks, threadsPerBlock>>>(
                d_weights, numLayers, d_layerSizes,
                d_trainData, thisBatchNumSamples, internalIterations,
                d_trueValues, learnRate, d_weightDeltas,
                d_nodeErrors, d_nodeValues, d_scratchWeights
            );
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            cudaMemcpy(scratchWeights, d_scratchWeights, thisBatchNumSamples * numWeights * sizeof(float), cudaMemcpyDeviceToHost);

            if (debug)
            {
                printf("start adding deltas\n");
            }
            for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1] + 1; weightIndex ++)
                    {
                        float delta = 0;
                        int weightFlatIndex = getIndex(layerIndex, nodeIndex, weightIndex, layerSizes);
                        for (int sampleIndex = 0; sampleIndex < thisBatchNumSamples; sampleIndex ++)
                        {
                            float thisDelta = scratchWeights[sampleIndex * numWeights + weightFlatIndex] - weights[weightFlatIndex];
                            delta += thisDelta;
                        }
                        weights[weightFlatIndex] += delta;
                    }
                }
            }
            if (debug)
            {
                printf("done adding deltas\n");
            }
            for (int i = 0; i < numWeights; i++)
            {
                weightDeltas[i] = 0;
            }
            if (1 || debug)
            {
                printf("Finished epoch %d / %d, batch %d / %d\n",
                    epochIndex, numEpochs,
                    batchNumber, numBatches);
            }
        }
        printf("finished epoch %d\n", epochIndex);
        if (testCases)
        {
            testNetwork(weights, numLayers, layerSizes, testCases);
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
            float rawValue = (float)strtol(token, NULL, 10);
            samples->inputSamples[numItemsTaken * numInputNodes + tokenIndex] = rawValue / (float)255;
            token = strtok(NULL, sep);
            tokenIndex ++;
        }
        numItemsTaken ++;
        itemIndex ++;
    }

    samples->numItems = itemIndex;

    return samples;
}

__device__ __host__ void printSampleSketch(float *pixelValues, int sampleIndex)
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
            if (value > .75) {
                ch = '#';
            }
            else if (value > .5)
            {
                ch = '=';
            }
            else if (value > .25)
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

int imageSampleTrueValue(float * trueValues, int sampleIndex)
{
    int nodesPerSample = 10;
    int startIndex = nodesPerSample * sampleIndex;

    for (int nodeIndex = 0; nodeIndex < nodesPerSample; nodeIndex ++)
    {
        if (trueValues[startIndex + nodeIndex] == 1)
        {
            return nodeIndex;
        }
    }
    return 0;
}

int imageSampleTestResult(float *trueValues, int sampleIndex, float *result)
{
    int trueValue = imageSampleTrueValue(trueValues, sampleIndex);

    int selectedValue = imageSampleResultToInt(result);

    if (trueValue == selectedValue) {
        return 1;
    }

    return 0;
}

int imageSampleResultToInt(float *result)
{
    int nodesPerSample = 10;
    int selectedValue = 0;
    float highestActivation = 0;
    for (int nodeIndex = 0; nodeIndex < nodesPerSample; nodeIndex ++)
    {
        if (result[nodeIndex] > highestActivation)
        {
            highestActivation = result[nodeIndex];
            selectedValue = nodeIndex;
        }
    }

    return selectedValue;
}

void testNetwork(float *weights, int numLayers, int *layerSizes, imageTrainingSamples *testCases)
{
    int numCorrect = 0;
    for (int testCaseIndex = 0; testCaseIndex < testCases->numItems; testCaseIndex ++)
    {
        int trueValue = imageSampleTrueValue(testCases->trueOutput, testCaseIndex);
        float *result = classify(weights, numLayers, layerSizes, testCases->inputSamples, testCaseIndex);
        int isCorrect = imageSampleTestResult(testCases->trueOutput, testCaseIndex, result);
        // printf("Actual / Result: %d / %d ", trueValue, imageSampleResultToInt(result));
        for (int i = 0; i < layerSizes[numLayers - 1]; i++)
        {
            // printf("%.3f ", result[i]);
        }
        if (isCorrect)
        {
            numCorrect ++;
            // printf("Correct");
        }
        else
        {
            // printf("NOPE");
        }
    }
    printf("Accuracy: %.2f\n", (float)numCorrect / (float) testCases->numItems);
}
