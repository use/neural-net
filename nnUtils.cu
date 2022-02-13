#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printVec(int** a, int n);

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

__global__ void sumVectors(float *vectors, int numVectors, int vectorLength)
{
    // this is optimized to handle a relatively small amount of large vectors
    // like 64 vectors that are 17000 long
    // if we use ~1000 threads, the number of adds it takes should be like 64 * 17 in the above example
    // each thread handles one index of the vectors, summing the value of all vectors at that index
    // all other vectors will be summed into the vector starting at index 0
    int threadsInGrid = blockDim.x * gridDim.x;
    for (int indexInVector = blockDim.x * blockIdx.x + threadIdx.x; indexInVector < vectorLength; indexInVector += threadsInGrid)
    {
        for (int vectorNumber = 1; vectorNumber < numVectors; vectorNumber ++)
        {
            vectors[indexInVector] += vectors[vectorNumber * vectorLength + indexInVector];
        }
    }
}

__global__ void k_updateNodeValues(
    int myWeightsIndex, int nodeDataValuesInOffset, int nodeDataValuesOutOffset,
    float *initialWeights, float *nodeValuesIn, float *nodeValuesOut,
    int *layerSizes, int layerIndex
)
{
    extern __shared__ float sharedMemNodes[];
    for (int i = threadIdx.x; i < layerSizes[layerIndex - 1]; i += blockDim.x)
    {
        sharedMemNodes[i] = nodeValuesIn[nodeDataValuesInOffset + i];
    }

    __syncthreads();

    updateNodeValues(
        threadIdx.x,
        myWeightsIndex, 0, nodeDataValuesOutOffset,
        initialWeights, sharedMemNodes, nodeValuesOut,
        layerSizes, layerIndex
    );
}

__device__ void updateNodeValues(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesInOffset, int nodeDataValuesOutOffset,
    float *initialWeights, float *nodeValuesIn, float *nodeValuesOut,
    int *layerSizes, int layerIndex
)
{
    float sum = 0;
    int index = getIndex(layerIndex, nodeIndex, 0, layerSizes);
    int valueIndex = 0;
    for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
    {
        float prevLayerValue = nodeValuesIn[nodeDataValuesInOffset + valueIndex];
        sum += prevLayerValue * initialWeights[index];
        index += 1;
        valueIndex += 1;
    }
    // add bias
    int biasIndex = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
    sum += initialWeights[biasIndex];
    nodeValuesOut[nodeDataValuesOutOffset + getValueIndex(layerSizes, layerIndex, nodeIndex)] = d_activationFunction(sum);
}

__global__ void k_updateNodeErrors(
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset, int trueValueStartIndex,
    float *initialWeights, float *nodeValues, float *nodeErrors, float *trueValues,
    int *layerSizes, int numLayers, int layerIndex
)
{
    updateNodeErrors(
        threadIdx.x,
        myWeightsIndex, nodeDataValuesOffset, nodeDataErrorsOffset, trueValueStartIndex,
        initialWeights, nodeValues, nodeErrors, trueValues,
        layerSizes, numLayers, layerIndex
    );
}

__device__ void updateNodeErrors(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset, int trueValueStartIndex,
    float *initialWeights, float *nodeValues, float *nodeErrors, float *trueValues,
    int *layerSizes, int numLayers, int layerIndex
)
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
        int index = getIndex(layerIndex + 1, 0, nodeIndex, layerSizes);
        int errorIndex = getErrorIndex(layerSizes, layerIndex + 1, 0);
        for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerSizes[layerIndex + 1]; nextLayerNodeIndex ++)
        {
            sum += initialWeights[index] *
                nodeErrors[nodeDataErrorsOffset + errorIndex];
            index += 1;
            errorIndex += 1;
        }
        float value = nodeValues[nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex, nodeIndex)];
        nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)] = sum * value * (1 - value);
    }
}

__global__ void k_updateWeights(
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset,
    float *scratchWeights, float *nodeValues, float *nodeErrors,
    int *layerSizes, int layerIndex, float learnRate
)
{
    updateWeights(
        threadIdx.x,
        myWeightsIndex, nodeDataValuesOffset, nodeDataErrorsOffset,
        scratchWeights, nodeValues, nodeErrors,
        layerSizes, layerIndex, learnRate
    );
}

__device__ void updateWeights(
    int nodeIndex,
    int myWeightsIndex, int nodeDataValuesOffset, int nodeDataErrorsOffset,
    float *scratchWeights, float *nodeValues, float *nodeErrors,
    int *layerSizes, int layerIndex, float learnRate
)
{
    int index = getIndex(layerIndex, nodeIndex, 0, layerSizes);
    int errorIndex = getErrorIndex(layerSizes, layerIndex, nodeIndex);
    int valueIndex = 0;
    for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex ++)
    {
        scratchWeights[myWeightsIndex + index] =
            -1 *
            learnRate *
            nodeErrors[nodeDataErrorsOffset + errorIndex] *
            nodeValues[nodeDataValuesOffset + valueIndex];
        index += 1;
        valueIndex += 1;
    }
    // update bias
    index = getIndex(layerIndex, nodeIndex, layerSizes[layerIndex - 1], layerSizes);
    scratchWeights[myWeightsIndex + index] =
        -1 *
        learnRate *
        nodeErrors[nodeDataErrorsOffset + getErrorIndex(layerSizes, layerIndex, nodeIndex)];
}

__global__ void trainNetworkGpu(float *weights, int numLayers, int *layerSizes,
    float *trainingData, int numTrainingData,
    int numIterations, float *trueValues, float learnRate,
    float *nodeErrors, float *nodeValues, float *scratchWeights,
    int useSubkernels
)
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
        // forward compute
        // start with first hidden layer
        float *nodeValuesIn = 0;
        int nodeValuesInOffset = 0;

        for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
        {
            // first layer is just the training data
            if (layerIndex == 1)
            {
                nodeValuesIn = trainingData;
                nodeValuesInOffset = dataStartIndex;
            }
            else
            {
                nodeValuesIn = nodeValues;
                nodeValuesInOffset = nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex - 1, 0);
            }

            if (useSubkernels)
            {
                int sharedMemSize = sizeof(float) * layerSizes[layerIndex - 1];
                k_updateNodeValues<<<1, layerSizes[layerIndex], sharedMemSize>>>(
                    myWeightsIndex, nodeValuesInOffset, nodeDataValuesOffset,
                    weights, nodeValuesIn, nodeValues,
                    layerSizes, layerIndex
                );
            }
            else
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    updateNodeValues(
                        nodeIndex,
                        myWeightsIndex, nodeValuesInOffset, nodeDataValuesOffset,
                        weights, nodeValuesIn, nodeValues,
                        layerSizes, layerIndex
                    );
                }
            }
        }

        // find error of layers
        for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex --)
        {
            if (useSubkernels)
            {
                k_updateNodeErrors<<<1, layerSizes[layerIndex]>>>(
                    myWeightsIndex, nodeDataValuesOffset, nodeDataErrorsOffset, trueValueStartIndex,
                    weights, nodeValues, nodeErrors, trueValues,
                    layerSizes, numLayers, layerIndex
                );
            }
            else
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    updateNodeErrors(
                        nodeIndex,
                        myWeightsIndex, nodeDataValuesOffset, nodeDataErrorsOffset, trueValueStartIndex,
                        weights, nodeValues, nodeErrors, trueValues,
                        layerSizes, numLayers, layerIndex
                    );
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
            // first layer is just the training data
            if (layerIndex == 1)
            {
                nodeValuesIn = trainingData;
                nodeValuesInOffset = dataStartIndex;
            }
            else
            {
                nodeValuesIn = nodeValues;
                nodeValuesInOffset = nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex - 1, 0);
            }

            if (useSubkernels)
            {
                k_updateWeights<<<1, layerSizes[layerIndex]>>>(
                    myWeightsIndex, nodeValuesInOffset, nodeDataErrorsOffset,
                    scratchWeights, nodeValuesIn, nodeErrors,
                    layerSizes, layerIndex, learnRate
                );
            }
            else
            {
                for (int nodeIndex = 0; nodeIndex < layerSizes[layerIndex]; nodeIndex ++)
                {
                    updateWeights(
                        nodeIndex,
                        myWeightsIndex, nodeValuesInOffset, nodeDataErrorsOffset,
                        scratchWeights, nodeValuesIn, nodeErrors,
                        layerSizes, layerIndex, learnRate
                    );
                }
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
    int numIterations, float *trueValues, float learnRate, float *trainingTime)
{
    // node delta
    float *errors = (float *)malloc(sizeof(float) * getNumErrorNodes(numLayers, layerSizes));

    // activation values
    float *values = (float *)malloc(sizeof(float) * getNumValueNodes(numLayers, layerSizes));

    struct timeval t1;
    struct timeval t2;

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex ++)
    {
        for (int dataIndex = 0; dataIndex < numTrainingData; dataIndex ++)
        {
            int dataStartIndex = dataIndex * layerSizes[0];
            int trueValueStartIndex = dataIndex * layerSizes[numLayers - 1];

            // load training sample
            for (int nodeIndex = 0; nodeIndex < layerSizes[0]; nodeIndex ++)
            {
                values[nodeIndex] = trainingData[dataStartIndex + nodeIndex];
            }

            gettimeofday(&t1, NULL);

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

            gettimeofday(&t2, NULL);
            *trainingTime +=
                (t2.tv_sec * 1000 + t2.tv_usec / 1000) -
                (t1.tv_sec * 1000 + t1.tv_usec / 1000);

        }
    }

    free(errors);
    free(values);
}

__device__ __host__ float *classify(float *weights, int numLayers, int *layerSizes, float *samples, int sampleIndex)
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
    free(values);
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
    int numEpochs, imageTrainingSamples *testCases,
    int useSubkernels, int threadsPerBlock
)
{
    int numWeights = getNumNetworkWeights(numLayers, layerSizes);
    int inDataWidth = layerSizes[0];

    float *d_weights = 0;
    int *d_layerSizes = 0;
    float *d_trainData = 0;
    float *d_trueValues = 0;
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
    cudaMalloc(&d_nodeErrors, sizeof(float) * getNumErrorNodes(numLayers, layerSizes) * numBlocks * threadsPerBlock);
    cudaMalloc(&d_nodeValues, sizeof(float) * getNumValueNodes(numLayers, layerSizes) * numBlocks * threadsPerBlock);
    cudaMalloc(&d_scratchWeights, sizeof(float) * batchSize * numWeights);

    cudaMemcpy(d_layerSizes, layerSizes, sizeof(int) * numLayers, cudaMemcpyHostToDevice);

    cudaEvent_t globalStart, globalStop, start, stop;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    bool showMetrics = true;
    float msGlobal = 0;
    float msTemp = 0;
    float msMemorySetup = 0;
    float msTraining = 0;
    float msAllTraining = 0;
    float msSumming = 0;
    float msWeightUpdates = 0;
    float msTesting = 0;
    struct timeval t1;
    struct timeval t2;
    float epochAccuracies[numEpochs];

    cudaEventRecord(globalStart);
    for (int epochIndex = 0; epochIndex < numEpochs; epochIndex++)
    {

        gettimeofday(&t1, NULL);

        cudaMemcpy(d_weights, weights, sizeof(float) * numWeights, cudaMemcpyHostToDevice);

        for (int batchNumber = 0; batchNumber < numBatches; batchNumber ++)
        {
            cudaEventRecord(start);

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

            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&msTemp, start, stop);
            msMemorySetup += msTemp;

            cudaEventRecord(start);
            trainNetworkGpu<<<numBlocks, threadsPerBlock>>>(
                d_weights, numLayers, d_layerSizes,
                d_trainData, thisBatchNumSamples, internalIterations,
                d_trueValues, learnRate,
                d_nodeErrors, d_nodeValues, d_scratchWeights,
                useSubkernels
            );
            cudaEventRecord(stop);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            cudaEventElapsedTime(&msTemp, start, stop);
            msTraining += msTemp;

            cudaEventRecord(start);
            // add up the weight delta vectors
            sumVectors<<<32, 1024>>>(d_scratchWeights, thisBatchNumSamples, numWeights);
            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&msTemp, start, stop);
            msSumming += msTemp;

            if (debug)
            {
                printf("start adding deltas\n");
            }

            struct timeval t1_weightUpdates;
            struct timeval t2_weightUpdates;
            gettimeofday(&t1_weightUpdates, NULL);

            int applyWeightsThreadsPer = 1024;
            int applyWeightsNumBlocks = (int)ceil((float)numWeights / (float)applyWeightsThreadsPer);
            applyBatchWeightUpdate<<<applyWeightsNumBlocks, applyWeightsThreadsPer>>>(
                d_weights, d_scratchWeights, numWeights
            );

            gettimeofday(&t2_weightUpdates, NULL);
            msWeightUpdates +=
                (t2_weightUpdates.tv_sec * 1000 + t2_weightUpdates.tv_usec / 1000) -
                (t1_weightUpdates.tv_sec * 1000 + t1_weightUpdates.tv_usec / 1000);

            if (debug)
            {
                printf("done adding deltas\n");
            }
        }

        cudaMemcpy(weights, d_weights, sizeof(float) * numWeights, cudaMemcpyDeviceToHost);

        gettimeofday(&t2, NULL);
        msAllTraining +=
            (t2.tv_sec * 1000 + t2.tv_usec / 1000) -
            (t1.tv_sec * 1000 + t1.tv_usec / 1000);

        printf("finished epoch %d\n", epochIndex);
        if (testCases)
        {
            gettimeofday(&t1, NULL);
            float accuracy = testNetworkGpu(weights, numLayers, layerSizes, testCases);
            epochAccuracies[epochIndex] = accuracy;
            printf("Accuracy: %.3f\n", accuracy);
            gettimeofday(&t2, NULL);
            msTesting +=
                (t2.tv_sec * 1000 + t2.tv_usec / 1000) -
                (t1.tv_sec * 1000 + t1.tv_usec / 1000);
        }
    }

    cudaEventRecord(globalStop);
    cudaEventSynchronize(globalStop);
    cudaEventElapsedTime(&msGlobal, globalStart, globalStop);


    if (showMetrics)
    {
        float totalAccountedFor =
            msTraining +
            msMemorySetup +
            msSumming +
            msTesting +
            msWeightUpdates
        ;
        printf(
            "msGlobal,"
            "msMemorySetup,"
            "msTraining,"
            "msSumming,"
            "msWeightUpdates,"
            "msAllTraining,"
            "msTesting,"
            "Total Accounted For\n"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f\n",

            msGlobal,
            msMemorySetup,
            msTraining,
            msSumming,
            msWeightUpdates,
            msAllTraining,
            msTesting,
            totalAccountedFor
        );

        printf("Epoch Accuracies:\n");
        for (int i = 0; i < numEpochs; i++)
        {
            printf("%.4f", epochAccuracies[i]);
            if (i != numEpochs - 1)
            {
                printf(",");
            }
        }
        printf("\n");
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
        while (token != NULL && tokenIndex < 28 * 28)
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

__device__ __host__ int imageSampleTrueValue(float * trueValues, int sampleIndex)
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

int __device__ __host__ imageSampleTestResult(float *trueValues, int sampleIndex, float *result)
{
    int trueValue = imageSampleTrueValue(trueValues, sampleIndex);

    int selectedValue = imageSampleResultToInt(result);

    if (trueValue == selectedValue) {
        return 1;
    }

    return 0;
}

int __device__ __host__ imageSampleResultToInt(float *result)
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
        free(result);
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
    printf("Accuracy: %.3f\n", (float)numCorrect / (float) testCases->numItems);
}

float testNetworkGpu(float *weights, int numLayers, int *layerSizes, imageTrainingSamples *testCases)
{
    int threadsPerBlock = 4;
    int numWeights = getNumNetworkWeights(numLayers, layerSizes);
    int batchSize = 2048; // this doesn't seem to affect performance as long as it's large enough
    int numBatches = (int)ceil((float)testCases->numItems / (float)batchSize);
    // The block and thread usage should be reworked. Currently it uses batchSize blocks and 1 thread per block
    int numBlocks = (int)ceil((float)batchSize / (float)threadsPerBlock); // need to check this math
    int inDataWidth = layerSizes[0];
    int outDataWidth = layerSizes[numLayers - 1];
    int results[batchSize];

    float *d_weights = 0;
    int *d_layerSizes = 0;
    float *d_testData = 0;
    float *d_trueValues = 0;
    float *d_nodeValues = 0;
    int *d_results = 0;

    cudaMalloc(&d_weights, sizeof(float) * numWeights);
    cudaMalloc(&d_layerSizes, sizeof(int) * numLayers);
    cudaMalloc(&d_testData, sizeof(float) * batchSize * inDataWidth);
    cudaMalloc(&d_trueValues, sizeof(float) * batchSize * outDataWidth);
    cudaMalloc(&d_nodeValues, sizeof(float) * getNumValueNodes(numLayers, layerSizes) * batchSize);
    cudaMalloc(&d_results, sizeof(int) * batchSize);

    cudaMemcpy(d_layerSizes, layerSizes, sizeof(int) * numLayers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(float) * numWeights, cudaMemcpyHostToDevice);

    printf("Begin testing (%d samples, %d samples per batch, %d batches, %d blocks, %d threads each)\n",
        testCases->numItems, batchSize, numBatches, numBlocks, threadsPerBlock);

    int numCorrect = 0;
    for (int batchNumber = 0; batchNumber < numBatches; batchNumber ++)
    {
        int testDataStartIndex = batchNumber * batchSize * inDataWidth;
        int trueValuesStartIndex = batchNumber * batchSize * outDataWidth;
        int thisBatchNumSamples = batchSize;
        if ((batchNumber + 1) * batchSize > testCases->numItems)
        {
            // in this case our final batch has more capacity than the number of remaining samples
            // need to copy less data in
            thisBatchNumSamples = batchSize - ((batchNumber + 1) * batchSize - testCases->numItems);
        }
        int testDataBytesToCopy = sizeof(float) * thisBatchNumSamples * inDataWidth;
        int trueValuesBytesToCopy = sizeof(float) * thisBatchNumSamples * outDataWidth;
        // copy in the samples of this batch
        cudaMemcpy(d_testData, testCases->inputSamples + testDataStartIndex, testDataBytesToCopy, cudaMemcpyHostToDevice);
        cudaMemcpy(d_trueValues, testCases->trueOutput + trueValuesStartIndex, trueValuesBytesToCopy, cudaMemcpyHostToDevice);
        classifyAndCheckSample<<<numBlocks, threadsPerBlock>>>(
            d_weights, numLayers, d_layerSizes,
            d_testData, thisBatchNumSamples,
            d_trueValues, d_nodeValues,
            d_results
        );
        cudaMemcpy(results, d_results, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

        // sum results
        for (int i = 0; i < thisBatchNumSamples; i ++)
        {
            numCorrect += results[i];
        }
    }

    return (float)numCorrect / (float) testCases->numItems;
}

__global__ void classifyAndCheckSample(
    float *weights, int numLayers, int *layerSizes,
    float *testData, int thisBatchNumSamples,
    float *trueValues, float *nodeValues,
    int *results)
{
    int testCaseIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (testCaseIndex >= thisBatchNumSamples)
    {
        return;
    }

    int numValueNodes = getNumValueNodes(numLayers, layerSizes);

    int nodeDataValuesOffset = (blockIdx.x * blockDim.x + threadIdx.x) * numValueNodes;

    // forward compute
    // start with first hidden layer
    float *nodeValuesIn = 0;
    int nodeValuesInOffset = 0;

    for (int layerIndex = 1; layerIndex < numLayers; layerIndex ++)
    {
        // first layer is just the training data
        if (layerIndex == 1)
        {
            nodeValuesIn = testData;
            nodeValuesInOffset = testCaseIndex * layerSizes[0];
        }
        else
        {
            nodeValuesIn = nodeValues;
            nodeValuesInOffset = nodeDataValuesOffset + getValueIndex(layerSizes, layerIndex - 1, 0);
        }

        int sharedMemSize = sizeof(float) * layerSizes[layerIndex - 1];
        k_updateNodeValues<<<1, layerSizes[layerIndex], sharedMemSize>>>(
            0, nodeValuesInOffset, nodeDataValuesOffset,
            weights, nodeValuesIn, nodeValues,
            layerSizes, layerIndex
        );
    }

    /*
        If the rest of the kernel needs to do operations on memory affected by a running subkernel,
        we need to wait for subkernel completion. Otherwise we'll get race conditions.
    */
    cudaDeviceSynchronize();

    int isCorrect = imageSampleTestResult(
        trueValues,
        testCaseIndex,
        nodeValues + testCaseIndex * numValueNodes + getValueIndex(layerSizes, numLayers - 1, 0)
    );

    results[testCaseIndex] = isCorrect;
}

__global__ void applyBatchWeightUpdate(float *weights, float *deltas, int numWeights)
{
    int weightIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (weightIndex > numWeights)
    {
        return;
    }

    weights[weightIndex] = weights[weightIndex] + deltas[weightIndex];
}
