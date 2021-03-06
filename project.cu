#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <getopt.h>

void testAndFunction();
void testAndFunctionGpu();
void testTonyFunction();
void testTonyFunctionGpu();
void testImageDataParsing();
void testImageTrainingGpu(int numHidden,
    int numSamples,
    int numTestCases,
    int numEpochs,
    int batchSize,
    float learnRate,
    int useSubkernels,
    int threadsPerBlock,
    int hiddenLayers,
    trainingOptions options
);
void testImageTraining(int numHidden, int numSamples, int numTestCases, int numEpochs, float learnRate, int hiddenLayers);
void testImageSampleTestResult();
void freeImageTrainingSamples(imageTrainingSamples *samples);
void usage();

int numNeuronsDefault = 20;
int numSamplesDefault = 10000;
int numTestCasesDefault = 1000;
int numEpochsDefault = 1;
int batchSizeDefault = 64;
float learnRateDefault = .05;
int useSubkernelsDefault = 1;
int threadsPerBlockDefault = 1;
int hiddenLayersDefault = 1;
int printBatchFinishDefault = false;

int main(int argc, char *argv[])
{
    // parse arguments
    int opt;

    int numNeurons = numNeuronsDefault;
    int numSamples = numSamplesDefault;
    int numTestCases = numTestCasesDefault;
    int numEpochs = numEpochsDefault;
    int batchSize = batchSizeDefault;
    float learnRate = learnRateDefault;
    int useSubkernels = useSubkernelsDefault;
    int threadsPerBlock = threadsPerBlockDefault;
    int hiddenLayers = hiddenLayersDefault;
    char type = 'g';
    struct trainingOptions options;
    options.printBatchFinish = printBatchFinishDefault;

    static struct option long_options[] = {
        {"print-batch-finish", no_argument, NULL, 100},
        {0, 0, 0, 0}
    };
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "n:t:v:e:b:l:s:p:i:",
        long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 100:
                options.printBatchFinish = true;
                break;
            case 'n':
                numNeurons = atoi(optarg);
                break;
            case 't':
                numSamples = atoi(optarg);
                break;
            case 'v':
                numTestCases = atoi(optarg);
                break;
            case 'e':
                numEpochs = atoi(optarg);
                break;
            case 'b':
                batchSize = atoi(optarg);
                break;
            case 'l':
                learnRate = (float)atof(optarg);
                break;
            case 's':
                useSubkernels = atoi(optarg);
                if (useSubkernels != 0 && useSubkernels != 1)
                {
                    usage();
                }
                break;
            case 'p':
                threadsPerBlock = atoi(optarg);
                break;
            case 'i':
                hiddenLayers = atoi(optarg);
                break;
            default:
                usage();
        }
    }

    if (
        optind >= argc ||
        numNeurons < 1 ||
        numSamples < 1 ||
        numTestCases < 1 ||
        numEpochs < 1 ||
        batchSize < 1 ||
        learnRate > 1 || learnRate <= 0 ||
        hiddenLayers < 1
    )
    {
        usage();
    }

    if (0 == strcmp(argv[optind], "gpu"))
    {
        type = 'g';
    }
    else if (0 == strcmp(argv[optind], "cpu"))
    {
        type = 'c';
    } else {
        usage();
    }

    printf("Hidden Layer Neurons (per layer): %d\n", numNeurons);
    printf("Hidden Layers: %d\n", hiddenLayers);
    printf("Training Samples: %d\n", numSamples);
    printf("Test Cases: %d\n", numTestCases);
    printf("Epochs: %d\n", numEpochs);
    printf("Learning Rate: %f\n", learnRate);

    if (type == 'g')
    {
        printf("Batch Size: %d\n", batchSize);
        printf("Training with GPU\n");
        if (!useSubkernels)
        {
            printf("Subkernels DISABLED\n");
        }
        testImageTrainingGpu(
            numNeurons,
            numSamples,
            numTestCases,
            numEpochs,
            batchSize,
            learnRate,
            useSubkernels,
            threadsPerBlock,
            hiddenLayers,
            options
        );
    }
    else if (type == 'c')
    {
        printf("Training with CPU\n");
        testImageTraining(numNeurons, numSamples, numTestCases, numEpochs, learnRate, hiddenLayers);
    }

    // these features were used during development but not really used now
    // maybe they could be moved to a test file

    // testAndFunctionGpu();
    // testTonyFunction();
    // testTonyFunctionGpu();
    // testImageDataParsing();
    // testImageSampleTestResult();
}

void usage()
{
    printf("Argument should be either \"gpu\" or \"cpu\" to run that type of training.\n");
    printf("  -n <number> Use <number> neurons in the hidden layer (default %d)\n",
        numNeuronsDefault);
    printf("  -t <number> Use <number> training samples (default %d)\n",
        numSamplesDefault);
    printf("  -v <number> Use <number> verification (testing) samples (default %d)\n",
        numTestCasesDefault);
    printf("  -e <number> Run <number> epochs (default %d)\n",
        numEpochsDefault);
    printf("  -l <number> Learning rate (default %f)\n",
        learnRateDefault);
    printf("  -b <number> Use <number> batch size (default %d) (\"gpu\" only)\n",
        batchSizeDefault);
    printf("  -s <0/1> Use subkernels during training (default %d) (\"gpu\" only)\n",
        useSubkernelsDefault);
    printf("  -p <number> Use threads per block in training kernel grid (default %d) (\"gpu\" only)\n",
        useSubkernelsDefault);
    printf("  --print-batch-finish Show a message after every batch completes (\"gpu\" only)\n");
    printf("Program defaults:\n");
    printf("  Hidden Layer Neurons: %d\n", numNeuronsDefault);
    printf("  Training Samples: %d\n", numSamplesDefault);
    printf("  Test Cases: %d\n", numTestCasesDefault);
    printf("  Epochs: %d\n", numEpochsDefault);
    printf("  Learning Rate: %f\n", learnRateDefault);
    printf("  Batch Size: %d\n", batchSizeDefault);
    printf("  Subkernels on/off: %d\n", useSubkernelsDefault);
    printf("  Threads Per Block: %d\n", threadsPerBlockDefault);

    exit(EXIT_FAILURE);
}

void testImageTraining(int numHidden, int numSamples, int numTestCases, int numEpochs, float learnRate, int hiddenLayers)
{
    int numLayers = hiddenLayers + 2;

    int layerSizes[numLayers];
    // input layer
    layerSizes[0] = 28 * 28;
    // hidden layers
    for (int i = 1; i < numLayers; i ++)
    {
        layerSizes[i] = numHidden;
    }
    // output layer
    layerSizes[numLayers - 1] = 10;
    float *weights = createNetwork(numLayers, layerSizes);
    initNetworkWeights(weights, numLayers, layerSizes);
    printf("Initialized weights\n");

    char filePath[] = "data/mnist_train.csv";
    imageTrainingSamples *samples = getImageData(filePath, numSamples, 0);
    printf("Got training data\n");

    char testFilePath[] = "data/mnist_test.csv";
    imageTrainingSamples *testCases = getImageData(testFilePath, numTestCases, 0);

    struct timeval t1;
    struct timeval t2;
    float msAllTraining = 0;
    float trainingTime = 0;

    for (int epochIndex = 0; epochIndex < numEpochs; epochIndex ++)
    {
        gettimeofday(&t1, NULL);

        trainNetwork(
            weights, numLayers, layerSizes,
            samples->inputSamples, numSamples,
            1, samples->trueOutput, learnRate,
            &trainingTime
        );

        gettimeofday(&t2, NULL);
        msAllTraining +=
            (t2.tv_sec * 1000 + t2.tv_usec / 1000) -
            (t1.tv_sec * 1000 + t1.tv_usec / 1000);

        printf("Done training epoch %d\n", epochIndex);

        testNetwork(weights, numLayers, layerSizes, testCases);
    }

    printf("msTraining: %.0f\n", trainingTime);
    printf("msAllTraining: %.0f\n", msAllTraining);

    free(weights);

    freeImageTrainingSamples(samples);
    freeImageTrainingSamples(testCases);
}

void freeImageTrainingSamples(imageTrainingSamples *samples)
{
    free(samples->inputSamples);
    free(samples->trueOutput);
    free(samples);
}

void testImageSampleTestResult()
{
    float trueValues[] = {
        0, 1, 0, 0, 0,   0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,   1, 0, 0, 0, 0,
    };

    float result1[] = {
        0, .8, .3, 0, 0,   0, 0, 0, 0, 0
    };
    assert(1 == imageSampleTestResult(trueValues, 0, result1));

    float result2[] = {
        0, 0, .2, 0, 0,   0, 0, .5, 0, 0
    };
    assert(1 == imageSampleTestResult(trueValues, 1, result2));

    float result3[] = {
        .5, .5, .5, .5, .5,   7, .5, .5, .5, .5
    };
    assert(1 == imageSampleTestResult(trueValues, 2, result3));

    float result4[] = {
        0, 0, 0, 0, 0,   0, 1, 0, 0, 0
    };
    assert(0 == imageSampleTestResult(trueValues, 2, result4));
}

void testImageTrainingGpu(
    int numHidden,
    int numSamples,
    int numTestCases,
    int numEpochs,
    int batchSize,
    float learnRate,
    int useSubkernels,
    int threadsPerBlock,
    int hiddenLayers,
    trainingOptions options
)
{
    int numLayers = hiddenLayers + 2;

    int layerSizes[numLayers];
    // input layer
    layerSizes[0] = 28 * 28;
    // hidden layers
    for (int i = 1; i < numLayers; i ++)
    {
        layerSizes[i] = numHidden;
    }
    // output layer
    layerSizes[numLayers - 1] = 10;

    float *weights = createNetwork(numLayers, layerSizes);
    initNetworkWeights(weights, numLayers, layerSizes);

    char filePath[] = "data/mnist_train.csv";
    char filePathTest[] = "data/mnist_test.csv";
    imageTrainingSamples *samples = getImageData(filePath, numSamples, 0);

    imageTrainingSamples *testCases = getImageData(filePathTest, numTestCases, 0);

    int internalIterations = 1;
    batchTrainNetworkGpu(
        weights, numLayers, layerSizes,
        samples->inputSamples, numSamples, internalIterations,
        samples->trueOutput, learnRate, batchSize,
        numEpochs, testCases,
        useSubkernels, threadsPerBlock,
        options
    );

    free(weights);
    freeImageTrainingSamples(samples);
    freeImageTrainingSamples(testCases);

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
    trainingOptions options;
    options.printBatchFinish = false;
    batchTrainNetworkGpu(
        weights, numLayers, layerSizes,
        trainData, inDataCount, internalIterations,
        trueValues, .05, 2,
        10000, NULL, 1, 1, options
    );

    printNetwork(weights, numLayers, layerSizes);

    for (int i = 0; i < inDataCount; i++)
    {
        float *sample = (float *)malloc(inDataWidth * sizeof(float));
        sample[0] = trainData[i * inDataWidth + 0];
        sample[1] = trainData[i * inDataWidth + 1];
        float *result = classify(weights, numLayers, layerSizes, sample, 0);

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
    float trainingTime = 0;
    trainNetwork(weights, numLayers, layerSizes, trainData, 4, 100001, trueValues, .05, &trainingTime);

    printNetwork(weights, numLayers, layerSizes);

    for (int i = 0; i < 4; i++)
    {
        float *sample = (float *)malloc(2 * sizeof(float));
        sample[0] = trainData[i * inDataWidth + 0];
        sample[1] = trainData[i * inDataWidth + 1];
        float *result = classify(weights, numLayers, layerSizes, sample, 0);

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
    float trainingTime = 0;
    trainNetwork(net, numLayers, layerSizes, input, 64, 100001, trueOut, .05, &trainingTime);

    for (int i = 0; i < 64; i++)
    {
        int dataIndex = i * inDataWidth;
        float *sample = (float *)malloc(inDataWidth * sizeof(float));
        sample[0] = input[dataIndex + 0];
        sample[1] = input[dataIndex + 1];
        float *result = classify(net, numLayers, layerSizes, sample, 0);

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
    trainingOptions options;
    options.printBatchFinish = false;
    batchTrainNetworkGpu(
        net, numLayers, layerSizes,
        input, inDataLength, 1,
        trueOut, .05, 64,
        100000, NULL,
        1, 1,
        options
    );

    printNetwork(net, numLayers, layerSizes);

    for (int i = 0; i < 64; i++)
    {
        int dataIndex = i * inDataWidth;
        float *sample = (float *)malloc(inDataWidth * sizeof(float));
        sample[0] = input[dataIndex + 0];
        sample[1] = input[dataIndex + 1];
        float *result = classify(net, numLayers, layerSizes, sample, 0);

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
