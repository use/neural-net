#include "nnUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

void testGetIndex();
void testGetNumNetworkWeights();
void testGetNumValueNodes();
void testGetValueIndex();
void testGetNumErrorNodes();
void testGetErrorIndex();
void testCreateNetwork();

int main(int argc, char *argv[])
{
    testGetIndex();
    testGetNumNetworkWeights();
    testGetNumValueNodes();
    testGetValueIndex();
    testGetNumErrorNodes();
    testGetErrorIndex();
    testCreateNetwork();

    printf("All assertions passed.");
}

void testGetIndex()
{
    int layerSizes[] = {2, 3, 2};
    assert(getIndex(1, 0, 0, layerSizes) == 0);
    assert(getIndex(1, 2, 2, layerSizes) == 8);
    assert(getIndex(2, 1, 1, layerSizes) == 14);

    int layerSizes2[] = {784, 20, 10};
    assert(getIndex(1, 15, 201, layerSizes2) == 11976);
    assert(getIndex(2, 7, 12, layerSizes2) == 15859);
}

void testGetNumNetworkWeights()
{
    int layerSizes[] = {2, 3, 2};
    assert(getNumNetworkWeights(3, layerSizes) == 17);

    int layerSizes2[] = {784, 20, 10};
    assert(getNumNetworkWeights(3, layerSizes2) == 15910);

    int layerSizes3[] = {558, 85, 238, 12};
    assert(getNumNetworkWeights(4, layerSizes3) == 70851);
}

void testGetNumValueNodes()
{
    int layerSizes[] = {2, 3, 2};
    assert(getNumValueNodes(3, layerSizes) == 2 + 3 + 2);

    int layerSizes3[] = {558, 85, 238, 12};
    assert(getNumValueNodes(4, layerSizes3) == 558 + 85 + 238 + 12);
}

void testGetValueIndex()
{
    int layerSizes[] = {2, 3, 2};
    assert(getValueIndex(layerSizes, 2, 0) == 5);
    int layerSizes3[] = {558, 85, 238, 12};
    assert(getValueIndex(layerSizes3, 2, 27) == 558 + 85 + 27);
}

void testGetNumErrorNodes()
{
    int layerSizes[] = {2, 3, 2};
    assert(getNumErrorNodes(3, layerSizes) == 3 + 2);

    int layerSizes3[] = {558, 85, 238, 12};
    assert(getNumErrorNodes(4, layerSizes3) == 85 + 238 + 12);
}

void testGetErrorIndex()
{
    int layerSizes[] = {2, 3, 2};
    assert(getErrorIndex(layerSizes, 2, 0) == 3);
    int layerSizes3[] = {558, 85, 238, 12};
    assert(getErrorIndex(layerSizes3, 2, 27) == 85 + 27);
}


void testCreateNetwork()
{
    int layerSizes[] = {784, 20, 10};
    float *net = createNetwork(3, layerSizes);
    assert(net[getIndex(2, 7, 12, layerSizes)] == 15859);

    int layerSizes2[] = {558, 85, 238, 12};
    float *net2 = createNetwork(4, layerSizes2);
    assert(net2[52134] == 52134);
}
