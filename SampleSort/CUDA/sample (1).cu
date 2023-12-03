#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <ctime>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

int numThreads;
int numBlocks;
int numValues;

/* Data generation */
__global__ void generateData(int* dataArray, int size, int inputType) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (inputType == 0) { // Random    
        if (idx < size) {
            unsigned int x = 12345687 + idx;
            x ^= (x << 16);
            x ^= (x << 25);
            x ^= (x << 4);
            dataArray[idx] = abs(static_cast<int>(x) % size);
        }
    } else if (inputType == 1) { // Sorted
        if (idx < size) {
            dataArray[idx] = idx;
        }
    } else if (inputType == 2) { // Reverse sorted
        if (idx < size) {
            dataArray[idx] = size - 1 - idx;
        }
    } 
}

/* Main Algorithm Stuff */
// CUDA kernel to select and gather samples from the array
__global__ void selectSamples(int* array, int* samples, int size, int sampleSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (sampleSize - 1)) {
        int step = size / sampleSize;
        samples[tid] = array[((tid + 1) * step)];
    }
}

// Compare function for sorting the samples
__device__ bool compareSamples(const int& a, const int& b) {
    return a > b;
}

// CUDA kernel to sort the samples
__global__ void sortSamples(int* samples, int sampleSize) {
    if (threadIdx.x < sampleSize - 1) {
        for (int i = threadIdx.x; i < sampleSize; i++) {
            for (int j = i + 1; j < sampleSize; j++) {
                if (compareSamples(samples[i], samples[j])) {
                    int temp = samples[i];
                    samples[i] = samples[j];
                    samples[j] = temp;
                }
            }
        }
    }
}

// CUDA kernel to calculate the data offsets for grouping
__global__ void partitionDataCalculation(int* array, int* samples, int* bucketOffsets, int size, int sampleSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sampleSize) {
        int myBucket = tid;
        for (int i = 0; i < size; i++) {
            if (array[i] < samples[myBucket]) {
                atomicAdd(&bucketOffsets[myBucket], 1);
            }
        }
    }
}

// CUDA kernel to partition the data into buckets based on the samples
__global__ void partitionData(int* unsortedData, int* groupedData, int* startPosition, int* pivots, int numThreads, int numValues, int* expandedPivots, int* expandedStarts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numThreads) {
        for (int i = 0; i < numThreads - 1; i++) {
            expandedPivots[i] = pivots[i];
        }
        expandedPivots[numThreads - 1] = numValues;

        for (int i = 1; i < numThreads; i++) {
            expandedStarts[i] = startPosition[i - 1];
        }
        expandedStarts[0] = 0;

        int previousCutoff = (tid == 0) ? 0 : expandedPivots[tid - 1];

        for (int i = 0; i < numValues; i++) {
            if (unsortedData[i] < expandedPivots[tid] && unsortedData[i] >= previousCutoff) {
                groupedData[expandedStarts[tid]] = unsortedData[i];
                expandedStarts[tid]++;
            }
        }
    }
}

// CUDA kernel to sort each bucket using insertion sort
__global__ void sortBuckets(int* array, int* bucketOffsets, int size, int numValues) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int bucket = tid;
        int start = (bucket == 0) ? 0 : bucketOffsets[bucket - 1];
        int end = (bucket == (size - 1)) ? (numValues) : bucketOffsets[bucket];
        for (int i = start + 1; i < end; i++) {
            int key = array[i];
            int j = i - 1;
            while (j >= start && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }
            array[j + 1] = key;
        }
    }
}

/* Verification */
// CUDA kernel to check if the array is sorted
__global__ void checkArraySorted(int* array, bool* isSorted, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
}

/* Program main */
int main(int argc, char* argv[]) {
    int sortingType;

    sortingType = atoi(argv[3]);
    numThreads = atoi(argv[2]);
    numValues = atoi(argv[1]);
    numBlocks = numValues / numThreads;

    printf("Input sorting type: %d\n", sortingType);
    printf("Number of threads: %d\n", numThreads);
    printf("Number of values: %d\n", numValues);
    printf("Number of blocks: %d\n", numBlocks);

    CALI_MARK_BEGIN("main");

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN("data_init");
    /* Data generation */
    int* d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void**)&d_unsortedArray, numValues * sizeof(int));
    if(sortingType < 3){
        generateData<<<numBlocks, numThreads>>>(d_unsortedArray, numValues, sortingType);
    }else{
        
        // Perturbed array
        std::srand(static_cast<unsigned>(std::time(0)));
        // Fill the array with perturbed values
        for (int i = 0; i < numValues; ++i) {
            // Generate a random number between 0 and 1
            double randomValue = static_cast<double>(rand()) / RAND_MAX;

            // Check if the current element should be perturbed
            if (randomValue < 0.01) { // 1% perturbation
                // Perturb the value by adding a small random integer value
                d_unsortedArray[i] = static_cast<int>(rand() % 10); // You can adjust the perturbation range
            } else {
                // Assign a regular value
                d_unsortedArray[i] = i + 1; // You can replace this with any desired value assignment
            }
         }
    
    }
    cudaDeviceSynchronize();
    CALI_MARK_END("data_init");

    /* Main Algorithm */
    int sampleSize = numThreads;  // Number of samples
    int* d_samples;
    int* d_bucketOffsets;
    int* d_groupedData;
    int* d_expandedPivots;
    int* d_expandedStarts;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_samples, (sampleSize - 1) * sizeof(int));
    cudaMalloc((void**)&d_bucketOffsets, sampleSize * sizeof(int));
    cudaMalloc((void**)&d_groupedData, numValues * sizeof(int));
    cudaMalloc((void**)&d_expandedPivots, numThreads * sizeof(int));
    cudaMalloc((void**)&d_expandedStarts, numThreads * sizeof(int));

    // Launch the kernel to select and gather samples
    CALI_MARK_BEGIN("computation");
    CALI_MARK_BEGIN("comp_small");
    selectSamples<<<numBlocks, numThreads>>>(d_unsortedArray, d_samples, numValues, sampleSize);

    // Launch the kernel to sort the samples
    sortSamples<<<1, 1>>>(d_samples, (sampleSize - 1));

    // Launch the kernel to count the data in each bucket
    partitionDataCalculation<<<numBlocks, numThreads>>>(d_unsortedArray, d_samples, d_bucketOffsets, numValues, (sampleSize - 1));
    CALI_MARK_END("comp_small");

    CALI_MARK_BEGIN("comp_large");
    // Launch the kernel to partition the data into buckets
    partitionData<<<numBlocks, numThreads>>>(d_unsortedArray, d_groupedData, d_bucketOffsets, d_samples, numThreads, numValues, d_expandedPivots, d_expandedStarts);

    // Launch the kernel to sort each bucket using insertion sort
    sortBuckets<<<numBlocks, numThreads>>>(d_groupedData, d_bucketOffsets, numThreads, numValues);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("computation");

    // Copy data back to the host
    int sortedArray[numValues];
    CALI_MARK_BEGIN("communication");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(sortedArray, d_groupedData, numValues * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("communication");

    CALI_MARK_BEGIN("correctness_check");
    /* Verify Correctness */
    bool isSorted[numValues - 1];
    bool* d_isSorted;
    cudaMalloc((void**)&d_isSorted, (numValues - 1) * sizeof(bool));
    checkArraySorted<<<numBlocks, numThreads>>>(d_groupedData, d_isSorted, numValues);
    cudaDeviceSynchronize();

    cudaMemcpy(isSorted, d_isSorted, (numValues - 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify if the array is sorted
    bool sorted = true;
    for (int i = 0; i < numValues - 1; i++) {
        if (!isSorted[i]) {
            sorted = false;
            break;
        }
    }
    CALI_MARK_END("correctness_check");

    CALI_MARK_BEGIN("communication");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("communication");

    // Free GPU memory
    cudaFree(d_samples);
    cudaFree(d_bucketOffsets);
    cudaFree(d_unsortedArray);
    cudaFree(d_groupedData);
    cudaFree(d_expandedPivots);
    cudaFree(d_expandedStarts);
    cudaFree(d_isSorted);

    CALI_MARK_END("main");

    if (sorted) {
        printf("sorted\n");
    } else {
        printf("not sorted\n");
    }

    string inputType;
    if (sortingType == 0) {
        inputType = "Randomized";
    } else if (sortingType == 1) {
        inputType = "Sorted";
    } else if (sortingType == 2) {
        inputType = "Reverse Sorted";
    }else {
        inputType = "1%";
    }

    adiak::init(NULL);
    adiak::launchdate();           // launch date of the job
    adiak::libraries();            // Libraries used
    adiak::cmdline();              // Command line used to launch the job
    adiak::clustername();          // Name of the cluster
    adiak::value("Algorithm", "SampleSort");        // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");        // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");  
    adiak::value("SizeOfDatatype", sizeof(int));     // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", numValues);            // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);            // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1% perturbed"
    adiak::value("num_threads", numThreads);         // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", numBlocks);           // The number of CUDA blocks 
    adiak::value("group_num", 9);                   // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");     // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}