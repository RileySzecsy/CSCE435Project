#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <caliper/cali.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, static_cast<unsigned int>(result), cudaGetErrorString(result), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

const int BLOCK_SIZE = 512;

CALI_CXX_MARK_FUNCTION;

__global__ void sampleSort(int *data, int size, int *sortedData) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Sort the local data using quicksort

    if (tid < size) {
        // You can replace this with a more efficient sorting algorithm
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
    }

    __syncthreads();

    // Select pivots

    int pivots[BLOCK_SIZE];

    if (tid == 0) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            pivots[i] = data[i * size / BLOCK_SIZE];
        }
    }

    __syncthreads();

    // Broadcast pivots to all threads
    for (int i = 0; i < BLOCK_SIZE; i++) {
        pivots[i] = __shfl_sync(0xFFFFFFFF, pivots[i], 0);
    }

    __syncthreads();

    // Perform partitioning
    int lowerCount = 0;
    int higherCount = 0;

    for (int i = 0; i < size; i++) {
        if (data[i] < pivots[tid]) {
            lowerCount++;
        } else if (data[i] > pivots[tid]) {
            higherCount++;
        }
    }

    __syncthreads();

    // Perform global prefix sum to find starting index for each partition
    if (tid < BLOCK_SIZE) {
        pivots[tid] = lowerCount;
    }

    __syncthreads();

    for (int i = 1; i < BLOCK_SIZE; i <<= 1) {
        int val = 0;

        if (tid >= i) {
            val = pivots[tid - i];
        }

        __syncthreads();

        if (tid >= i) {
            pivots[tid] += val;
        }

        __syncthreads();
    }

    __syncthreads();
    

    // Move elements to their respective partitions

    if (tid < size) {
        if (data[tid] < pivots[tid % BLOCK_SIZE]) {
            sortedData[pivots[tid % BLOCK_SIZE]++] = data[tid];
        } else if (data[tid] > pivots[tid % BLOCK_SIZE]) {
            sortedData[pivots[tid % BLOCK_SIZE] + lowerCount++] = data[tid];
        }
    }

}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");

    if (argc != 2) {
        printf("Usage: %s <NoofElements>\n", argv[0]);
        return -1;
    }

    int NoofElements = atoi(argv[1]);

    int *h_data = (int *)malloc(NoofElements * sizeof(int));
    int *d_data, *d_sortedData;

    // Initialize data on the host
    CALI_MARK_BEGIN("data_init");
    for (int i = 0; i < NoofElements; i++) {
        h_data[i] = rand() % 100;  // Modify this according to your input data
    }
    CALI_MARK_END("data_init");

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void **)&d_data, NoofElements * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_sortedData, NoofElements * sizeof(int)));

    // Copy data from host to device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CUDA_CHECK(cudaMemcpy(d_data, h_data, NoofElements * sizeof(int), cudaMemcpyHostToDevice));
    CALI_MARK_END("comm_large");
CALI_MARK_END("comm");
    // Launch CUDA kernel for sample sort
    int blocks = (NoofElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
    sampleSort<<<blocks, BLOCK_SIZE>>>(d_data, NoofElements, d_sortedData);
      CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy sorted data back to host
    int *h_sortedData = (int *)malloc(NoofElements * sizeof(int));
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CUDA_CHECK(cudaMemcpy(h_sortedData, d_sortedData, NoofElements * sizeof(int), cudaMemcpyDeviceToHost));
    CALI_MARK_END("comm_large");
CALI_MARK_END("comm");
    // Print the sorted sequence
    CALI_MARK_BEGIN("correctness_check");
    printf("Number of Elements to be sorted: %d\n", NoofElements);
    printf("The original sequence is:\n");
    for (int i = 0; i < NoofElements; i++) {
        printf("%d   ", h_data[i]);
    }
    printf("\n");

    printf("The sorted sequence is:\n");
    for (int i = 0; i < NoofElements; i++) {
        printf("%d   ", h_sortedData[i]);
    }
    printf("\n");
    CALI_MARK_END("correctness_check");

    // Free host and device memory
    free(h_data);
    free(h_sortedData);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_sortedData));

    CALI_MARK_END("main");

    return 0;
}
