#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <time.h>
#include <sys/time.h>
#include <set>

#define cmp(A,B) ((A)<(B))
#define nTPB 512
#define nBLK 128

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

const char* main_region = "main_region";
const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_toDevice = "comm_scatter";
const char* comm_toHost = "comm_gather";


/*
    This algorithm was found from this StackOverflow thread
    https://stackoverflow.com/questions/30729106/merge-sort-using-cuda-efficient-implementation-for-small-input-arrays

    It has been modifed to meet the needs of the final project

*/

void checkCorrectness(int *arr, int size) {
    bool sorted = true;

    for (int i = 1; i < size; i++) {
        if (arr[i - 1] > arr[i]) {
            sorted = false;
            break;
        }
    }

    if (sorted) {
        std::cout << "Array is sorted correctly!" << std::endl;
    } else {
        std::cerr << "Array is not sorted correctly!" << std::endl;
    }
}


long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

__host__ __device__ void smerge(const int * __restrict__ a, const int * __restrict__ b, int * __restrict__ c, const unsigned len_a, const unsigned len_b, const unsigned stride_a = 1, const unsigned stride_b = 1, const unsigned stride_c = 1) {
    unsigned len_c = len_a + len_b;
    unsigned nc = 0;
    unsigned na = 0;
    unsigned nb = 0;
    unsigned fa = (len_b == 0);
    unsigned fb = (len_a == 0);
    int nxta = a[0];
    int nxtb = b[0];
    while (nc < len_c) {
        if (fa) {
            c[stride_c * nc++] = nxta;
            na++;
            nxta = a[stride_a * na];
        } else if (fb) {
            c[stride_c * nc++] = nxtb;
            nb++;
            nxtb = b[stride_b * nb];
        } else if (cmp(nxta, nxtb)) {
            c[stride_c * nc++] = nxta;
            na++;
            if (na == len_a) fb++;
            else nxta = a[stride_a * na];
        } else {
            c[stride_c * nc++] = nxtb;
            nb++;
            if (nb == len_b) fa++;
            else nxtb = b[stride_b * nb];
        }
    }
}

__global__ void rmtest(const int * __restrict__ a, const int * __restrict__ b, int * __restrict__ c, int num_arr, int len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < num_arr) {
        int sel = idx * len;
        smerge(a + sel, b + sel, c + (2 * sel), len, len);
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void cmtest(const int * __restrict__ a, const int * __restrict__ b, int * __restrict__ c, int num_arr, int len, int stride_a, int stride_b, int stride_c) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < num_arr) {
        smerge(a + idx, b + idx, c + idx, len, len, stride_a, stride_b, stride_c);
        idx += blockDim.x * gridDim.x;
    }
}

int rmvalidate(int *a, int *b, int *c, int num_arr, int len) {
    int *vc = (int *)malloc(2 * len * sizeof(int));
    for (int i = 0; i < num_arr; i++) {
        thrust::merge(a + (i * len), a + ((i + 1) * len), b + (i * len), b + ((i + 1) * len), vc);
#ifndef TIMING
        for (int j = 0; j < len * 2; j++)
            if (vc[j] != c[(i * 2 * len) + j]) {
                return 0;
            }
#endif
    }
    return 1;
}

int cmvalidate(const int *c1, const int *c2, int num_arr, int len) {
    for (int i = 0; i < num_arr; i++)
        for (int j = 0; j < 2 * len; j++)
            if (c1[i * (2 * len) + j] != c2[j * (num_arr) + i]) {
                
                return 0;
            }
    return 1;
}

int main(int argc, char* argv[]) {
    CALI_MARK_BEGIN(main_region);

    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_END(comp_small);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <array_size> <input_type> <num_threads>" << std::endl;
        return 1;
    }

    int arraySize = std::stoi(argv[1]);
    int inputType = std::stoi(argv[2]);
    int numThreads = std::stoi(argv[3]);

    if (inputType < 0 || inputType > 3 || numThreads <= 0) {
        std::cerr << "Invalid input type, array size, or number of threads. Please provide valid values." << std::endl;
        return 1;
    }

    int* h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = (int*)malloc(arraySize * sizeof(int));
    h_b = (int*)malloc(arraySize * sizeof(int));
    h_c = (int*)malloc(arraySize * sizeof(int) * 2);

    CALI_MARK_BEGIN(whole_computation);
    std::set<int> uniqueElements;
    CALI_MARK_BEGIN(data_init);
    // Generate input based on the specified input type
    if (inputType == 0) {
        for (int i = 0; i < arraySize; i++) {
            h_a[i] = i;
        }
    } else if (inputType == 1) {
        std::srand(static_cast<unsigned>(std::time(0)));
        while (uniqueElements.size() < arraySize) {
            int randomValue = std::rand() % arraySize;
            uniqueElements.insert(randomValue);
        }
        int i = 0;
        for (int elem : uniqueElements) {
            h_a[i++] = elem;
        }
    } else if (inputType == 2) {
        for (int i = 0; i < arraySize; i++) {
            h_a[i] = arraySize - i;
        }
    } else {
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < arraySize; ++i) {
            double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                h_a[i] = std::rand() % 10;
            } else {
                h_a[i] = i + 1;
            }
        }
    }
    CALI_MARK_END(data_init);

    std::cout << "Original Array:" << std::endl;
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    // Sorting the input arrays
    CALI_MARK_BEGIN (comp);
    CALI_MARK_BEGIN (comp_large);
    thrust::sort(h_a, h_a + arraySize);
    thrust::sort(h_b, h_b + arraySize);
    CALI_MARK_END (comp_large);
    CALI_MARK_END (comp);

    

    CALI_MARK_BEGIN(check_correctness);
     // Check correctness after sorting
    checkCorrectness(h_a, arraySize);
    checkCorrectness(h_b, arraySize);

    CALI_MARK_END(check_correctness);

    std::cout << "Sorted Array:" << std::endl;
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    // Allocate device memory
    cudaMalloc(&d_a, (arraySize + 1) * sizeof(int));
    cudaMalloc(&d_b, (arraySize + 1) * sizeof(int));
    cudaMalloc(&d_c, arraySize * sizeof(int) * 2);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_toDevice);
    // Copy data to the device
    cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_toDevice);
    CALI_MARK_END(comm);

    // Perform row-major merge operation on the GPU
    unsigned long gtime = dtime_usec(0);
    rmtest<<<nBLK, nTPB>>>(d_a, d_b, d_c, 1, arraySize);
    cudaDeviceSynchronize();
    gtime = dtime_usec(gtime);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_toHost);
    // Copy result back to the host
    cudaMemcpy(h_c, d_c, arraySize * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_toHost);
    CALI_MARK_END(comm);
    // Validate row-major result
    unsigned long ctime = dtime_usec(0);
    if (!rmvalidate(h_a, h_b, h_c, 1, arraySize)) {
        //std::cerr << "Row-major validation failed!" << std::endl;
        return 1;
    }
    ctime = dtime_usec(ctime);
    std::cout << "CPU time: " << ctime / static_cast<float>(USECPSEC) << ", GPU RM time: " << gtime / static_cast<float>(USECPSEC) << std::endl;

    // Allocate host memory for column-major storage
    int* ch_a = (int*)malloc(arraySize * sizeof(int));
    int* ch_b = (int*)malloc(arraySize * sizeof(int));
    int* ch_c = (int*)malloc(arraySize * sizeof(int));

    // Create column-major storage
    for (int i = 0; i < arraySize; i++) {
        ch_a[i] = h_a[i];
        ch_b[i] = h_b[i];
    }

    // Copy column-major data to the device
    cudaMemcpy(d_a, ch_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, ch_b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Perform column-major merge operation on the GPU
    gtime = dtime_usec(0);
    cmtest<<<nBLK, nTPB>>>(d_a, d_b, d_c, 1, arraySize, 1, 1, 1);
    cudaDeviceSynchronize();
    gtime = dtime_usec(gtime);

    // Copy result back to the host
    cudaMemcpy(ch_c, d_c, arraySize * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU CM time: " << gtime / static_cast<float>(USECPSEC) << std::endl;
    std::cout << "Overall time: " << (ctime + gtime) / static_cast<float>(USECPSEC) << std::endl;
    
    CALI_MARK_END(whole_computation);
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(ch_a);
    free(ch_b);
    free(ch_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    uniqueElements.clear();

    CALI_MARK_END(main_region);

    std::string input_string; 
    if(inputType == 0){
        input_string = "Sorted";
    }
    else if(inputType == 1){
        input_string = "Random";
    }
    else if(inputType == 2){
        input_string = "ReverseSorted";
    }
    else{
        input_string = "1%%perturbed";
    }
     // Adiak data
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", arraySize);
    adiak::value("InputType", input_string);
    adiak::value("num_procs", numThreads);
    adiak::value("group_num", 9);
    adiak::value("implementation_source", "Online");

    return 0;

}
