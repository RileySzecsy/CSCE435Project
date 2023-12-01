#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime> 

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance

 source: https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu
 ---- The source was used as a reference and was modified using CHATGPT to fit the requirements of this project
 */

const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_deviceToHost = "comm_DeviceToHost";
const char* comm_hostToDevice = "comm_hostToDevice";

// Generate input data based on INPUT_TYPE
void generateInputData(float* a, int INPUT_TYPE, int NUM_VALS) {
    if (INPUT_TYPE == 0) {
        for (int i = 0; i < NUM_VALS; i++) {
            a[i] = i;
        }
    } else if (INPUT_TYPE == 1) {
        srand(time(NULL));
        for (int i = 0; i < NUM_VALS; i++) {
            a[i] = rand() % NUM_VALS;  // Assuming upper limit is 100, adjust as needed
        }
    } else if (INPUT_TYPE == 2) {
        for (int i = 0; i < NUM_VALS; i++) {
            a[i] = NUM_VALS - i;
        }
    } else {
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < NUM_VALS; ++i) {
            double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                a[i] = static_cast<int>(rand() % 10);
            } else {
                a[i] = i + 1;
            }
        }
    }
}
// check correctness
void checkCorrectness(float *arr, int size) {
    //bool sorted = true;

    for (int i = 1; i < size; i++) {
        if (arr[i - 1] > arr[i]) {
            //sorted = false;
            std::cout<<"array not sorted";
            break;
        }
    }
    std::cout<<"array is sorted";
}

void merge(float arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    float L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// mergeSort for the overall array check
void mergeSort(float arr[], int l, int r) {
    if (l < r) {
        int m = l+(r-l)/2;

        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);

        merge(arr, l, m, r);
    }
}

__global__ void gpu_bottom_up_merge(float *source, float *dest, int array_size, int width, int slice)
{
    //sort on gpu and merge 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= slice) return; // out of range

    int start = width*idx*2;
    int middle = min(start + width, array_size);
    int end = min(start + 2*width, array_size);

    int i=start;
    int j=middle;
    int k;

    for(k=start; k<end; k++){
        if(i<middle && (j>=end || source[i]<source[j])){
            dest[k] = source[i];
            i++;
        }else{
            dest[k] = source[j];
            j++;
        }
    }
}

void gpu_mergesort(float *source, float *dest, int array_size, int num_processes)
{
    float *d_source, *d_dest;
    int size = array_size * sizeof(float);

    //allocate on gpu device
    cudaMalloc((void**) &d_source, size);
    cudaMalloc((void**) &d_dest, size);

    //send to gpu device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_hostToDevice);
    cudaMemcpy(d_source, source, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_hostToDevice);
    CALI_MARK_END(comm);

    //sort and merge on gpu device
    int nThreads = num_processes;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for(int width=1; width<array_size; width*=2){
        int slices = array_size/(2*width);
        int nBlocks = slices/nThreads + ((slices%nThreads)?1:0);
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        gpu_bottom_up_merge<<<nBlocks, nThreads>>>(d_source, d_dest, array_size, width, slices);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        float *temp = d_source;
        d_source = d_dest;
        d_dest = temp;
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Copy the sorted array back to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_deviceToHost);
    cudaMemcpy(dest, d_source, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_deviceToHost);
    CALI_MARK_END(comm);

    // Free the device memory
    cudaFree(d_source);
    cudaFree(d_dest);
}

int main(int argc, char** argv)
{
    //CALI_MARK_BEGIN("main");
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    srand(time(NULL));

    std::string numVals = argv[1];
    int array_size = std::stoi(numVals);
    
    std::string type = argv[2];
    const int input = std::stoi(type);
    std::string inputType;
        
    std::string processes = argv[3];
    const int numProcesses = std::stoi(processes);
    //space allocation on host for original array
    float *h_array = (float*) malloc(array_size * sizeof(float));

    CALI_MARK_BEGIN(whole_computation);
    CALI_MARK_BEGIN(data_init);
    generateInputData(h_array, input, array_size); //initiaalize the array with the number generator
    if (input == 0) {
        inputType = "Sorted";
    } else if (input == 1) {
        inputType = "Random";
    } else if (input == 2) {
        inputType = "ReverseSorted";
    } else {
        inputType = "1%perturbed";
    }
    CALI_MARK_END(data_init);

    //space allocation on host for sorted array
    float *h_sorted = (float*) malloc(array_size * sizeof(float));

    //GPU mergesort call
    gpu_mergesort(h_array, h_sorted, array_size, numProcesses);

    //CPU mergeSort call to check correctness
    mergeSort(h_sorted, 0, array_size - 1);

    CALI_MARK_BEGIN(check_correctness);
    checkCorrectness(h_sorted, array_size); //correctness check
    CALI_MARK_END(check_correctness);

    CALI_MARK_END(whole_computation);

    int slices = array_size/(2);
    int nBlocks = slices/numProcesses + ((slices%numProcesses)?1:0);
    
    // Free the host memory
    free(h_array);
    free(h_sorted);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", numProcesses); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", nBlocks); // The number of CUDA blocks 
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online + AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    mgr.stop();
    mgr.flush();
    //CALI_MARK_END("main");
}