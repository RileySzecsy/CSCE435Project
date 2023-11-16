/*

    This code was referenced from:
    https://github.com/abdullah-taha/RadixSort-with-CUDA/tree/master

    However, I adjusted it to not just sort 8 integers, but user specified and fit the overall needs of the project.
    
    Jonathan Kutsch


*/

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>    
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

//CALI Regions
const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";

int threads;

__global__ void oddeven(int* x, int I, int n) {
    int id = blockIdx.x;
    if (I == 0 && ((id * 2 + 1) < n)) { // even phase
        if (x[id * 2] > x[id * 2 + 1]) {
            int X = x[id * 2];
            x[id * 2] = x[id * 2 + 1];
            x[id * 2 + 1] = X;
        }
    }
    if (I == 1 && ((id * 2 + 2) < n)) { // odd phase
        if (x[id * 2 + 1] > x[id * 2 + 2]) {
            int X = x[id * 2 + 1];
            x[id * 2 + 1] = x[id * 2 + 2];
            x[id * 2 + 2] = X;
        }
    }
}

bool correctness_check(int A[], int n) {
    for (int i = 0; i < n - 1; i++) {
        if (A[i] > A[i + 1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Setting up variables
    int n = atoi(argv[1]);
    threads = atoi(argv[2]);
    int input = atoi(argv[3]); // 0 is sorted, 1 is random, 2 is reverse sorted, 3 is 1% perturbed
    int* c = new int[n];
    int* a = new int[n];

    const int upperlimit = 1000; // setting upper limit on random values

    CALI_MARK_BEGIN("whole_computation");
    CALI_MARK_BEGIN("data_init");

    // Determining what type of input is happening
    if (input == 0) {
        for (int i = 0; i < n; i++) {
            a[i] = i + 1;
        }
    } else if (input == 1) {
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            a[i] = rand() % upperlimit + 1;
        }
    } else if (input == 2) {
        for (int i = 0; i < n; i++) {
            a[i] = n - i;
        }
    } else {
        std::srand(static_cast<unsigned>(std::time(0)));
        // Fill the array with random values
        for (int i = 0; i < n; ++i) {
            a[i] = rand() % upperlimit + 1;
        }
    }

    CALI_MARK_END("data_init");

    int* d;
    cudaMalloc((void**)&d, n * sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < n; i++) {
        oddeven<<<n / 2, threads>>>(d, i % 2, n);
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(c, d, n * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    bool cc = correctness_check(c, n);

    CALI_MARK_END("whole_computation");

    printf("Initial Array is:\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", a[i]);
    }
    printf("\n");

    printf("Sorted Array is:\t");
    for (int i = 0; i < n; i++) {
        printf("%d\t", c[i]);
    }
    printf("\n");

    if (cc == true) {
        printf("The array is sorted from least to greatest \n");
    } else {
        printf("The array is not sorted \n");
    }

    cudaFree(d);

    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_BEGIN("comp_small");
    CALI_MARK_END("comp_small");

    // Used for the adiak data
    std::string input_string;
    if (input == 0) {
        input_string = "Sorted";
    } else if (input == 1) {
        input_string = "Random";
    } else if (input == 2) {
        input_string = "ReverseSorted";
    } else {
        input_string = "1%%perturbed";
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Radix Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", n / 2); // The number of CUDA blocks 
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Cleanup
    delete[] a;
    delete[] c;

    return 0;
}