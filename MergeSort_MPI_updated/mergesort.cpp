#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


/*
    This algorithm was written by racorretjer
    https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c

    It has been modifed to meet the needs of the final project

*/

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);
void checkCorrectness(int *, int);

const char* main_region = "main_region";
const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_scatter = "comm_scatter";
const char* comm_gather = "comm_gather";


int main(int argc, char** argv) {
    CALI_MARK_BEGIN(main_region);
    /********** Create and populate the array **********/
    int n = std::atoi(argv[1]);
    int *original_array = new int[n];

    /********** Initialize MPI **********/
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int input = atoi(argv[2]); // 0 for sorted, 1 for random, 2 for reverse sorted, others for perturbed
    std::srand(time(nullptr));
    std::set<int> generated_numbers;  // Use a set to store generated numbers
    
    CALI_MARK_BEGIN(whole_computation);
    CALI_MARK_BEGIN(data_init);
    if (input == 0) {
        for (int i = 0; i < n; i++) {
            original_array[i] = i;
        }
    } else if (input == 1) {
        for (int c = 0; c < n; c++) {
            int generated;
            do {
                generated = std::rand() % n;
            } while (generated_numbers.count(generated) > 0);
            generated_numbers.insert(generated);
            original_array[c] = generated;
        }
    } else if (input == 2) {
        for (int i = 0; i < n; i++) {
            original_array[i] = n - i;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                original_array[i] = static_cast<double>(rand() % 10);
            } else {
                original_array[i] = i + 1;
            }
        }
    }
    CALI_MARK_END(data_init);

    // Print the original generated array just once
    if (world_rank == 0) {
        std::cout << "This is the input array: ";
        for (int i = 0; i < n; i++) {
            std::cout << original_array[i] << " ";
        }
        std::cout << "\n\n";
    }
    //std::cout << "\n\n";

    //CALI_MARK_BEGIN(whole_computation);

    /********** Divide the array in equal-sized chunks **********/
    int size = n / world_size;

    /********** Send each subarray to each process **********/
    CALI_MARK_BEGIN(comm);
    int *sub_array = new int[size];
    CALI_MARK_BEGIN(comm_scatter);
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_scatter);
    CALI_MARK_END(comm);
    
    /********** Perform the mergesort on each process **********/
    CALI_MARK_BEGIN(comp);

    int *tmp_array = new int[size];
    CALI_MARK_BEGIN(comp_small);
    // Record the start time
    double start_time = MPI_Wtime();
    mergeSort(sub_array, tmp_array, 0, (size - 1));
    // Record the end time
    double end_time = MPI_Wtime();
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
   
   // Synchronize all processes before printing the unsorted array
   //MPI_Barrier(MPI_COMM_WORLD);
   //printing each process data
    // std::cout << "Process " << world_rank << " : ";
    // for (int c = 0; c < size; c++) {
    //   std::cout << sub_array[c] << " ";
    // }
    // std::cout << "\n\n";

    /********** Gather the sorted subarrays into one **********/
    int *sorted = nullptr;
    if (world_rank == 0) {
        sorted = new int[n];
    }
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_gather);
    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_gather);
    CALI_MARK_END(comm);
    /********** Make the final mergeSort call **********/
    if (world_rank == 0) {
        int *other_array = new int[n];
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        mergeSort(sorted, other_array, 0, (n - 1));
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        /********** Display the sorted array **********/
        std::cout << "This is the sorted array: ";
        for (int c = 0; c < n; c++) {
            std::cout << sorted[c] << " ";
        }
        std::cout << "\n\n";

        // Check the correctness of the sorted array
        CALI_MARK_BEGIN("check_correctness");
        checkCorrectness(sorted, n);
        CALI_MARK_END("check_correctness");


        /********** Clean up root **********/
        delete[] sorted;
        delete[] other_array;
    }

    /********** Clean up rest **********/
    delete[] original_array;
    delete[] sub_array;
    delete[] tmp_array;

    /********** Finalize MPI **********/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    CALI_MARK_END(whole_computation);

    //Used for the adiak data
    std::string input_string; 
    if(input == 0){
        input_string = "Sorted";
    }
    else if(input == 1){
        input_string = "Random";
    }
    else if(input == 2){
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
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", n);
    adiak::value("InputType", input_string);
    adiak::value("num_procs", world_size);
    adiak::value("group_num", 9);
    adiak::value("implementation_source", "Online");

    if (world_rank == 0) {
        std::cout << "Elapsed time (in Seconds): " << end_time - start_time << " seconds" << std::endl;
    }
    CALI_MARK_END(main_region);
}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;

    while ((h <= m) && (j <= r)) {
        if (a[h] <= a[j]) {
            b[i] = a[h];
            h++;
        } else {
            b[i] = a[j];
            j++;
        }
        i++;
    }

    if (m < h) {
        for (k = j; k <= r; k++) {
            b[i] = a[k];
            i++;
        }
    } else {
        for (k = h; k <= m; k++) {
            b[i] = a[k];
            i++;
        }
    }

    for (k = l; k <= r; k++) {
        a[k] = b[k];
    }
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
    int m;

    if (l < r) {
        m = (l + r) / 2;

        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}

// check correctness
void checkCorrectness(int *arr, int size) {
    bool sorted = true;

    for (int i = 1; i < size; i++) {
        if (arr[i - 1] > arr[i]) {
            sorted = false;
            break;
        }
    }
}