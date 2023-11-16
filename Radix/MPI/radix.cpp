/*

Jonathan Kutsch

Radix Sort implemented with the help of the following resources:
    https://www.geeksforgeeks.org/radix-sort/
    https://researchwith.njit.edu/en/publications/partitioned-parallel-radix-sort
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.79.4515&rep=rep1&type=pdf
    https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf

 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdbool.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function to get the largest element from an array
int getMax(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Using counting sort to sort the elements based on significant places
void countingSort(int arr[], int n, int place) {
    const int max = 10;
    int output[n];
    int count[max];

    // Initialize count array
    for (int i = 0; i < max; ++i)
        count[i] = 0;

    CALI_MARK_BEGIN("counting_sort_small");  // New Caliper region for counting sort small computation

    // Calculate the count of elements
    for (int i = 0; i < n; i++)
        count[(arr[i] / place) % 10]++;

    // Calculate the cumulative count
    for (int i = 1; i < max; i++)
        count[i] += count[i - 1];

    // Place the elements in sorted order
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / place) % 10] - 1] = arr[i];
        count[(arr[i] / place) % 10]--;
    }

    // Copy the sorted elements back to the original array
    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    CALI_MARK_END("counting_sort_small");  // End of the Caliper region for counting sort small computation
}

// Radix Sort implementation
void radixsort(int arr[], int n) {
    CALI_MARK_BEGIN("radix_sort_whole_computation");  // New Caliper region for the whole radix sort computation

    // Get the maximum element
    int max = getMax(arr, n);

    // Apply counting sort to sort elements based on place value
    for (int place = 1; max / place > 0; place *= 10) {
        CALI_MARK_BEGIN("radix_sort_counting_sort");  // Caliper region for counting sort within radix sort
        countingSort(arr, n, place);
        CALI_MARK_END("radix_sort_counting_sort");
    }

    CALI_MARK_END("radix_sort_whole_computation");  // End of the Caliper region for the whole radix sort computation
}

// Function to check if an array is sorted in ascending order
bool isSorted(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

// Function to print an array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main(int argc, char* argv[]) {
    int n = 1000; // Number of elements in the randomized array
    int* arr;
    
    int my_rank, p;
    char g_i;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    double start_time, end_time; // Variables for recording time

    CALI_MARK_BEGIN("data_init");
    if (my_rank == 0) {
        // Command line argument validation
        if (argc != 3) {
            fprintf(stderr, "usage: mpirun -np <p> %s <g|i> <global_n>\n", argv[0]);
            fprintf(stderr, "   - p: the number of processes\n");
            fprintf(stderr, "   - g: generate a random, distributed list\n");
            fprintf(stderr, "   - i: user will input a list on process 0\n");
            fprintf(stderr, "   - global_n: number of elements in the global list (must be evenly divisible by p)\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Parse command line arguments
        g_i = argv[1][0];
        if (g_i != 'g' && g_i != 'i') {
            fprintf(stderr, "Invalid input source. Use 'g' for generating or 'i' for inputting the list.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        n = strtol(argv[2], NULL, 10);
        if (n % p != 0) {
            fprintf(stderr, "Global number of elements must be evenly divisible by the number of processes.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    // Broadcast command line arguments
    MPI_Bcast(&g_i, 1, MPI_CHAR, 0, comm);
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    int local_n = n / p;

    // Allocate memory for local array
    arr = (int*)malloc(local_n * sizeof(int));

    CALI_MARK_BEGIN("comm");
    // Initialize the initial array elements
    if (g_i == 'g') {
        for (int i = 0; i < local_n; i++) {
            arr[i] = rand() % 1000; // Random values between 0 and 999
        }
    } else if (g_i == 'i') {
        // User input on process 0
        if (my_rank == 0) {
            CALI_MARK_BEGIN("comm_large");
            printf("Enter %d elements for process 0:\n", local_n);
            for (int i = 0; i < local_n; i++) {
                scanf("%d", &arr[i]);
            }
            // Distribute portions of the input to other processes
            CALI_MARK_BEGIN("comm_small");
            for (int i = 1; i < p; i++) {
                MPI_Send(&arr[i * local_n], local_n, MPI_INT, i, 0, comm);
            }
            CALI_MARK_END("comm_small");
            CALI_MARK_END("comm_large");
        } else {
            // Receive the portion of the input on other processes
            MPI_Recv(arr, local_n, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        }
    }
    CALI_MARK_END("comm");

    // Print the initial array on process 0
    if (my_rank == 0) {
        printf("Initial array: \n");
        printArray(arr, local_n);
        printf("\n");
    }

    // Record the start time
    start_time = MPI_Wtime();

    {
        CALI_MARK_BEGIN("radix_sort_main");  // New Caliper region for the main radix sort operation
        radixsort(arr, local_n);
        CALI_MARK_END("radix_sort_main");
    }

    // Record the end time
    end_time = MPI_Wtime();

    // Print the sorted array on process 0
    if (my_rank == 0) {
        printf("Sorted array: \n");
        printArray(arr, local_n);
        printf("\n");
    }

    // Check if the array is sorted in ascending order
    bool sorted = isSorted(arr, local_n);

    // Check if the array is sorted in ascending order on process 0
    CALI_MARK_BEGIN("check_correctness");
    if (my_rank == 0) {
        if (sorted) {
            printf("The array is sorted from least to greatest\n");
        } else {
            printf("The array is not sorted from least to greatest\n");
        }

        printf("Elapsed time = %e seconds\n", end_time - start_time);
    }
    CALI_MARK_END("check_correctness");

    // Free allocated memory and finalize MPI
    free(arr);
    MPI_Finalize();

    // Adiak data
    adiak::init(NULL);
    adiak::launchdate();  // launch date of the job
    adiak::libraries();   // Libraries used
    adiak::cmdline();     // Command line used to launch the job
    adiak::clustername(); // Name of the cluster
    adiak::value("Algorithm", "Radix Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in the input dataset
    adiak::value("InputType", (g_i == 'g') ? "Random" : "UserInput"); // For sorting, this would be "Random", "Sorted", "ReverseSorted", "UserInput"
    adiak::value("num_procs", p); // The number of processors (MPI ranks)
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    return 0;
}