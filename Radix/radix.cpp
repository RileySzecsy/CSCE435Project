/*

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

    for (int i = 0; i < max; ++i)
        count[i] = 0;

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

    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

// Main function to implement radix sort
void radixsort(int arr[], int n) {
    // Get the maximum element
    int max = getMax(arr, n);

    // Apply counting sort to sort elements based on place value.
    for (int place = 1; max / place > 0; place *= 10)
        countingSort(arr, n, place);
}

// Print an array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main(int argc, char* argv[]) {
    int n = 1000; // Number of elements in the randomized array
    int* arr;
    int sortedArr[n];

    int my_rank, p;
    char g_i;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    double start_time, end_time; // Variables for recording time

    if (my_rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "usage: mpirun -np <p> %s <g|i> <global_n>\n", argv[0]);
            fprintf(stderr, "   - p: the number of processes\n");
            fprintf(stderr, "   - g: generate a random, distributed list\n");
            fprintf(stderr, "   - i: user will input a list on process 0\n");
            fprintf(stderr, "   - global_n: number of elements in the global list (must be evenly divisible by p)\n");
            MPI_Finalize();
            exit(-1);
        }
        g_i = argv[1][0];
        if (g_i != 'g' && g_i != 'i') {
            fprintf(stderr, "Invalid input source. Use 'g' for generating or 'i' for inputting the list.\n");
            MPI_Finalize();
            exit(-1);
        }
        n = strtol(argv[2], NULL, 10);
        if (n % p != 0) {
            fprintf(stderr, "Global number of elements must be evenly divisible by the number of processes.\n");
            MPI_Finalize();
            exit(-1);
        }
    }

    MPI_Bcast(&g_i, 1, MPI_CHAR, 0, comm);
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    int local_n = n / p;

    arr = (int*)malloc(local_n * sizeof(int));

    // Initialize the initial array elements
    if (g_i == 'g') {
        for (int i = 0; i < local_n; i++) {
            arr[i] = rand() % 1000; // Random values between 0 and 999
        }
    } else if (g_i == 'i') {
        if (my_rank == 0) {
            printf("Enter %d elements for process 0:\n", local_n);
            for (int i = 0; i < local_n; i++) {
                scanf("%d", &arr[i]);
            }
            for (int i = 1; i < p; i++) {
                MPI_Send(arr, local_n, MPI_INT, i, 0, comm);
            }
        } else {
            MPI_Recv(arr, local_n, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    if (my_rank == 0) {
        printf("Initial array: \n");
        printArray(arr, local_n);
        printf("\n");
    }

    // Record the start time
    start_time = MPI_Wtime();

    {
        CALI_MARK_BEGIN("radix_sort");
        radixsort(arr, local_n);
        CALI_MARK_END("radix_sort");
    }

    // Check if the array is sorted in ascending order
    int sorted = 1;
    for (int i = 1; i < local_n; i++) {
        if (arr[i] < arr[i - 1]) {
            sorted = 0;
            break;
        }
    }

    // Record the end time
    end_time = MPI_Wtime();

    if (my_rank == 0) {
        printf("Sorted array: \n");
        printArray(arr, local_n);
        printf("\n");

        if (sorted) {
            printf("The array is sorted from least to greatest\n");
        } else {
            printf("The array is not sorted from least to greatest\n");
        }

        printf("Elapsed time = %e seconds\n", end_time - start_time);
    }

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
    adiak::value("InputSize", n); // The number of elements in input dataset
    adiak::value("InputType", (g_i == 'g') ? "Random" : "UserInput"); // For sorting, this would be "Random", "Sorted", "ReverseSorted", "UserInput"
    adiak::value("num_procs", p); // The number of processors (MPI ranks)
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    return 0;
}