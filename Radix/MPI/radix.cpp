#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* check_correctness = "check_correctness";
const char* mpi_gather = "MPI_Gather";
const char* mpi_recv = "MPI_Recv";
const char* mpi_scatter = "MPI_Scatter";
const char* mpi_send = "MPI_Send";

bool isSorted(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

int getMax(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void countingSort(int arr[], int n, int place) {
    const int max = 10;
    int* output = new int[n]();
    int* count = new int[max]();

    CALI_MARK_BEGIN("comm_large");

    for (int i = 0; i < n; i++)
        count[(arr[i] / place) % 10]++;

    for (int i = 1; i < max; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / place) % 10] - 1] = arr[i];
        count[(arr[i] / place) % 10]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    CALI_MARK_END("comm_large");

    delete[] output;
    delete[] count;
}

void radixsort(int arr[], int n) {
    CALI_MARK_BEGIN("comp_large");

    int max = getMax(arr, n);

    for (int place = 1; max / place > 0; place *= 10) {
        countingSort(arr, n, place);
    }

    CALI_MARK_END("comp_large");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    CALI_MARK_BEGIN("main");

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <array_size> <input_type> <num_processes>" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int array_size = atoi(argv[1]);
    int input_type = atoi(argv[2]);
    int num_processes = atoi(argv[3]);
    int upper_limit = 250;

    CALI_MARK_BEGIN("whole_computation");

    CALI_MARK_BEGIN("data_init");

    int* a = new int[array_size]();

    if (input_type == 0) {
        for (int i = 0; i < array_size; i++) {
            a[i] = i;
        }
    } else if (input_type == 1) {
        srand(time(NULL));
        for (int i = 0; i < array_size; i++) {
            a[i] = rand() % upper_limit;
        }
    } else if (input_type == 2) {
        for (int i = 0; i < array_size; i++) {
            a[i] = array_size - i;
        }
    } else {
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < array_size; ++i) {
            double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                a[i] = static_cast<int>(rand() % 10);
            } else {
                a[i] = i + 1;
            }
        }
    }

    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comp");

    radixsort(a, array_size);

    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");

    int* globalArr = nullptr;

    if (num_processes > 1) {
        globalArr = new int[array_size * num_processes]();
    }

    MPI_Gather(a, array_size, MPI_INT, globalArr, array_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (num_processes > 1) {
        radixsort(globalArr, array_size * num_processes);
    }

    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("check_correctness");

    std::string input_string;
    if (input_type == 0) {
        input_string = "Sorted";
    } else if (input_type == 1) {
        input_string = "Random";
    } else if (input_type == 2) {
        input_string = "ReverseSorted";
    } else {
        input_string = "1%%perturbed";
    }

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Radix Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", array_size);
    adiak::value("InputType", input_string);
    adiak::value("num_procs", num_processes);
    adiak::value("group_num", 9);
    adiak::value("implementation_source", "Online");

    CALI_MARK_END("check_correctness");

    if (num_processes > 1) {
        int* resultArr = new int[array_size]();
        MPI_Scatter(globalArr, array_size, MPI_INT, resultArr, array_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (isSorted(resultArr, array_size)) {
            printf("Array is sorted from least to greatest.\n");
        } else {
            printf("Array is not sorted.\n");
        }

        delete[] resultArr;
    } else {
        if (isSorted(a, array_size)) {
            printf("Array is sorted from least to greatest.\n");
        } else {
            printf("Array is not sorted.\n");
        }
    }

    if (num_processes > 1) {
        delete[] globalArr;
    }

    delete[] a;
    
    CALI_MARK_END("whole_computation");

    MPI_Finalize();

    CALI_MARK_END("main");

    return 0;
}