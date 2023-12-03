#include "mpi.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using std::vector;
using std::string;
using std::swap;

int size, num_processes;

/* Data generation */
void generate_data(vector<int> &local_data, int starting_sort_choice, int amount_to_generate, int starting_position, int my_rank) {
    if (starting_sort_choice == 0) { // Random
        srand((my_rank + 5) * (my_rank + 12) * 1235);
        for (int i = 0; i < amount_to_generate; i++) {
            local_data.push_back(rand() % size);
        }
    } else if (starting_sort_choice == 1) { // Sorted
        int end_value = starting_position + amount_to_generate;
        for (int i = starting_position; i < end_value; i++) {
            local_data.push_back(i);
        }
    } else if (starting_sort_choice == 2) { // Reverse sorted
        int start_value = size - 1 - starting_position;
        int end_value = size - amount_to_generate - starting_position;
        for (int i = start_value; i >= end_value; i--) {
            local_data.push_back(i);
        }
    }else if (starting_sort_choice == 3){
        for (int i = 0; i < amount_to_generate; ++i) {
        double randomValue = static_cast<double>(rand()) / RAND_MAX;

        if (randomValue < 0.01) {
            // 1% chance: Insert a random value between 0 and 9
            local_data.push_back(rand() % 10);
        } else {
            // 99% chance: Insert the index value
            local_data.push_back(starting_position + i);
        }
    }
    }
}

/* Sequential Quick Sort & Helpers */
int partition(int arr[], int start, int end) {
    int pivot = arr[start];

    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if (arr[i] <= pivot)
            count++;
    }

    int pivot_index = start + count;
    swap(arr[pivot_index], arr[start]);

    int i = start, j = end;

    while (i < pivot_index && j > pivot_index) {
        while (arr[i] <= pivot) {
            i++;
        }

        while (arr[j] > pivot) {
            j--;
        }

        if (i < pivot_index && j > pivot_index) {
            swap(arr[i++], arr[j--]);
        }
    }

    return pivot_index;
}

void quick_sort(int arr[], int start, int end) {
    if (start >= end)
        return;

    int p = partition(arr, start, end);

    quick_sort(arr, start, p - 1);
    quick_sort(arr, p + 1, end);
}

/* Main Algorithm */
void sample_sort(vector<int> &local_data, vector<int> &sorted_data, int my_rank) {
    /* Sample splitters */
    int num_splitters = 4; //# Sampled per node
    vector<int> sampled_splitters;
    srand(84723840);

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    for (int i = 0; i < num_splitters; i++) {
        sampled_splitters.push_back(local_data.at(rand() % local_data.size()));
    }
    CALI_MARK_END("comp_small");
    

    /* Combine splitters */
    int total_splitter_array_size = num_splitters * num_processes;
    int all_splitters[total_splitter_array_size];

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("Allgather");
    MPI_Allgather(&sampled_splitters[0], num_splitters, MPI_INT, &all_splitters[0], num_splitters, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("Allgather");
    CALI_MARK_END("comm_small");
  

    /* Sort splitters & Decide cuts */
 
    CALI_MARK_BEGIN("comp_small");
    quick_sort(all_splitters, 0, total_splitter_array_size - 1); //In-place sort

    vector<int> chosen_splitters;
    for (int i = 1; i < num_processes; i++) {
        chosen_splitters.push_back(all_splitters[i * num_splitters]);
    }
    CALI_MARK_END("comp_small");


    /* Evaluate local elements and place into buffers */
    
    CALI_MARK_BEGIN("comp_large");
    vector<vector<int>> send_buckets;
    for (int i = 0; i < num_processes; i++) { send_buckets.push_back(vector<int>()); }

    for (int i = 0; i < local_data.size(); i++) {
        int not_used = 1;
        for (int j = 0; j < chosen_splitters.size(); j++) {
            if (local_data.at(i) < chosen_splitters.at(j)) {
                send_buckets.at(j).push_back(local_data.at(i));
                not_used = 0;
                break;
            }
        }
        if (not_used) { send_buckets.at(send_buckets.size() - 1).push_back(local_data.at(i)); }
    }
    CALI_MARK_END("comp_large");
   

    /* Send/Receive Data */
    int local_bucket_sizes[num_processes];
    for (int i = 0; i < num_processes; i++) { local_bucket_sizes[i] = send_buckets.at(i).size(); }

    int target_sizes[num_processes];
   
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("Gather");
    for (int i = 0; i < num_processes; i++) {
        MPI_Gather(&local_bucket_sizes[i], 1, MPI_INT, &target_sizes[0], 1, MPI_INT, i, MPI_COMM_WORLD);
    }
    CALI_MARK_END("Gather");
    CALI_MARK_END("comm_small");
 

    int my_total_size = 0;
    for (int i = 0; i < num_processes; i++) { my_total_size += target_sizes[i]; }

    int displacements[num_processes];
    displacements[0] = 0;
    for (int i = 0; i < (num_processes - 1); i++) { displacements[i + 1] = displacements[i] + target_sizes[i]; }

    int unsorted_data[my_total_size];

    
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("Gatherv");
    for (int i = 0; i < num_processes; i++) {
        MPI_Gatherv(&send_buckets[i][0], send_buckets.at(i).size(), MPI_INT, &unsorted_data, target_sizes, displacements, MPI_INT, i, MPI_COMM_WORLD);
    }
    CALI_MARK_END("Gatherv");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    /* Sort */
   
    CALI_MARK_BEGIN("comp_large");
    quick_sort(unsorted_data, 0, my_total_size - 1);
    sorted_data.insert(sorted_data.end(), &unsorted_data[0], &unsorted_data[my_total_size]);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
}

/* Verify */
bool verify_correct(vector<int> &sorted_data, int my_rank) {
    // Verify local data is in order
    for (int i = 1; i < sorted_data.size() - 1; i++) {
        if (sorted_data.at(i - 1) > sorted_data.at(i)) {
            printf("Sorting issue\n");
            return false;
        }
    }

    // Verify my start and end line up
    int my_data_bounds[] = {sorted_data.at(0), sorted_data.at(sorted_data.size() - 1)};
    int bounds_array_size = 2 * num_processes;
    int all_data_bounds[bounds_array_size];
    MPI_Allgather(&my_data_bounds, 2, MPI_INT, &all_data_bounds, 2, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < bounds_array_size - 1; i++) {
        if (all_data_bounds[i - 1] > all_data_bounds[i]) {
            printf("Sorting bounds issue\n");
            return false;
        }
    }

    return true;
}

/* Program Main */
int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");
    int sorting_type;

    sorting_type = atoi(argv[2]);
    num_processes = atoi(argv[3]);
    size = atoi(argv[1]);

    string input_type;
    if (sorting_type == 0) {
        input_type = "Randomized";
    } else if (sorting_type == 1) {
        input_type = "Sorted";
    } else if (sorting_type == 2) {
        input_type = "Reverse Sorted";
    }

    int my_rank, num_ranks, rc;

    /* MPI Setup */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks < 2) {
        printf("Minimum of Two Tasks Required\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    // if (num_ranks != num_processes) {
    //     printf("Processes Does Not Match Ranks\n");
    //     MPI_Abort(MPI_COMM_WORLD, rc);
    //     exit(1);
    // }

    if (my_rank == 0) {
        printf("Input type: %d\n", sorting_type);
        printf("Number Processes: %d\n", num_processes);
        printf("Input Size: %d\n", size);
    }



    // Create caliper ConfigManager object
    cali::ConfigManager cali_mgr;
    cali_mgr.start();

    // Data generation
    vector<int> my_local_data;
    int amount_to_generate_myself = size / num_processes; // Should always be based around powers of 2
    CALI_MARK_BEGIN("data_initialization");
    generate_data(my_local_data, sorting_type, amount_to_generate_myself, my_rank * amount_to_generate_myself, my_rank);
    CALI_MARK_END("data_initialization");

    // Main Alg
    vector<int> sorted_data;
    sample_sort(my_local_data, sorted_data, my_rank);

    // Verification
    CALI_MARK_BEGIN("correctness_check");
    bool correct = verify_correct(sorted_data, my_rank);
    CALI_MARK_END("correctness_check");

 
    if (!correct) {
        printf("Data is not sorted.\n");
    } else {
        if (my_rank == 0) {
            printf("\nData is sorted.\n");
        }
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in the input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    cali_mgr.stop();
    cali_mgr.flush();

    MPI_Finalize();
    CALI_MARK_END("main");
}
