#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* check_correctness = "check_correctness";
//const char* main = "main";


/*
    This code was written by Jonathan Dursi
    https://stackoverflow.com/questions/23633916/how-does-mpi-odd-even-sort-work

    It has been modified to meet the requiements of our project


*/



/*
correctness_check determines if the array is sorted in least to greatest order
    - returns false if any value in the array is out of place
    - returns true if it has reached the end
*/
bool correctness_check(double A[], int n){
    for(int i = 0; i < n - 1; i++){
        if (A[i] > A[i+1]){
            return false;
        }
    }
    return true;
}


/*
    merge- does the actual sorting for the two arrays passed in
        -is the recusive call for the domerge_sort
*/
int merge(double *ina, int lena, double *inb, int lenb, double *out) {
    int i,j;
    int outcount=0;

    for (i=0,j=0; i<lena; i++) {
        while ((inb[j] < ina[i]) && j < lenb) {
            out[outcount++] = inb[j++];
        }
        out[outcount++] = ina[i];
    }
    while (j<lenb)
        out[outcount++] = inb[j++];

    return 0;
}

/*
    domerge_sort- recursive loop for the merge sorting of the local arrays a and b
        -calls the merge function to actually sort the arrays

*/
int domerge_sort(double *a, int start, int end, double *b) {
    if ((end - start) <= 1) return 0;

    int mid = (end+start)/2;
    domerge_sort(a, start, mid, b);
    domerge_sort(a, mid,   end, b);
    merge(&(a[start]), mid-start, &(a[mid]), end-mid, &(b[start]));
    for (int i=start; i<end; i++)
        a[i] = b[i];

    return 0;
}


/*
    merge_sort- way to sort local arrays
        -only 1 array is passed in instead of 2
*/
int merge_sort(int n, double *a) {
    double b[n];
    domerge_sort(a, 0, n, b);
    return 0;
}


/*
    printstat- prints about each iteration of the odd even sort

*/
void printstat(int rank, int iter, char *txt, double *la, int n) {
    printf("[%d] %s iter %d: <", rank, txt, iter);
    for (int j=0; j<n-1; j++)
        printf("%6.3lf,",la[j]);
    printf("%6.3lf>\n", la[n-1]);
}

/*
    MPI_Pairwise_Echange-  does the communication between odd/even phase partners
     -the sending rank just sends the data and waits for the results;
     -the receiving rank receives it, sorts the combined data, and returns
     -the correct half of the data.
*/
void MPI_Pairwise_Exchange(int localn, double *locala, int sendrank, int recvrank,
                           MPI_Comm comm) {


    int rank;
    double remote[localn];
    double all[2*localn];
    const int mergetag = 1;
    const int sortedtag = 2;

    MPI_Comm_rank(comm, &rank);
    if (rank == sendrank) {
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(locala, localn, MPI_DOUBLE, recvrank, mergetag, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
        CALI_MARK_BEGIN("MPI_Recv");
        MPI_Recv(locala, localn, MPI_DOUBLE, recvrank, sortedtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("MPI_Recv");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    } else {

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Recv");
        MPI_Recv(remote, localn, MPI_DOUBLE, sendrank, mergetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("MPI_Recv");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        

        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        merge(locala, localn, remote, localn, all);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        int theirstart = 0, mystart = localn;
        if (sendrank > rank) {
            theirstart = localn;
            mystart = 0;
        }
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(&(all[theirstart]), localn, MPI_DOUBLE, sendrank, sortedtag, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        for (int i=mystart; i<mystart+localn; i++)
            locala[i-mystart] = all[i];
    
    
    }
}



/*
    MPI_OddEven_Sort- entire odd-even sort of the array
        -communication and computation is performed in this function
        -called directly by main

*/
int MPI_OddEven_Sort(int n, double *a, int root, MPI_Comm comm)
{
    int rank, size, i;
    double *local_a;

// get rank and size of comm
    MPI_Comm_rank(comm, &rank); //&rank = address of rank
    MPI_Comm_size(comm, &size);

    local_a = (double *) calloc(n / size, sizeof(double));


// scatter the array a to local_a
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(a, n / size, MPI_DOUBLE, local_a, n / size, MPI_DOUBLE, root, comm);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
// sort local_a


    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    merge_sort(n / size, local_a);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
//odd-even part
    for (i = 1; i <= size; i++) {

        //printstat(rank, i, "before", local_a, n/size);



        if ((i + rank) % 2 == 0) {  // means i and rank have same nature
            if (rank < size - 1) {
                MPI_Pairwise_Exchange(n / size, local_a, rank, rank + 1, comm);
            }
        } else if (rank > 0) {
            MPI_Pairwise_Exchange(n / size, local_a, rank - 1, rank, comm);
        }



    }

    //printstat(rank, i-1, "after", local_a, n/size);

// gather local_a to a
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_a, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (rank == root){
        //printstat(rank, i, " all done ", a, n);
        CALI_MARK_BEGIN("check_correctness");
        bool cc = correctness_check(a, n);
        CALI_MARK_END("check_correctness");
        if(cc == true){
            printf("Array is sorted from least to greatest \n");

        }
        else{
            printf("Array is not sorted \n");
        }
    }

    return MPI_SUCCESS;
}

int main(int argc, char **argv) {

    CALI_MARK_BEGIN("main");


    MPI_Init(&argc, &argv);

    //CALI_MARK_BEGIN("main");
    //initatlizing and getting the command line arguments
    int array_size = atoi(argv[1]);
    double a[array_size];
    int input = atoi(argv[2]);
    int num_procsesses = atoi(argv[3]);
    int upperlimit = 250;


    CALI_MARK_BEGIN("whole_computation");

    CALI_MARK_BEGIN("data_init");

    //data initalization depending on the input value on the command line
    if(input == 0){
        for(int i = 0; i<array_size; i++){
            a[i] = i;
        }

    }
    else if(input == 1){
        srand(time(NULL));
        for (int i = 0; i < array_size; i++) {
            a[i] = rand()%upperlimit;
        }
    }
    else if(input == 2){
        for(int i = 0; i<array_size; i++){
            a[i] = array_size - i;
        }
    }
    else{
        std::srand(static_cast<unsigned>(std::time(0)));
        // Fill the array with perturbed values
        for (int i = 0; i < array_size; ++i) {
            // Generate a random number between 0 and 1
            double randomValue = static_cast<double>(rand()) / RAND_MAX;

            // Check if the current element should be perturbed
            if (randomValue < 0.01) { // 1% perturbation
                // Perturb the value by adding a small random integer value
                a[i] = static_cast<double>(rand() % 10); // You can adjust the perturbation range
            } else {
                // Assign a regular value
                a[i] = i + 1; // You can replace this with any desired value assignment
            }
        }
    }
    CALI_MARK_END("data_init");

    //calling the actual odd_even sort on the array a
    MPI_OddEven_Sort(array_size, a, 0, MPI_COMM_WORLD);

    CALI_MARK_END("whole_computation");

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



    //adiak data
   
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Odd Even Transposition Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procsesses); // The number of processors (MPI ranks)
    //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


    MPI_Finalize();

    

    //CALI_MARK_END("main");
    
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_BEGIN("comp_small");
    CALI_MARK_END("comp_small");




    CALI_MARK_END("main");

    return 0;
}