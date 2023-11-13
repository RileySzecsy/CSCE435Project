/*
 * File:     parallel_odd_even.c
 * Purpose:  Implement parallel odd-even sort of an array of 
 *           nonegative ints
 * Input:
 *    A:     elements of array (optional)
 * Output:
 *    A:     elements of A after sorting
 *
 * Compile:  mpicc -g -Wall -o parallel_odd_even parallel_odd_even.c
 * Run:
 *    mpiexec -n <p> parallel_odd_even <g|i> <global_n> 
 *       - p: the number of processes
 *       - g: generate random, distributed list
 *       - i: user will input list on process 0
 *       - global_n: number of elements in global list
 *
 * Notes:
 * 1.  global_n must be evenly divisible by p
 * 2.  DEBUG flag prints original and final sublists
 */



/*
    This algorithm was written by umbc 
    https://redirect.cs.umbc.edu/~tsimo1/CMSC483/cs220/code/sorting/parallel_odd_even.c

    It has been modifed to meet the needs of the final project

*/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// const int RMAX = 1000000000;
const int RMAX = 100;


//CALI Regions
const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* check_correctness = "check_correctness";


/* Local functions */
void Usage(char* program);
void Print_list(int local_A[], int local_n, int rank);
void Merge_split_low(int local_A[], int temp_B[], int temp_C[], int local_n); //comp_small
void Merge_split_high(int local_A[], int temp_B[], int temp_C[], int local_n); //comp_small
void Generate_list(int local_A[], int local_n, int my_rank, int input_type); //data_init
int  Compare(const void* a_p, const void* b_p); //Used as a comparitor in Sort

/* Functions involving communication */
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p, char* gi_p, int my_rank, int p, MPI_Comm comm_mpi); //N/A
void Sort(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm_mpi); //
void Odd_even_iter(int local_A[], int temp_B[], int temp_C[], int local_n, int phase, int even_partner, int odd_partner, int my_rank, int p, MPI_Comm comm_mpi);
void Print_local_lists(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm_mpi); //N/A
void Print_global_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm_mpi); //N/A
void Read_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm_mpi); //comm_large


/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {

   //Initalizing variables
   int my_rank, p;
   char g_i;
   int *local_A;
   int global_n;
   int local_n;
   MPI_Comm comm_mpi;
   double start, finish;

   //Determining input types
   int input_type = std::atoi(argv[3]);
   //printf("input type: %d", input_type); 

   //Setting up MPI
   MPI_Init(&argc, &argv);
   comm_mpi = MPI_COMM_WORLD;
   MPI_Comm_size(comm_mpi, &p);
   MPI_Comm_rank(comm_mpi, &my_rank);

   //Getting the arguments to determine to fill the variables initalized above
   Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, p, comm_mpi);
   local_A = (int*) malloc(local_n*sizeof(int));
   if (g_i == 'g') {
      CALI_MARK_BEGIN("data_init");
      Generate_list(local_A, local_n, my_rank, input_type); //Generating input depending on what has been passesd in
      CALI_MARK_END("data_init");
   } else { //this will never happen
      Read_list(local_A, local_n, my_rank, p, comm_mpi);
   }
//#  ifdef DEBUG
   Print_local_lists(local_A, local_n, my_rank, p, comm_mpi);
//#  endif

   start = MPI_Wtime();
   //CALI_MARK_BEGIN("whole_computation");
   Sort(local_A, local_n, my_rank, p, comm_mpi); //Performing the sorting algorithm
   //CALI_MARK_END("whole_computation");
   finish = MPI_Wtime();
   if (my_rank == 0)
      printf("Elapsed time = %e seconds\n", finish-start);

#  ifdef DEBUG
   Print_local_lists(local_A, local_n, my_rank, p, comm_mpi);
   fflush(stdout);
#  endif

   Print_global_list(local_A, local_n, my_rank, p, comm_mpi); //correctness check is performed before printing the lists inside here

   free(local_A);

   MPI_Finalize();
   


   //Used for the adiak data
   std::string input_string; 
   if(input_type == 0){
      input_string = "Sorted";
   }
   else if(input_type == 1){
      input_string = "Random";
   }
   else if(input_type == 2){
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
   adiak::value("InputSize", global_n); // The number of elements in input dataset (1000)
   adiak::value("InputType", input_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
   adiak::value("num_procs", p); // The number of processors (MPI ranks)
   adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
   adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").



   return 0;
}  /* main */


/*
correctness_check determines if the array is sorted in least to greatest order
    - returns false if any value in the array is out of place
    - returns true if it has reached the end
*/
bool correctness_check(int A[], int n){
    for(int i = 0; i < n - 1; i++){
        if (A[i] > A[i+1]){
            return false;
        }
    }
    return true;
}





/*-------------------------------------------------------------------
 * Function:   Generate_list
 * Purpose:    Fill list with random ints
 * Input Args: local_n, my_rank
 * Output Arg: local_A
 */
void Generate_list(int local_A[], int local_n, int my_rank, int input_type) {
   int i;
   if(input_type == 0){
      for(i = 0; i<local_n; i++){
         local_A[i] = i;
      }
   }
   else if(input_type == 1){
      srandom(my_rank+1);
      for (i = 0; i < local_n; i++){
         local_A[i] = random() % RMAX;
      }
   }
   else if(input_type == 2){
      for(i = 0; i<local_n; i++){
         local_A[i] = local_n - i;
      }
   }
   else{
      std::srand(static_cast<unsigned>(std::time(0)));
      // Fill the array with perturbed values
      for (i = 0; i < local_n; ++i) {
         // Generate a random number between 0 and 1
         double randomValue = static_cast<double>(rand()) / RAND_MAX;

         // Check if the current element should be perturbed
         if (randomValue < 0.01) { // 1% perturbation
               // Perturb the value by adding a small random integer value
               local_A[i] = static_cast<int>(rand() % 10); // You can adjust the perturbation range
         } else {
               // Assign a regular value
               local_A[i] = i + 1; // You can replace this with any desired value assignment
         }
      }

   }


}  /* Generate_list */


/*-------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print command line to start program
 * In arg:    program:  name of executable
 * Note:      Purely local, run only by process 0;
 */
void Usage(char* program) {
   fprintf(stderr, "usage:  mpirun -np <p> %s <g|i> <global_n>\n",
       program);
   fprintf(stderr, "   - p: the number of processes \n");
   fprintf(stderr, "   - g: generate random, distributed list\n");
   fprintf(stderr, "   - i: user will input list on process 0\n");
   fprintf(stderr, "   - global_n: number of elements in global list");
   fprintf(stderr, " (must be evenly divisible by p)\n");
   fflush(stderr);
}  /* Usage */


/*-------------------------------------------------------------------
 * Function:    Get_args
 * Purpose:     Get and check command line arguments
 * Input args:  argc, argv, my_rank, p, comm
 * Output args: global_n_p, local_n_p, gi_p
 */
void Get_args(int argc, char* argv[], int* global_n_p, int* local_n_p, 
         char* gi_p, int my_rank, int p, MPI_Comm comm_mpi) {

   if (my_rank == 0) {
      if (argc != 4) {
         printf("stopped at 1 \n");
         Usage(argv[0]);
         *global_n_p = -1;  /* Bad args, quit */
      } else {
         *gi_p = argv[1][0];
         if (*gi_p != 'g' && *gi_p != 'i') {
             printf("stopped at 2 \n");
            Usage(argv[0]);
            *global_n_p = -1;  /* Bad args, quit */
         } else {
            *global_n_p = strtol(argv[2], NULL, 10);
            if (*global_n_p % p != 0) {
                printf("stopped at 3 \n");
               Usage(argv[0]);
               *global_n_p = -1;
            }
         }
      }
   }  /* my_rank == 0 */

   MPI_Bcast(gi_p, 1, MPI_CHAR, 0, comm_mpi);
   MPI_Bcast(global_n_p, 1, MPI_INT, 0, comm_mpi);

   if (*global_n_p <= 0) {
      MPI_Finalize();
      exit(-1);
   }

   *local_n_p = *global_n_p/p;

}  /* Get_args */


/*-------------------------------------------------------------------
 * Function:   Read_list
 * Purpose:    process 0 reads the list from stdin and scatters it
 *             to the other processes.
 * In args:    local_n, my_rank, p, comm
 * Out arg:    local_A
 */
void Read_list(int local_A[], int local_n, int my_rank, int p,
         MPI_Comm comm_mpi) {
   int i;
   int *temp = NULL;

   if (my_rank == 0) {
      temp = (int*) malloc(p*local_n*sizeof(int));
      printf("Enter the elements of the list\n");
      for (i = 0; i < p*local_n; i++)
         scanf("%d", &temp[i]);
   } 

   MPI_Scatter(temp, local_n, MPI_INT, local_A, local_n, MPI_INT,
       0, comm_mpi);

   if (my_rank == 0)
      free(temp);
}  /* Read_list */


/*-------------------------------------------------------------------
 * Function:   Print_global_list
 * Purpose:    Print the contents of the global list A
 * Input args:  
 *    n, the number of elements 
 *    A, the list
 * Note:       Purely local, called only by process 0
 */
void Print_global_list(int local_A[], int local_n, int my_rank, int p, 
      MPI_Comm comm_mpi) {
   int* A = NULL;
   int i, n;

   if (my_rank == 0) {
      n = p*local_n;
      A = (int*) malloc(n*sizeof(int));
      MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0,
            comm_mpi);
      printf("Global list:\n");
      for (i = 0; i < n; i++)
         printf("%d ", A[i]);
      printf("\n\n");


      CALI_MARK_BEGIN("check_correctness");
      bool cc = correctness_check(A,n);
      CALI_MARK_END("check_correctness");
      if(cc == true){
            printf("The array is sorted from least to greatest \n");
        }
        else{
            printf("The array is not sorted \n");
        }
      free(A);
   } else {
      MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0,
            comm_mpi);
   }

}  /* Print_global_list */

/*-------------------------------------------------------------------
 * Function:    Compare
 * Purpose:     Compare 2 ints, return -1, 0, or 1, respectively, when
 *              the first int is less than, equal, or greater than
 *              the second.  Used by qsort.
 */
int Compare(const void* a_p, const void* b_p) {
   int a = *((int*)a_p);
   int b = *((int*)b_p);

   if (a < b)
      return -1;
   else if (a == b)
      return 0;
   else /* a > b */
      return 1;
}  /* Compare */

/*-------------------------------------------------------------------
 * Function:    Sort
 * Purpose:     Use odd-even sort to sort global list.
 * Input args:  local_n, my_rank, p, comm
 * In/out args: local_A 
 */
void Sort(int local_A[], int local_n, int my_rank, 
         int p, MPI_Comm comm_mpi) {
   int phase;
   int *temp_B, *temp_C;
   int even_partner;  /* phase is even or left-looking */
   int odd_partner;   /* phase is odd or right-looking */

   /* Temporary storage used in merge-split */
   temp_B = (int*) malloc(local_n*sizeof(int));
   temp_C = (int*) malloc(local_n*sizeof(int));

   /* Find partners:  negative rank => do nothing during phase */
   if (my_rank % 2 != 0) {
      even_partner = my_rank - 1;
      odd_partner = my_rank + 1;
      if (odd_partner == p) odd_partner = -1;  // Idle during odd phase
   } else {
      even_partner = my_rank + 1;
      if (even_partner == p) even_partner = -1;  // Idle during even phase
      odd_partner = my_rank-1;  
   }

   /* Sort local list using built-in quick sort */
   qsort(local_A, local_n, sizeof(int), Compare);

   for (phase = 0; phase < p; phase++)
      Odd_even_iter(local_A, temp_B, temp_C, local_n, phase, 
             even_partner, odd_partner, my_rank, p, comm_mpi);

   free(temp_B);
   free(temp_C);
}  /* Sort */


/*-------------------------------------------------------------------
 * Function:    Odd_even_iter
 * Purpose:     One iteration of Odd-even transposition sort
 * In args:     local_n, phase, my_rank, p, comm
 * In/out args: local_A
 * Scratch:     temp_B, temp_C
 */
void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
        int local_n, int phase, int even_partner, int odd_partner,
        int my_rank, int p, MPI_Comm comm_mpi) {
   MPI_Status status;

   if (phase % 2 == 0) {  /* Even phase, odd process <-> rank-1 */
      if (even_partner >= 0) {
        
         CALI_MARK_BEGIN("comm");
         CALI_MARK_BEGIN("comm_small");
         MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0, 
            temp_B, local_n, MPI_INT, even_partner, 0, comm_mpi,
            &status);
         CALI_MARK_END("comm_small");
         CALI_MARK_END("comm");
        
         CALI_MARK_BEGIN("comp");
         CALI_MARK_BEGIN("comp_large");
         if (my_rank % 2 != 0){

            Merge_split_high(local_A, temp_B, temp_C, local_n);
         }
         else{
            Merge_split_low(local_A, temp_B, temp_C, local_n);
         }
         CALI_MARK_END("comp_large");
         CALI_MARK_END("comp");
      }
   } else { /* Odd phase, odd process <-> rank+1 */
      if (odd_partner >= 0) {
         CALI_MARK_BEGIN("comm");
         CALI_MARK_BEGIN("comm_small");
         MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0, 
            temp_B, local_n, MPI_INT, odd_partner, 0, comm_mpi,
            &status);
         CALI_MARK_END("comm_small");
         CALI_MARK_END("comm");

         CALI_MARK_BEGIN("comp");
         CALI_MARK_BEGIN("comp_large");
         if (my_rank % 2 != 0){
            Merge_split_low(local_A, temp_B, temp_C, local_n);
        }
         else{
            Merge_split_high(local_A, temp_B, temp_C, local_n);
         }
         CALI_MARK_END("comp_large");
         CALI_MARK_END("comp");
      }
   }
}  /* Odd_even_iter */


/*-------------------------------------------------------------------
 * Function:    Merge_split_low
 * Purpose:     Merge the smallest local_n elements in local_A 
 *              and temp_B into temp_C.  Then copy temp_C
 *              back into local_A.
 * In args:     local_n, temp_B
 * In/out args: local_A
 * Scratch:     temp_C
 */
void Merge_split_low(int local_A[], int temp_B[], int temp_C[], 
        int local_n) {
   int ai, bi, ci;
   
   ai = 0;
   bi = 0;
   ci = 0;
   while (ci < local_n) {
      if (local_A[ai] <= temp_B[bi]) {
         temp_C[ci] = local_A[ai];
         ci++; ai++;
      } else {
         temp_C[ci] = temp_B[bi];
         ci++; bi++;
      }
   }

   memcpy(local_A, temp_C, local_n*sizeof(int));
}  /* Merge_split_low */

/*-------------------------------------------------------------------
 * Function:    Merge_split_high
 * Purpose:     Merge the largest local_n elements in local_A 
 *              and temp_B into temp_C.  Then copy temp_C
 *              back into local_A.
 * In args:     local_n, temp_B
 * In/out args: local_A
 * Scratch:     temp_C
 */
void Merge_split_high(int local_A[], int temp_B[], int temp_C[], 
        int local_n) {
   int ai, bi, ci;
   
   ai = local_n-1;
   bi = local_n-1;
   ci = local_n-1;
   while (ci >= 0) {
      if (local_A[ai] >= temp_B[bi]) {
         temp_C[ci] = local_A[ai];
         ci--; ai--;
      } else {
         temp_C[ci] = temp_B[bi];
         ci--; bi--;
      }
   }

   memcpy(local_A, temp_C, local_n*sizeof(int));
}  /* Merge_split_low */


/*-------------------------------------------------------------------
 * Only called by process 0
 */
void Print_list(int local_A[], int local_n, int rank) {
   int i;
   printf("%d: ", rank);
   for (i = 0; i < local_n; i++)
      printf("%d ", local_A[i]);
   printf("\n");
}  /* Print_list */

/*-------------------------------------------------------------------
 * Function:   Print_local_lists
 * Purpose:    Print each process' current list contents
 * Input args: all
 * Notes:
 * 1.  Assumes all participating processes are contributing local_n 
 *     elements
 */
void Print_local_lists(int local_A[], int local_n, 
         int my_rank, int p, MPI_Comm comm_mpi) {
   int*       A;
   int        q;
   MPI_Status status;

   if (my_rank == 0) {
      A = (int*) malloc(local_n*sizeof(int));
      Print_list(local_A, local_n, my_rank);
      for (q = 1; q < p; q++) {
         MPI_Recv(A, local_n, MPI_INT, q, 0, comm_mpi, &status);
         Print_list(A, local_n, q);
      }
      free(A);
   } else {
      MPI_Send(local_A, local_n, MPI_INT, 0, 0, comm_mpi);
   }
}  /* Print_local_lists */