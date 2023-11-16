#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/**** Function Declaration Section ****/
static int intcompare(const void *i, const void *j) {
  if ((*(int *)i) > (*(int *)j))
    return (1);
  if ((*(int *)i) < (*(int *)j))
    return (-1);
  return (0);
}

void data_init(int array_size, int **Input, int MyRank) {
  int i;

  CALI_MARK_BEGIN("data_init");

  *Input = (int *)malloc(array_size * sizeof(int));

  if (*Input == NULL) {
    printf("Error: Can not allocate memory\n");
    MPI_Finalize();
    exit(0);
  }

  srand48((unsigned int)array_size);
  printf("Input Array for Sorting\n\n");
  for (i = 0; i < array_size; i++) {
    (*Input)[i] = rand();
    printf("%d   ", (*Input)[i]);
  }
  printf("\n\n");

  CALI_MARK_END("data_init");
}

void correctness_check(int array_size, int *Output) {
  CALI_MARK_BEGIN("correctness_check");

  for (int i = 1; i < array_size; i++) {
    if (Output[i] < Output[i - 1]) {
      printf("Error: Output is not sorted correctly.\n");
      MPI_Finalize();
      exit(0);
    }
  }

  CALI_MARK_END("correctness_check");
}

int main(int argc, char *argv[]) {
  /* Variable Declarations */
  int Numprocs, MyRank, Root = 0;
  int NoofElements, NoofElements_Bloc, NoElementsToSort;
  int *Input = NULL, *InputData, *Splitter, *AllSplitter, *Buckets, *BucketBuffer, *LocalBucket, *OutputBuffer, *Output;
  FILE *InputFile, *fp;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

  CALI_MARK_BEGIN("main");

  double start_time, end_time;

  /**** Initialising ****/
  if (MyRank == Root) {
    if (argc != 4) {
      if (MyRank == 0)
        printf(" Usage : %s processes array_size input_type num_processes\n", argv[0]);
      MPI_Finalize();
      exit(0);
    }

    NoofElements =  atoi(argv[1]);
    int input_type =  atoi(argv[2]);
    int num_processes = atoi(argv[3]);

 std::string input_string = ""; 
   
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
   adiak::init(NULL);
   adiak::launchdate();    // launch date of the job
   adiak::libraries();     // Libraries used
   adiak::cmdline();       // Command line used to launch the job
   adiak::clustername();   // Name of the cluster
   adiak::value("Algorithm", "Odd Even Transposition Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
   adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
   adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
   adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
   adiak::value("InputSize", NoofElements); // The number of elements in input dataset (1000)
   adiak::value("InputType", input_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
   adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
   adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
   adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Now you can use these variables as needed
    data_init(NoofElements, &Input, MyRank);
  }

  /**** Data Initialization Timing ****/
  if (MyRank == Root) {
    start_time = MPI_Wtime();
  }

  CALI_MARK_BEGIN("comm");

  // Broadcasting array_size
  MPI_Bcast(&NoofElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if ((NoofElements % Numprocs) != 0) {
    if (MyRank == Root) {
      printf("Number of Elements are not divisible by Numprocs\n");
    }
    MPI_Finalize();
    exit(0);
  }

  NoofElements_Bloc = NoofElements / Numprocs;
  InputData = (int *)malloc(NoofElements_Bloc * sizeof(int));

  if (MyRank == Root) {
    // Copy the data to InputData for the root process
    memcpy(InputData, Input, NoofElements_Bloc * sizeof(int));
  }

  MPI_Scatter(Input, NoofElements_Bloc, MPI_INT, InputData, NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

  CALI_MARK_BEGIN("comm_small");

  /**** Sorting Locally ****/
  qsort((char *)InputData, NoofElements_Bloc, sizeof(int), intcompare);

  CALI_MARK_END("comm_small");

  /**** Choosing Local Splitters ****/
  Splitter = (int *)malloc(sizeof(int) * (Numprocs - 1));
  for (int i = 0; i < (Numprocs - 1); i++) {
    Splitter[i] = InputData[NoofElements / (Numprocs * Numprocs) * (i + 1)];
  }

  CALI_MARK_BEGIN("comm_large");

  /**** Gathering Local Splitters at Root ****/
  AllSplitter = (int *)malloc(sizeof(int) * Numprocs * (Numprocs - 1));
  MPI_Gather(Splitter, Numprocs - 1, MPI_INT, AllSplitter, Numprocs - 1, MPI_INT, Root, MPI_COMM_WORLD);

  /**** Choosing Global Splitters ****/
  if (MyRank == Root) {
    qsort((char *)AllSplitter, Numprocs * (Numprocs - 1), sizeof(int), intcompare);
    for (int i = 0; i < Numprocs - 1; i++) {
      Splitter[i] = AllSplitter[(Numprocs - 1) * (i + 1)];
    }
  }

  /**** Broadcasting Global Splitters ****/
  MPI_Bcast(Splitter, Numprocs - 1, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm_large");

  /**** Creating Numprocs Buckets locally ****/
  Buckets = (int *)malloc(sizeof(int) * (NoofElements + Numprocs));

  int j = 0;
  int k = 1;

  for (int i = 0; i < NoofElements_Bloc; i++) {
    if (j < (Numprocs - 1)) {
      if (InputData[i] < Splitter[j])
        Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
      else {
        Buckets[(NoofElements_Bloc + 1) * j] = k - 1;
        k = 1;
        j++;
        i--;
      }
    } else
      Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
  }
  Buckets[(NoofElements_Bloc + 1) * j] = k - 1;



  /**** Sending buckets to respective processors ****/
  BucketBuffer = (int *)malloc(sizeof(int) * (NoofElements + Numprocs));

  MPI_Alltoall(Buckets, NoofElements_Bloc + 1, MPI_INT, BucketBuffer, NoofElements_Bloc + 1, MPI_INT, MPI_COMM_WORLD);

  CALI_MARK_END("comm");

  CALI_MARK_BEGIN("comp");

  /**** Rearranging BucketBuffer ****/
  CALI_MARK_BEGIN("comp_large");
  LocalBucket = (int *)malloc(sizeof(int) * 2 * NoofElements / Numprocs);

  int count = 1;

  for (int j = 0; j < Numprocs; j++) {
    int k = 1;
    for (int i = 0; i < BucketBuffer[(NoofElements / Numprocs + 1) * j]; i++)
      LocalBucket[count++] = BucketBuffer[(NoofElements / Numprocs + 1) * j + k++];
  }
  LocalBucket[0] = count - 1;

  CALI_MARK_BEGIN("comp_small");

  /**** Sorting Local Buckets using Bubble Sort ****/
  NoElementsToSort = LocalBucket[0];
  qsort((char *)&LocalBucket[1], NoElementsToSort, sizeof(int), intcompare);
  
  CALI_MARK_END("comp_small");

  CALI_MARK_END("comp_large");

  CALI_MARK_END("comp");

  CALI_MARK_BEGIN("comm");

  /**** Gathering sorted sub-blocks at root ****/
  if (MyRank == Root) {
    OutputBuffer = (int *)malloc(sizeof(int) * 2 * NoofElements);
    Output = (int *)malloc(sizeof(int) * NoofElements);
  }

  MPI_Gather(LocalBucket, 2 * NoofElements_Bloc, MPI_INT, OutputBuffer, 2 * NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

  CALI_MARK_END("comm");

  CALI_MARK_BEGIN("comp");

  /**** Rearranging output buffer and printing it to standard output ****/
  if (MyRank == Root) {
       
    count = 0;
    for (int j = 0; j < Numprocs; j++) {
      int k = 1;
      for (int i = 0; i < OutputBuffer[(2 * NoofElements / Numprocs) * j]; i++)
        Output[count++] = OutputBuffer[(2 * NoofElements / Numprocs) * j + k++];
    }

    printf("Number of Elements to be sorted: %d\n", NoofElements);
    printf("The sorted sequence is:\n");
    for (int i = 0; i < NoofElements; i++) {
      printf("%d   ", Output[i]);
    }
    printf("\n");
    
  }
  
   

  CALI_MARK_END("comp");

   free(InputData);
  free(Splitter);
  free(AllSplitter);
  free(Buckets);
  free(BucketBuffer);
  free(LocalBucket);

  if (MyRank == Root) {
    free(Input);
    free(OutputBuffer);
    free(Output);

    end_time = MPI_Wtime();
    printf("Total computation time: %f seconds\n", end_time - start_time);
  }

  /**** Finalize ****/
  MPI_Finalize();
  CALI_MARK_END("main");
  
   
  
  
}