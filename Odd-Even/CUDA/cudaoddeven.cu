#include<stdio.h>
#include<cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>    

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


/*

    This code was written by Kshitij421
    https://github.com/Kshitij421/Odd-Even-Sort-using-Cuda-/blob/master/oddeven.cu

    It has been modified to meet the needs of our project


*/

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
//const char* main = "main";



int threads;


/*
oddeven completes one step of the odd or even phase
    - Depending on the phases that determines who each block is partnering with
    - in each if statement swapping the values if needed
*/

__global__ void oddeven(int* x,int I,int n)
{
	int id=blockIdx.x;
	if(I==0 && ((id*2+1)< n)){ //even phase
		if(x[id*2]>x[id*2+1]){
			int X=x[id*2];
			x[id*2]=x[id*2+1];
			x[id*2+1]=X;
		}
	}
	if(I==1 && ((id*2+2)< n)){ //odd phase
		if(x[id*2+1]>x[id*2+2]){
			int X=x[id*2+1];
			x[id*2+1]=x[id*2+2];
			x[id*2+2]=X;
		}
	}
}

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


int main(int argc, char *argv[])
{

    //a is unsorted array
    //c is sorted array
    //n is number of elements in array
    //d is device array

    CALI_MARK_BEGIN("main");

    //Seetting up variables
    int n = atoi(argv[1]);
    threads = atoi(argv[2]);
    int input = atoi(argv[3]); //0 is sorted, 1 is random, 2 is reverse sorted, 3 is 1%pertubed 
    int c[n];
    int a[n]; 

    const int upperlimit = 250; //setting upper limit on random values

    CALI_MARK_BEGIN("whole_computation");

    CALI_MARK_BEGIN("data_init");


    //Determining what type of input is happening
    if(input == 0){
        for(int i = 0; i<n; i++){
            a[i] = i;
        }

    }
    else if(input == 1){
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            a[i] = rand()%upperlimit;
        }
    }
    else if(input == 2){
        for(int i = 0; i<n; i++){
            a[i] = n - i;
        }
    }
    else{
        std::srand(static_cast<unsigned>(std::time(0)));
        // Fill the array with perturbed values
        for (int i = 0; i < n; ++i) {
            // Generate a random number between 0 and 1
            double randomValue = static_cast<double>(rand()) / RAND_MAX;

            // Check if the current element should be perturbed
            if (randomValue < 0.01) { // 1% perturbation
                // Perturb the value by adding a small random integer value
                a[i] = static_cast<int>(rand() % 10); // You can adjust the perturbation range
            } else {
                // Assign a regular value
                a[i] = i + 1; // You can replace this with any desired value assignment
            }
        }
    }

    CALI_MARK_END("data_init");




    /*
    printf("Orignial Array is:\t");
    for(int i = 0; i<n; i++){
        printf("%d \t",a[i]);
    }
    printf("\n");
    */
    

    //Initalizing device array and communication
    int *d; 
	cudaMalloc((void**)&d, n*sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
	cudaMemcpy(d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    
    //Performing the odd even sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
	for(int i=0;i<n;i++){
		oddeven<<<n/2, threads>>>(d,i%2,n);
	}
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    
	//printf("\n");

    //Copying the sorted device array to a new array c
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
	cudaMemcpy(c,d,n*sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    //Checking if c is sorted correctly
    CALI_MARK_BEGIN("check_correctness");
    bool cc = correctness_check(c,n);
    CALI_MARK_END("check_correctness");

    CALI_MARK_END("whole_computation");


    //Printing sorted array and confirming correctness check
    printf("Sorted Array is:\t");
	for(int i=0; i<n; i++)
	{
		//printf("%d\t",c[i]);
	}
    printf("\n");

    if( cc == true){
        printf("The array is sorted from least to greatest \n");
    }
    else{
        printf("The array is not sorted \n");
    }

	cudaFree(d);

    //CALI_MARK_END("main");

    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_BEGIN("comp_small");
    CALI_MARK_END("comp_small");


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
        input_string = "1%%pertubed";
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Odd Even Transposition Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", n/2); // The number of CUDA blocks 
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").



    CALI_MARK_END("main");


	return 0;
}