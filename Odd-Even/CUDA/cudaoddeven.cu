#include<stdio.h>
#include<cuda.h>

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
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_large = "comp_large";



int threads;


__global__ void oddeven(int* x,int I,int n)
{
	int id=blockIdx.x;
	if(I==0 && ((id*2+1)< n)){
		if(x[id*2]>x[id*2+1]){
			int X=x[id*2];
			x[id*2]=x[id*2+1];
			x[id*2+1]=X;
		}
	}
	if(I==1 && ((id*2+2)< n)){
		if(x[id*2+1]>x[id*2+2]){
			int X=x[id*2+1];
			x[id*2+1]=x[id*2+2];
			x[id*2+2]=X;
		}
	}
}


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


    int n = atoi(argv[1]);
    threads = atoi(argv[2]);
    int c[n];


    int a[n]; 

    const int upperlimit = 250;

    CALI_MARK_BEGIN("whole_computation");

    CALI_MARK_BEGIN("data_init");
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i] = rand()%upperlimit;
    }
    CALI_MARK_END("data_init");

    /*
    printf("Orignial Array is:\t");
    for(int i = 0; i<n; i++){
        printf("%d \t",a[i]);
    }
    printf("\n");
    */
    

    int *d; 


	cudaMalloc((void**)&d, n*sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
	cudaMemcpy(d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
	for(int i=0;i<n;i++){
		oddeven<<<n/2, threads>>>(d,i%2,n);
	}
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    
	//printf("\n");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
	cudaMemcpy(c,d,n*sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    bool cc = correctness_check(c,n);

    CALI_MARK_END("whole_computation");


    printf("Sorted Array is:\t");
	for(int i=0; i<n; i++)
	{
		printf("%d\t",c[i]);
	}
    printf("\n");

    if( cc == true){
        printf("The array is sorted from least to greatest \n");
    }
    else{
        printf("The array is not sorted \n");
    }

	cudaFree(d);


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
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", n/2); // The number of CUDA blocks 
    adiak::value("group_num", 9); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").






	return 0;
}