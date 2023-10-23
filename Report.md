# CSCE 435 Group project

## 1. Group members:
1. First
2. Second
3. Third
4. Fourth

---

## 2. _due 10/25_ Project topic

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core)

## 3. _due 11/08_ Pseudocode for each algorithm and implementation

## 3. _due 11/08_ Evaluation plan - what and how will you measure and compare

For example:
- Effective use of a GPU (play with problem size and number of threads)
- Strong scaling to more nodes (same problem size, increase number of processors)
- Weak scaling (increase problem size, increase number of processors)

## 4. _due 11/15_ Performance evaluation

Include detailed analysis of computation performance, communication performance. Include figures and explanation of your analysis.

### A. Input Data
Test your algorithms on arrays of structure:
- Sorted
- Random
- Reverse sorted
- 1% perturbed

For data sizes:
- 2^16, 2^20, 2^24, 2^28

### B. Caliper

**Caliper output is required**. Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` (same as lab1 build.sh) to collect caliper files per experiment.

#### I. Caliper Region Structure
Your caliper region structure should look like this:
```
main
|_ comm
|    |_ MPI_Bcast_splitters
|    |_ MPI_Barrier
|    |_ MPI_Send_splitters
|_ comp
     |_ sort_local
     |_ sort_splitters
```
All functions will be called from `main` and grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

To nest calls you must simply do the following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("sort_local");
sort_local();
CALI_MARK_END("sort_local");
CALI_MARK_END("comp");
```
The structure can be validated using `Thicket.tree()`.

#### II. Required Performance Metrics
- Time
    - Min
    - Max
    - Average
    - Total
    - Variance

You can use `CALI_CONFIG="spot" ./my_program` in your jobfile to automatically collect these. They will show up in the `Thicket.dataframe` if the caliper file is read into Thicket.

#### III. Required Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("algorithm", algorithm); // Where algorithm is the type of algorithm used ("Merge sort", "Bitonic sort")
adiak::value("hardware", hardware); // Where hardware is "CPU" or "GPU"
adiak::value("input_data_length", input_data_length); // Where input_data_length is the length of elements in input dataset (1000)
adiak::value("input_data_element_datatype", input_data_element_datatype); Where input_data_element_datatype is the datatype of individual elements (double, int, float)
adiak::value("input_data_element_datasize", input_data_element_datasize); Where input_data_element_datasize is the datasize in bytes of individual elements (1, 2, 4)
adiak::value("input_data_structure", input_data_structure); // Where input_data_structure is the structure of the input data ("Sorted", "Reverse", "Random", "1% perturbed")
adiak::value("num_procs", num_procs); // Where num_procs is the number of processors (if this is a CPU workload).
adiak::value("num_threads", num_threads); // Where num_threads is the number of threads on the CPU/GPU.
adiak::value("num_blocks", num_blocks); // Where num_blocks is the number of blocks (if this is a GPU workload).
adiak::value("team", team); // Where team is your team ("Team 1", "Team 10")
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

### C. Hints for performance analysis

Parameterize your program!!  Integers are easiest so enumerate your parameters:
- Number of MPI ranks is something you can query inside an MPI program (so that’s a parameter to mpirun that you can know inside the program) – or just pass it in. Similarly, pass in the number of threads to use for your GPU kernels.
- If you are sorting, you can decide that your algorithms are now Sort0, Sort1, Sort2 – and pass in the number as the parameter. If you are running all sorting algorithms separately because different people are responsible for them, that’s fine too – you’ll just have to run 3x the jobs.
- If you are sorting, you can decide that 0 represents sorted input, 1 represents reverse sorted input, 2 represents random – and pass that inputType in as an integer parameter.

You are running parallel jobs; you should only print output from process 0 for MPI jobs, and only from the CPU for CUDA jobs.
You will save yourself a lot of headache if you only print the output you need.  A well formatted line (comma separated or tab separated is best) so you can import later.
Pick a format and stick to it – reverse engineer from what you would like in your spreadsheet or Pandas DataFrame:
- algoName, processCount, inputType, total_time, comp_time, comm_time

## 5. _due 11/29-12/08_ Presentation

## 6. _due 11/30-12/09_ Final Report