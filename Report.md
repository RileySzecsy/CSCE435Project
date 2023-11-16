# CSCE 435 Group project

## 1. Group members:
1. Riley Szecsy
2. Vincent Lobello
3. Jonathan Kutsch
4. Nebiyou Ersabo <br>
Group is communicating over groupme. 
---

## 2. Project topic
Comparing the performance of comparision based sorting algorithms. We will be scaling the GPU settings and change the number of threads and proccessors of the CPU based on problem size. We will be testing these on sorted, random, and reverse sorted inputs. 
For Radix sort we will be comparing it directly to sample sort as we can only test integers. 

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

- Sample Sort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode: <br>
  ```
  Where p = # processors and k = oversampling factor:
  1. Sample p (* k) elements and sort them
  2. Share these samples with every processor
  (MPI_Allgather)
  3. Each p select p-1 pivots aka splitters. These are
  the same across p’s.
  a. Each b-th pair of splitters denotes a “bucket” that
  will be sent to the b-th processor.
  4. Re-arrange local data into the buckets described
  by the pivots.
  5. Send the b-th bucket to the b-th processor
  (MPI_Alltoall[v])
  6. Combine buckets and sort local data.
  ```

- Mergesort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode:
  
  ```
  function parallel_merge_sort(arr):
    if length(arr) <= 1:
        return arr
    
    middle = length(arr) / 2
    left_half = arr[0:middle]
    right_half = arr[middle:]

    left_sorted = parallel_merge_sort(left_half)
    right_sorted = parallel_merge_sort(right_half)

    result = parallel_merge(left_sorted, right_sorted)

    return result

  function parallel_merge(left, right):
    result = []
    left_index = 0
    right_index = 0
    while left_index < length(left) and right_index < length(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index++
        else:
            result.append(right[right_index])
            right_index++
    while left_index < length(left):
        result.append(left[left_index])
        left_index++
    while right_index < length(right):
        result.append(right[right_index])
        right_index++
    return result
  
  ```


- Odd-Even Transposition Sort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode: <br>
  ```
    procedure ODD-EVEN PAR(n)
    begin
      id := proccees's label
      for i := 1 to n do
      begin
        if i is odd then
          if id is odd then
           compare-exchange min (id+1);
          else
            compare-exchange max(id-1);
          if i is even then
            if id is even then
              compare-exchange min(id+1);
            else
              compare-exhange max(id-1);
        end for
     end ODD-EVEN PAR
```
```
- Radix Sort (MPI + CUDA)
  - MPI on each core
  - Pesudocode <br>
    
  ```
  procedure RadixSort(arr, n)
  begin
      id := process's label
      for exp := 1 to maximum_digit_position do
      begin
          if exp is odd then
              if id is odd then
                  CompareExchangeMin(id + 1)
              else
                  CompareExchangeMax(id - 1)
          end if
          if exp is even then
              if id is even then
                  CompareExchangeMin(id + 1)
              else
                  CompareExchangeMax(id - 1)
          end if
          Call CountSort(arr, n, exp)
      end for
  end RadixSort
  
  procedure CountSort(arr, n, exp)
  begin
      Create an output array of size n
      Create a count array of size 10 and initialize to 0
      for i := 0 to n - 1 do
          count[(arr[i] / exp) % 10]++
      for i := 1 to 9 do
          count[i] += count[i - 1]
      for i := n - 1 down to 0 do
          output[count[(arr[i] / exp) % 10] - 1] := arr[i]
          Decrement count[(arr[i] / exp) % 10]
      Copy the output array to arr
  end CountSort
  ```
### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

### 2c. Evaluation plan - what and how will you measure and compare
- Random inputs
- Strong scaling
- Weak scaling


## 3. Project implementation
- Sample Sort:
   - MPI - MPI_Init(), MPI_BCast(), MPI_Scatter(), MPI_Gather(), MPI_Finalize()
   - Cuda -  Could not get it to compile since Grace is down.
- Mergesort:
  - MPI - MPI_Init(), MPI_Scatter(), MPI_Gather(), MPI_Barrier(), MPI_Finalize()
  - Cuda - Still trying to figure out how to approach the Cuda implementation, and Grace is down which makes it harder to experiment
- Odd-Even Transposition Sort:
  - MPI: MPI_Init(), MPI_Recv(), MPI_Bcast(), MPI_Scatter(), MPI_Sendrecv(), MPI_Finalize()
  - Cuda: Stuck on figuring out how to make processes in the GPU communicate with eachother as data needs to be transferred between processes inside the GPU, and Grace is down which makes it harder to experiment
- Radix Sort:
  - MPI - MPI_Init(), MPI_Bcast(), MPI_Send(), MPI_Recv(), MPI_finalize()
  - Cuda - Integration for MPI took longer than expected. Did not get a chance to test with Cuda before the Grace maintainance.

## 4. Performance Evaluation
- Sample Sort:
  - MPI -
    - (Explaination how it scales just by looking at the numbers on jupyter, and how we are working on plotting)
    - (Thicket Tree)
  - CUDA
    - (Explaination how it scales just by looking at the numbers on jupyter, and how we are working on plotting)
    - (Thicket Tree)
- Mergesort:
  - MPI -
    - Tried running the algorithm with larger array sizes & threds to observe how it scales. Read it into thicket to see the thicket tree and data frames, currently working on the plotting aspect of things.  
    - Thicket Tree:
        ```
          1.000 main_region
            └─ 1.000 whole_computation
               ├─ 1.000 check_correctness
               ├─ 1.000 comm
               │  ├─ 1.000 comm_gather
               │  └─ 1.000 comm_scatter
               ├─ 1.000 comp
               │  ├─ 1.000 comp_large
               │  └─ 1.000 comp_small
               └─ 1.000 data_init
        ```
  - CUDA -
    - Implemented the cuda version of mergesort algorithm and ensured that it scales and works with larger array sizes and threads. Read it into thicket to see the thicket tree and data frames, currently working on the plotting aspect of things.  
    - Thicket Tree:
        ```
        1.000 main_region
          ├─ 1.000 comp_small
          └─ 1.000 whole_computation
             ├─ 1.000 check_correctness
             ├─ 1.000 comm
             │  ├─ 1.000 comm_toDevice
             │  └─ 1.000 comm_toHost
             ├─ 1.000 comp
             │  └─ 1.000 comp_large
             └─ 1.000 data_init
          ```
- Odd-Even:
  - MPI -
    - Managed to get algorithm to scale, generated some CALI files and produced thicket tree. Working on plotting the CALI files. Looking at the data frame inside of Jupyter the algorithm seems to scale normally. 
    - Thicket Tree:
      ```
        1.000 comm_small
        1.000 comp_small
        1.000 whole_computation
        ├─ 1.000 check_correctness
        ├─ 1.000 comm
        │  └─ 1.000 comm_large
        ├─ 1.000 comp
        │  └─ 1.000 comp_large
        └─ 1.000 data_init
      ```
  - CUDA -
    - Managed to get algorithm to scale, generated some CALI files and produced thicket tree. Working on plotting the CALI files. Scaling for this CUDA as of now is somewhat unique as the blocks are always going to be array_size/2 as the block id determines which phase (either odd or even) gets completed. 
    - Thicket Tree:
      ```
      1.000 comm_small
      1.000 comp_small
      1.000 whole_computation
      ├─ 1.000 comm
      │  └─ 1.000 comm_large
      ├─ 1.000 comp
      │  └─ 1.000 comp_large
      └─ 1.000 data_init
      ```
- Radix Sort:
  - MPI -
    - (Explaination how it scales just by looking at the numbers on jupyter, and how we are working on plotting)
    - (Thicket Tree)
  - CUDA -
    - (Explaination how it scales just by looking at the numbers on jupyter, and how we are working on plotting)
    - (Thicket Tree) 
