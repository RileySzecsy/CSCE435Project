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
   - Cuda -  I beleive I finished the implementation but Grace has been running slow and will not run my jobs
- Mergesort:
  - MPI - MPI_Init(), MPI_Scatter(), MPI_Gather(), MPI_Barrier(), MPI_Finalize()
  - Cuda - Completed CUDA implementation
- Odd-Even Transposition Sort:
  - MPI: MPI_Init(), MPI_Recv(), MPI_Bcast(), MPI_Scatter(), MPI_Sendrecv(), MPI_Finalize()
  - Cuda: Stuck on figuring out how to make processes in the GPU communicate with eachother as data needs to be transferred between processes inside the GPU, and Grace is down which makes it harder to experiment
- Radix Sort:
  - MPI - MPI_Init(), MPI_Bcast(), MPI_Send(), MPI_Recv(), MPI_finalize()
  - Cuda - Integration for MPI took longer than expected. Did not get a chance to test with Cuda before the Grace maintainance.

## 4. Performance Evaluation
- Sample Sort:
  - MPI -
    - By running the algorithm with different input sizes and threads, I was able to observe the timing and scaling of the algorithm.
    - Thicket Tree
      ```
      1.000 main
      ├─ 1.000 comm
      │  ├─ 1.000 comm_large
      │  └─ 1.000 comm_small
      ├─ 1.000 comp
      │  └─ 1.000 comp_large
      │     └─ 1.000 comp_small
      └─ 1.000 data_init
       ```
  - CUDA
    - Grace has paused my jobs and will not run them. I have over 5 queued jobs and all of them are waiting on Grace.
    - Unable to prodice thicket tree because Grace is not running my jobs. They have been queued for over an hour
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
      1.000 main
      └─ 1.000 whole_computation
         ├─ 1.000 check_correctness
         ├─ 1.000 comm
         │  └─ 1.000 comm_large
         │     ├─ 1.000 MPI_Gather
         │     ├─ 1.000 MPI_Recv
         │     ├─ 1.000 MPI_Scatter
         │     └─ 1.000 MPI_Send
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
    - I got the algorithm to scale, got a lot of CALI files and produced a thicket tree; more files will be produced in coming trials. Working on plotting more CALI files. Looking at the data frame inside of Jupyter the algorithm seems to scale normally.
    - Thicket Tree:
      ```
      1.000 data_init
      ├─ 1.000 check_correctness
      ├─ 1.000 comm
      └─ 1.000 radix_sort_main
         └─ 1.000 radix_sort_whole_computation
            └─ 1.000 radix_sort_counting_sort
               └─ 1.000 counting_sort_small
      ``` 
  - CUDA -
    - I was able to produce the CUDA implementation to its fullest, got a ton of CALI files for testing purposes, produced a thicket tree as seen below and I am going to continue to test with more input sizes for data analysis. Looking at the data frame inside of Jupyter the algorithm seems to scale normally.
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


## Final Report (Plots images are within each implementations folders)

- Merge Sort:
  - MPI:
      - Strong Scaling: With constant problem size and increased number of processors we expected to see a better time. We also expected that there could be diminishing results due to communication overheads. This expectaion was validated with trend from our computation graph below for array size of 2^24. As we can see from the plot the general trend is that computaiton time goes down up to 128 threads and then it starts declining at lower rates possibly due to communication overhead.
        
        ![img](https://github.com/RileySzecsy/CSCE435Project/blob/master/MergeSort/MPI/Plotting/MPI_plots/Strong%20Scaling%20MPI%20%7C%20comp%20%7C%20Size%3A%3A%2016777216.png)

        For whole computation, from the graph below of array size 2^24, we can see that as the number of processes increases the overall time decreases upto 64 nodes, and then we start getting diminishing returns due to communication overhead.
        
        ![img](https://github.com/RileySzecsy/CSCE435Project/blob/master/MergeSort/MPI/Plotting/MPI_plots/Strong%20Scaling%20MPI%20%7C%20whole_computation%20%7C%20Size%3A%3A%2016777216.png)
        
      - Speedup: Since speedup shows us how much faster the algorithm runs on multiple processors compared to one, we were wxpecting to see a more linear or logarithmic graph. This expectation was also validated with the computation graph for array size 2^24 with sorted input type.
        
        ![img](https://github.com/RileySzecsy/CSCE435Project/blob/master/MergeSort/MPI/Plotting/MPI_plots/Speedup%20MPI%20%7C%20comp%20%7C%20Input%20Type%3A%3A%20Sorted.png)

        However, the the whole_computation graph shown did not speed up as expected due to inefficient data initialization methods which took a longer computation time for larger arrays and communication overhead.
        
       ![img](https://github.com/RileySzecsy/CSCE435Project/blob/master/MergeSort/MPI/Plotting/MPI_plots/Speedup%20MPI%20%7C%20whole_computation%20%7C%20Input%20Type%3A%3A%20Sorted.png)


      -  Weak Scaling: when we increase both the problem size and threads proportionally we expected the runtimes to be relatively constant since it would be a constant load per processor.


  - CUDA:

 
- Odd-Even Sort:
  - MPI: The MPI implementation of odd-even sort algorithm time is going down based on the number of processes when the array size is fixed which is expected which is shown by this [plot](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Strong%20Scaling%20MPI%20_%20comp_large%20_%20Size__%20268435456.png). However when looking at a [speedup graph](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Strong%20Scaling%20Speedup%20MPI%20_%20comp_large%20_%20Input%20Type__%20Sorted.png) for lower array sizes (2^16, 2^18, 2^20) the optimal number of processes is 64. This makes sense as these sizes of arrays may not need as many processes which could result in processes waiting which adds time. For bigger array sizes (2^22, 2^24, 2^26, 2^28) speedup is still happening even at 1024 processes which is reasonable as these array sizes have more work that needs to be done which can be leveraged by more processes. As for the whole_computation there are similar trends going on with the [strong scaling](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Strong%20Scaling%20MPI%20_%20whole_computation%20_%20Size__%20268435456.png) and [speedup](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Strong%20Scaling%20Speedup%20MPI%20_%20whole_computation%20_%20Input%20Type__%20Sorted.png) plot as the whole_computation region is dominated by comp/comp_large. The only thing different about the plots when looking at the whole computation is that there is a dip in the speedup graph for sizes 2^16 and 2^18 however this is most likely due to resource allocation when queuing jobs as those particular jobs may have been allocated nodes that were father apart causing the unusual dip. When looking at a [weak scaling graph](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Weak%20Scaling%20MPI%20_%20comp_large%20_%20Input%20Type__%20Sorted.png) that has similar results to the speedup graphs, for smaller array sizes (2^16, 2^18, 2^20) there is a valley at 64 processes which is the same as the optimal number of processes found in the speedup graph. For bigger arrays the scaling is slowly going down which also reflects what the speedup graph was showing. When looking at a [weak scaling graph](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/MPI/Plotting/Plots/Weak%20Scaling%20MPI%20_%20whole_computation%20_%20Input%20Type__%20Sorted.png) for the whole_computation the trends are almost identical, smaller array sizes (2^16, 2^18, 2^20) start to skew up a processes increase where bigger array sizes (2^22, 2^24, 2^26, 2^28) are basically flat across.  The same dip occurs which adds more evidence to that being a resource allocation anomaly rather than a problem with the algorithm.  

  - CUDA: The CUDA implementation of odd-even sort does not scale well as evidenced by the [strong scaling plot](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/CUDA/Plotting/Plots/Strong%20Scaling%20CUDA%20_%20whole_computation%20_%20Size__%204194304.png), [speedup plot](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/CUDA/Plotting/Plots/Strong%20Scaling%20Speedup%20CUDA%20_%20comp%20_%20Input%20Type__%20Sorted.png). The reason I think this is due to the personal implementation of the algorithm. The number of blocks used in the kernel call is [InputSize/2](https://github.com/RileySzecsy/CSCE435Project/blob/f86db5e18009593409e3cb48610f0c976ebb7c78/Odd-Even/CUDA/cudaoddeven.cu#L170), it was originally attempted with n/threads number of blocks however the array would end up unsorted. InputSize/2 blocks is significantly more blocks than InputSize/threads which means more blocks are trying to access the same memory and resources which could be the reason for the poor scaling in the CUDA implementation of odd-even sort. When looking at the [weak scaling plot](https://github.com/RileySzecsy/CSCE435Project/blob/master/Odd-Even/CUDA/Plotting/Plots/Weak%20Scaling%20CUDA%20_%20whole_computation%20_%20Input%20Type__%20Sorted.png) the trends are consistent and as shown by the almost horizontal lines however these plots do not give very much insight as to why there is poor scaling.
 
  The plots linked in the report are the ones that  clearly and concisely represent the algorithm's behavior. If more context is wanted or needed all of the [MPI](https://github.com/RileySzecsy/CSCE435Project/tree/master/Odd-Even/MPI/Plotting/Plots) and [CUDA](https://github.com/RileySzecsy/CSCE435Project/tree/master/Odd-Even/CUDA/Plotting/Plots) odd-even plots are located within the repo. 

 
- Radix Sort:
  - MPI:

  - CUDA:


- Sample Sort:
  - MPI:

  - CUDA:

