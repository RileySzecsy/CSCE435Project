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
  - Pesudocode 
    
  ```

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
- Mergesort:
- Odd-Even Transposition Sort:
  - MPI: MPI_Recv
  - Cuda: Explaination on why you cant do it or why its not possible
- Radix Sort:


