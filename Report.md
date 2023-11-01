# CSCE 435 Group project

## 1. Group members:
1. Riley Szecsy
2. Vincent Lobello
3. Jonathan Kutsch
4. Nebiyou Ersabo <br>
Group is communicating over groupme. 
---

## 2. _due 10/25_ Project topic
Comparing the performance of comparision based sorting algorithms. We will be scaling the GPU settings and change the number of threads and proccessors of the CPU based on problem size. We will be testing these on sorted, random, and reverse sorted inputs.  

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

- Quicksort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode: <br>
  ```
  procedure QUICKSORT (A, q, r )
  begin
    if g < r then
      begin
        X:= A[q];
        S:= q;<br>
        for i:=q+1 to r do:
          if A[i] <= x then
          begin
            s:=s+1
            swap(A[s], A[i]);
          end if
      swap(A[q], A[s]);
      QUICKSORT (A, q, s)
      QUICKSORT (A, s + 1, r )
      end if
  end QUICKSORTlp
```
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
  -  Presudocode: <br>
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
