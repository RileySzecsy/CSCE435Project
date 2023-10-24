# CSCE 435 Group project

## 1. Group members:
1. Riley Szecsy
2. Vincent Lobello
3. Jonathan Kutsch
4. Nebiyou Ersabo <br>
Group is communicating over groupme. 
---

## 2. _due 10/25_ Project topic
Comparing the performance of comparision based sorting algorithms. 

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

- Quicksort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode: <br>
  procedure QUICKSORT (A, q, r )
  **begin**<br>
  &emsp;**if** g < r **then**<br>
  &emsp;&emsp;**begin**<br>
  &emsp;&emsp;X:= A[q];<br>
  &emsp;&emsp;S:= q;<br>
  &emsp;&emsp;**for** i:=q+1 to r **do**:<br>
  &emsp;&emsp;&emsp;if A[i] <= x then <br>
  &emsp;&emsp;&emsp;**begin** <br>
  &emsp;&emsp;&emsp;&emsp;s:=s+1;<br>
  &emsp;&emsp;&emsp;&emsp;swap(A[s], A[i]);<br>
  &emsp;&emsp;&emsp;**end if** <br>
  &emsp;&emsp;swap(A[q], A[s]); <br>
  &emsp;&emsp;QUICKSORT (A, q, s); <br>
  &emsp;&emsp;QUICKSORT (A, s + 1, r ); <br>
  &emsp;**end if** <br>
  **end** QUICKSORT <br>
- Mergesort (MPI + CUDA)
  -  MPI on each core
  -  Pesudocode:
- Odd-Even Transposition Sort (MPI + CUDA)
  -  MPI on each core
  -  Presudocode: <br>
    **procedure** ODD-EVEN PAR(n) <br>
    **begin** <br>
      &emsp; id := proccees's label <br>
      &emsp; **for** i := 1 to n **do** <br>
      &emsp; **begin** <br>
      &emsp;&emsp; **if** i is odd **then** <br>
      &emsp;&emsp;&emsp;**if** id is odd **then** <br>
      &emsp;&emsp;&emsp;&emsp;*compare-exchange min (id+1);* <br>
      &emsp;&emsp;&emsp;**else** <br>
      &emsp;&emsp;&emsp;*compare-exchange max(id-1);* <br>
      &emsp;&emsp;**if** i is even **then** <br>
      &emsp;&emsp;&emsp;**if** id is even **then** <br>
      &emsp;&emsp;&emsp;&emsp;*compare-exchange min(id+1);* <br>
      &emsp;&emsp;&emsp;**else** <br>
      &emsp;&emsp;&emsp;&emsp;*compare-exhange max(id-1);* <br>
       &emsp;**end for** <br>
     **end** ODD-EVEN PAR <br>
       
