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
  -  Pesudocode:
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
       
