# CSCE 435 Group project

## 1. Group members:
1. Riley Szecsy
2. Vincent Lobello
3. Jonathan Kutsch
4. Nebiyou Ersabo
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
  -  Presudocode:
    **procedure** ODD-EVEN PAR(n) <br>
    **begin** <br>
      id := proccees's label <br>
      **for** i := 1 to n **do** <br>
      **begin** <br>
         **if** i is odd **then** <br>
            *compare-exchange min (id+1);* <br>
         **else** <br>
             *compare-exchange max(id-1);* <br>
       **if** i is even **then** <br>
           **if** id is even **then** <br>
               *compare-exchange min(id+1);* <br>
           **else** <br>
               *compare-exhange max(id-1);* <br>
       **end for** <br>
     **end** ODD-EVEN PAR <br>
       
