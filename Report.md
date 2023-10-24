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
    **procedure** ODD-EVEN PAR(n)
    **begin**
      id := proccees's label
      **for** i := 1 to n **do**
      **begin**
         **if** i is odd **then**
            *compare-exchange min (id+1);*
         **else**
             *compare-exchange max(id-1);*
       **if** i is even **then**
           **if** id is even **then**
               *compare-exchange min(id+1);*
           **else**
               *compare-exhange max(id-1);*
       **end for**
     **end** ODD-EVEN PAR
       
