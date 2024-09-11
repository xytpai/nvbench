nvcc $2 --std=c++17 -Iutils -arch sm_80 $1 -o a.out
