#include <stdio.h>

typedef struct{
        double* x; // length of L
        int* p; // length of J+1
        int* i; // length of L and value ranging [0..N-1]
        int L; // N of non-zeros
        int N; // N of cells
        int J; // N of genes
} SparseMat;


SparseMat* loadSparseMat(char* fname);
SparseMat* newSparseMatOnDeviceFromDenseMat(int N, int J, double* Z, int trans, int* map_d);
SparseMat* newSparseMatOnDevice(SparseMat* Y_h);


__global__ void transpose_d(SparseMat *Y, SparseMat *Yt, int* map);
__global__ void printYonDevice(SparseMat* Y);
