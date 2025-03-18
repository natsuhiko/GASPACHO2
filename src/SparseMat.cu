#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "SparseMat.cuh"

__global__ void transpose_d(SparseMat *Y, SparseMat *Yt, int* map){
        for(int l=0; l<Y->L; l++){
                Yt->x[map[l]] = Y->x[l];
        }
        /*Yt->N = Y->J;
        Yt->J = Y->N;
        Yt->L = Y->L;
        for(int j=0; j<Yt->J; j++){Yt->p[j+1]=0;}
        for(int j=0; j<Y->J; j++){
            for(int l=Y->p[j]; l<Y->p[j+1]; l++){
                Yt->p[Y->i[l]+1]++;
                Yt->i[l] = j;
                Yt->x[l] = Y->x[l];
                tmp[l] = Y->i[l];
            }
        }
        for(int i=0; i<Yt->J; i++){
            Yt->p[i+1] += Yt->p[i];
        }
        bsort(tmp, Yt->i, Yt->x, Yt->L);*/
}


__global__ void printYonDevice(SparseMat* Y){
        printf("%d %d %d\n", Y->N, Y->J, Y->L);
        for(int i=0; i<100; i++){printf("%d %lf\n", Y->i[i], Y->x[i]); }
}

SparseMat* loadSparseMat(char* fname){
        SparseMat *Y;
        Y = (SparseMat*)malloc(sizeof(SparseMat));
        FILE* f; f=fopen(fname,"rb");
        fread(&(Y->N), sizeof(int), 1, f);
        fread(&(Y->J), sizeof(int), 1, f);
        fread(&(Y->L), sizeof(int), 1, f);
        Y->i = (int*)malloc(sizeof(int)*(Y->L+1));
        Y->p = (int*)malloc(sizeof(int)*(Y->J+1));
        Y->x = (double*)malloc(sizeof(double)*(Y->L+1));
        fread(Y->i, sizeof(int), Y->L, f);
        fread(Y->p, sizeof(int), Y->J+1, f);
        fread(Y->x, sizeof(double), Y->L, f);
        fclose(f);
        return Y;
}

SparseMat* newSparseMatOnDeviceFromDenseMat(int N, int J, double* Z, int trans, int* map_d){
        int L=0;
        for(int l=0; l<N*J; l++){
            if(Z[l]!=0.0){L++;}
        }
        SparseMat* Y_h;
        Y_h = (SparseMat*)malloc(sizeof(SparseMat));
        if(trans>0){
                Y_h->N = J;
                Y_h->J = N;
        }else{
                Y_h->N = N;
                Y_h->J = J;
        }
        Y_h->L = L;
        Y_h->i = (int*)malloc(sizeof(int)*(L+1));
        Y_h->p = (int*)malloc(sizeof(int)*(Y_h->J+1)); Y_h->p[0] = 0;
        Y_h->x = (double*)malloc(sizeof(double)*(L+1));
        int *map, *iZ;
        if(trans>0){
                map=(int*)malloc(sizeof(int)*Y_h->L);
                iZ=(int*)malloc(sizeof(int)*Y_h->N*Y_h->J);
        }
        int l=0;
        for(int j=0; j<Y_h->J; j++){
            Y_h->p[j+1] = Y_h->p[j];
            for(int i=0; i<Y_h->N; i++){
                if(trans>0){
                    if(Z[j+i*N]!=0.0){
                        iZ[j+i*N] = l;
                        Y_h->i[l] = i;
                        Y_h->p[j+1]++;
                        Y_h->x[l] = Z[j+i*N];
                        l++;
                    }
                }else{
                    if(Z[i+j*N]!=0.0){
                        Y_h->i[l] = i;
                        Y_h->p[j+1]++;
                        Y_h->x[l] = Z[i+j*N];
                        l++;
                    }
                }
            }
        }
        if(trans>0){
                l=0;
                for(int i=0; i<Y_h->N*Y_h->J; i++){
                        if(Z[i]!=0.0){
                                map[l] = iZ[i];
                                l++;
                        }
                }
        }
		SparseMat *Y, *iY;
        // intermediate Y
        iY = (SparseMat*)malloc(sizeof(SparseMat));
        cudaMemcpy(iY,    Y_h,      sizeof(SparseMat),        cudaMemcpyHostToHost);
        cudaMalloc((void **) &(iY->i),  sizeof(int)   *(Y_h->L+1));
        cudaMalloc((void **) &(iY->p),  sizeof(int)   *(Y_h->J+1));
        cudaMalloc((void **) &(iY->x),  sizeof(double)*(Y_h->L+1));
        cudaMemcpy(iY->i,   Y_h->i,   sizeof(int)*   Y_h->L,    cudaMemcpyHostToDevice);
        cudaMemcpy(iY->p,   Y_h->p,   sizeof(int)*  (Y_h->J+1), cudaMemcpyHostToDevice);
        cudaMemcpy(iY->x,   Y_h->x,   sizeof(double)*Y_h->L,    cudaMemcpyHostToDevice);
        // full Y
        cudaMalloc((void **) &Y,  sizeof(SparseMat));
        cudaMemcpy(Y,     iY,      sizeof(SparseMat),        cudaMemcpyHostToDevice);

        free(Y_h->i);free(Y_h->p);free(Y_h->x);free(Y_h);
        if(trans>0){cudaMemcpy(map_d, map, sizeof(int)*Y_h->L, cudaMemcpyHostToDevice); free(map); free(iZ);}
        return Y;
}

SparseMat* newSparseMatOnDevice(SparseMat* Y_h){
        SparseMat *Y, *iY;
        // intermediate Y
        if((iY = (SparseMat*)malloc(sizeof(SparseMat)))==NULL){printf("iY is not alloced.\n"); return NULL;};
        cudaMemcpy(iY,    Y_h,      sizeof(SparseMat),        cudaMemcpyHostToHost);
        if(cudaMalloc((void **) &(iY->i),  sizeof(int)   *(Y_h->L+1)) == cudaErrorMemoryAllocation){printf("iY->i not allocated.\n"); return NULL;};
        if(cudaMalloc((void **) &(iY->p),  sizeof(int)   *(Y_h->J+1)) == cudaErrorMemoryAllocation){printf("iY->p not allocated.\n"); return NULL;};
        if(cudaMalloc((void **) &(iY->x),  sizeof(double)*(Y_h->L+1)) == cudaErrorMemoryAllocation){printf("iY->x not allocated.\n"); return NULL;};
        cudaMemcpy(iY->i,   Y_h->i,   sizeof(int)*   Y_h->L,    cudaMemcpyHostToDevice);
        cudaMemcpy(iY->p,   Y_h->p,   sizeof(int)*  (Y_h->J+1), cudaMemcpyHostToDevice);
        cudaMemcpy(iY->x,   Y_h->x,   sizeof(double)*Y_h->L,    cudaMemcpyHostToDevice);
        // full Y
        cudaMalloc((void **) &Y,  sizeof(SparseMat));
        cudaMemcpy(Y,     iY,      sizeof(SparseMat),        cudaMemcpyHostToDevice);
        return Y;
}


