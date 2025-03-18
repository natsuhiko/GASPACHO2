#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DataFrame.cuh"


// sanity check
__global__ void printModelMatrix(DataFrame* Y, int N, int P){
	int i, k;
	double res;
	//for(i=0; i<N; i++){double *pd = (double*)(Y->x + i*sizeof(double)); printf("%lf,",*pd);}
	//for(i=0; i<N; i++){double *pd = (double*)(Y->x + i*sizeof(double)  +2*N*(sizeof(double))); printf("%lf,",*pd);}
	//for(i=0; i<N; i++){int    *pd = (int*)   (Y->x + i*sizeof(int)     +3*N*sizeof(double)); printf("%d,",*pd);}
	//for(i=0; i<N; i++){int    *pd = (int*)   (Y->x + i*sizeof(int)     +N*sizeof(double)); printf("%d,",*pd);}
	//printf("%d",Y->B);
	//for(i=0; i<1; i++){double *pd = (double*)(Y->x + i*sizeof(double)  +N*(sizeof(double)+sizeof(int))); printf("%lf,",*pd);}
	int strlen=0;
	Y->Knm = (double*)(Y->x + Y->b[Y->p-1]);
	for(i=0;i<Y->B;i++){if((Y->x)[i]=='\0'){break;}else{strlen++;}}
	//printf("N=%d P=%d p=%d M=%d B=%d strlen=%d %lf\n",Y->N, Y->P, Y->p, Y->M, Y->B, strlen, (Y->Knm)[0]);
	//for(i=0; i<N; i++){printf("%lf,",(Y->Knm)[i]);}
	//return;
	for(i=0; i<N; i++){
		printf("%d:",i);
		for(k=0; k<P; k++){
			//printf("mbk=%d\n",Y->mb[k]);
			if(Y->mc[k]==1){ // numerical
        			double *pd = (double*)(Y->x + Y->mb[k] + sizeof(double)*i);
        			res =  *pd;
    			}else{ // categorical
        			int *pi = (int*)(Y->x + Y->mb[k] + sizeof(int)*i);
       			 	res = (*pi==Y->mv[k] ? 1.0 : 0.0);
    			}
			printf("%lf ",res);
		}
		printf("\n");
	}
}

__device__ double getModelMatrix(DataFrame* Y, int i, int k){
    double res;
    if(Y->mc[k]==1){ // numerical
        double *pd = (double*)(Y->x + Y->mb[k] + sizeof(double)*i);
        res =  *pd;
    }else{ // categorical
        int *pi = (int*)(Y->x + Y->mb[k] + sizeof(int)*i);
        res = (*pi==Y->mv[k] ? 1.0 : 0.0);
    }
    return res;
}

DataFrame* loadDataFrame(char* fname){
	return loadDataFrame(fname, 0);
}

DataFrame* loadDataFrame(char* fname, int M){
    DataFrame* Y;
    Y=(DataFrame*)malloc(sizeof(DataFrame));
    Y->M=M;
    //printf("%s\n", fname);
    FILE* f; f=fopen(fname,"rb");
    fread(&(Y->N), sizeof(int), 1, f);
    fread(&(Y->p), sizeof(int), 1, f);
    Y->p++;// intercept
    if(M>0){Y->p++;} // Knm

    Y->c = (int*)malloc(sizeof(int)*(Y->p));
    Y->v = (int*)malloc(sizeof(int)*(Y->p));
    Y->b = (int*)malloc(sizeof(int)*(Y->p+1));
    Y->nh = (int*)malloc(sizeof(int)*(Y->p+1));
    Y->c[0] = 1;
    Y->v[0] = 1;
    fread(Y->c+1, sizeof(int), Y->p-1-(M>0 ? 1:0), f);
    fread(Y->v+1, sizeof(int), Y->p-1-(M>0 ? 1:0), f);
    if(M>0){Y->c[Y->p-1] = 1; Y->v[Y->p-1] = M;}

    int B;
    int NH;
    Y->b[0] = Y->nh[0] = 0;
    Y->b[1] = B = sizeof(double)*Y->N;
    Y->nh[1] = NH = 1;
    for(int j=1; j<Y->p; j++){
        B += (Y->N) * (Y->v[j]) * (Y->c[j]==1 ? sizeof(double) : sizeof(int));
        Y->b[j+1] = B;
        NH += (Y->v[j]) * (Y->c[j]);
        Y->nh[j+1] = NH;
    }
    B=0;
    Y->P=Y->nh[Y->p];
    Y->mc = (int*)malloc(sizeof(int)*(Y->P));
    Y->mv = (int*)malloc(sizeof(int)*(Y->P));
    Y->mb = (int*)malloc(sizeof(int)*(Y->P));
    Y->mk = (int*)malloc(sizeof(int)*(Y->P));
    for(int j=0; j<Y->p; j++){
        for(int k=Y->nh[j]; k<Y->nh[j+1]; k++){
            Y->mc[k] = Y->c[j];
            Y->mv[k] = k-Y->nh[j];
            Y->mb[k] = B + (Y->c[j]==1 ? sizeof(double)*(Y->N)*(Y->mv[k]) : 0);
            Y->mk[k] = j;
        }
        B += (Y->N) * (Y->v[j]) * (Y->c[j]==1 ? sizeof(double) : sizeof(int));
    }
    Y->B = B;
    printf("Total bytes of data body in %s = %d\n", fname, B);

    Y->x = (char*)malloc(sizeof(char)*B);
    double* pd; pd = (double*)Y->x;
    for(int i=0; i<Y->N; i++){pd[i] = 1.0;}
    fread(Y->x+sizeof(double)*(Y->N), sizeof(char), B-sizeof(double)*(Y->N)*(1+M), f);

    fclose(f);

    Y->Knm = (double*)(Y->x+Y->b[Y->p-1]); // meaningless for M=0
    Y->Knm[0]=99.0;

    return Y;
}


DataFrame* newDataFrameOnDevice(DataFrame *Z_h){
    // intermediate Y
    DataFrame *iZ, *Z;
    iZ = (DataFrame*)malloc(sizeof(DataFrame));

    cudaMemcpy(iZ,    Z_h,      sizeof(DataFrame),        cudaMemcpyHostToHost);
    if(cudaMalloc((void **) &(iZ->x),  sizeof(char)*(Z_h->B))  == cudaErrorMemoryAllocation){printf("iZ->x not allocated.\n"); return NULL;}else{printf("iZ->x %d allocated.\n", Z_h->B);};
    cudaMalloc((void **) &(iZ->c),  sizeof(int) *(Z_h->p));
    cudaMalloc((void **) &(iZ->v),  sizeof(int) *(Z_h->p));

    cudaMalloc((void **) &(iZ->b),  sizeof(int) *(Z_h->p+1));
    cudaMalloc((void **) &(iZ->nh), sizeof(int) *(Z_h->p+1));

    cudaMalloc((void **) &(iZ->mc),  sizeof(int)*(Z_h->P));
    cudaMalloc((void **) &(iZ->mv),  sizeof(int)*(Z_h->P));
    cudaMalloc((void **) &(iZ->mb),  sizeof(int)*(Z_h->P));
    cudaMalloc((void **) &(iZ->mk),  sizeof(int)*(Z_h->P));

Z_h->Knm[0]=199.0;
    if(cudaSuccess != cudaMemcpy(iZ->x,   Z_h->x,   sizeof(char)*(Z_h->B),    cudaMemcpyHostToDevice)){printf("Z_h->x was not coppied...\n");}else{printf("Z_h->x was coppied.\n");};
    iZ->Knm = (double*)(iZ->x + Z_h->b[Z_h->p-1]);
    //iZ->Knm = (double*)(iZ->x);
    //iZ->Knm += (Z_h->b[Z_h->p-1]);

    cudaMemcpy(iZ->c,   Z_h->c,   sizeof(int)*(Z_h->p),   cudaMemcpyHostToDevice);
    cudaMemcpy(iZ->v,   Z_h->v,   sizeof(int)*(Z_h->p),   cudaMemcpyHostToDevice);

    cudaMemcpy(iZ->b,   Z_h->b,   sizeof(int)*(Z_h->p+1), cudaMemcpyHostToDevice);
    cudaMemcpy(iZ->nh,  Z_h->nh,  sizeof(int)*(Z_h->p+1), cudaMemcpyHostToDevice);

    cudaMemcpy(iZ->mc,  Z_h->mc,  sizeof(int)*(Z_h->P),   cudaMemcpyHostToDevice);
    cudaMemcpy(iZ->mv,  Z_h->mv,  sizeof(int)*(Z_h->P),   cudaMemcpyHostToDevice);
    cudaMemcpy(iZ->mb,  Z_h->mb,  sizeof(int)*(Z_h->P),   cudaMemcpyHostToDevice);
    cudaMemcpy(iZ->mk,  Z_h->mk,  sizeof(int)*(Z_h->P),   cudaMemcpyHostToDevice);

    cudaMalloc((void **) &(Z),  sizeof(DataFrame));
    cudaMemcpy(Z,  iZ,  sizeof(DataFrame),   cudaMemcpyHostToDevice);

    return Z;
}



__global__ void printMetaDataFrame(DataFrame* Y){
    printf("%d\n", Y->P);
    for(int i=0; i<Y->P; i++){
        printf("%d %d %d %d\n", Y->mk[i], Y->mc[i], Y->mv[i], Y->mb[i]);
    }
}



void printMetaDataFrameHost(DataFrame* Y){
    printf("%d\n", Y->P);
    for(int i=0; i<Y->P; i++){
        printf("%d %d %d %d\n", Y->mk[i], Y->mc[i], Y->mv[i], Y->mb[i]);
    }
}




void printModelMatrixHost(DataFrame* Y, int N, int P){
	int i, k;
	double res;
	for(i=0; i<N; i++){
		for(k=0; k<P; k++){
			if(Y->mc[k]==1){ // numerical
        			double *pd = (double*)(Y->x + Y->mb[k] + sizeof(double)*i);
        			res =  *pd;
    			}else{ // categorical
        			int *pi = (int*)(Y->x + Y->mb[k] + sizeof(int)*i);
       			 	res = (*pi==Y->mv[k] ? 1.0 : 0.0);
    			}
			printf("%lf ",res);
		}
		printf("\n");
	}
}


