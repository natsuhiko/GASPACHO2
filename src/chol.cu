#include <stdio.h>
#include <stdlib.h>
#include "SparseMat.cuh"
#include "util_cuda.cuh"
//#include "dsyev.cuh"

__device__ void getRb(double* Lmat, double* b, int n){
    int i, j;
    double tmp;
    for(j=0; j<n; j++){
	tmp = 0.0;
        for(i=j; i<n; i++){	
            tmp += Lmat[i+j*n]*b[i];
	}
	b[j] = tmp;
    }
}

__device__ void choldc(double *a, int n){
    int i, j, k;
    double sum;
    for (i=0; i<n; i++) {
        for (j=i; j<n; j++) {
            for (sum=a[i+j*n], k=i-1; k>=0; k--){ sum -= a[i+k*n]*a[j+k*n]; }
            if (i == j) {
                if (sum <= 0.0){
		    //printf("failed\n");
                    fill0_d(a, n*n); return;
                }
                //p[i]=sqrt(sum);
                a[j+i*n] = sqrt(sum);
            } else {
                //a[j+i*n]=sum/p[i];
                a[j+i*n]=sum/a[i+i*n];
            }
        }
    }
    /*for (i=0; i<n; i++) { // upper tri to be 0
        for (j=i+1; j<n; j++) {
            a[i+j*n] = 0.0;
        }
    }*/
}

__device__ void cholsl(double *a, int n, double* b, double* x){
    int i, k;
    double sum;
    for (i=0; i<n; i++) { //Solve L · y = b, storing y in x.
        for (sum=b[i],k=i-1; k>=0; k--) sum -= a[i+k*n]*x[k];
        x[i]=sum/a[i+i*n];
    }
    for (i=n-1; i>=0; i--) { //Solve LT · x = y.
        for (sum=x[i], k=i+1; k<n; k++) sum -= a[k+i*n]*x[k];
        x[i]=sum/a[i+i*n];
    }
}

__device__ void forwardsl(double *a, int n, double* x){
	int i, k;
	double sum;
	for (i=0; i<n; i++) { //Solve L · y = x, storing y in x.
		for (sum=x[i],k=i-1; k>=0; k--) sum -= a[i+k*n]*x[k];
		x[i]=sum/a[i+i*n];
	}
}


__device__ void forwardslg(double *a, int n, double* x, int ldx){
	int i, k;
	double sum;
	for (i=0; i<n; i++) { //Solve L · y = x, storing y in x.
		for (sum=x[i*ldx],k=i-1; k>=0; k--) sum -= a[i+k*n]*x[k*ldx];
		x[i*ldx]=sum/a[i+i*n];
	}
}

__device__ void backsl(double *a, int n, double* x){
	int i, k;
	double sum;
	for (i=n-1; i>=0; i--) { //Solve LT · x = y.
		for (sum=x[i], k=i+1; k<n; k++) sum -= a[k+i*n]*x[k];
		x[i]=sum/a[i+i*n];
	}
}

__device__ void cholslg(double *a, int n, double* b, int ldb, double* x, int ldx){
    int i, k;
    double sum;
    for (i=0; i<n; i++) { //Solve L · y = b, storing y in x.
        for (sum=b[i*ldb],k=i-1; k>=0; k--) sum -= a[i+k*n]*x[k*ldx];
        x[i*ldx]=sum/a[i+i*n];
    }
    for (i=n-1; i>=0; i--) { //Solve LT · x = y.
        for (sum=x[i*ldx], k=i+1; k<n; k++) sum -= a[k+i*n]*x[k*ldx];
        x[i*ldx]=sum/a[i+i*n];
    }
}









