#include <stdio.h>
#include <math.h>
//#include "DataFrame.cuh"
#include "SparseMat.cuh"
#include "util_cuda.cuh"
#include "chol.cuh"
#include "kernel_funcs.cuh"

__global__ void getKernelMatsDense(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* K, double* Kinv, double* KinvKmn){
	//getKernelMatsOnlyARDSE(N, P, M, Q, X, Ta, rho, Z, K, Kinv, KinvKmn);
	getKernelMatsWithPer(N, P, M, Q, X, Ta, rho, Z+N*(P-M), K, Kinv, KinvKmn);
}



// q in 0..Q-1
__global__ void getKernelMats(int Q, double* Xi, double* Ta, double* rho, DataFrame *Z, int q, double* K, double* Kinv, double* KinvKmn){
	getKernelMatsWithPerSpecificLV(Z->N, Z->P, Z->M, Q, Xi, Ta, rho, q, Z->Knm, K, Kinv, KinvKmn);
}


__global__ void getKernelMats(int Q, double* Xi, double* Ta, double* rho, DataFrame *Z, double* K, double* Kinv, double* KinvKmn){
	getKernelMatsWithPer(Z->N, Z->P, Z->M, Q, Xi, Ta, rho, Z->Knm, K, Kinv, KinvKmn);
}


__global__ void getKernelMatsSparse(int p, int* nh, int M, int Q, double* X, double* Ta, double* rho, SparseMat *Z, double* K, double* Kinv, double* KinvKmn){
	getKernelMatsWithPer(Z->N, Z->J, M, Q, X, Ta, rho, Z->x + Z->p[nh[p-1]], K, Kinv, KinvKmn);
}

// not used
__global__ void getKernelMatsPer(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* K, double* Kinv, double* KinvKmn){
	double sum = 0.0, tmp;
	//double *Knm; Knm = Z + N*(P-M);
	int i = blockIdx.x;
	int j = blockIdx.y;
	if(j<M){if(i<N){ // Knm
		tmp = sin((X[i]-Ta[j])/2.0);
		sum += tmp*tmp/rho[0];
		Z[N*(P-M) + i+j*N] = exp(-sum);
	}else if(i>=N && i<(N+M)){ // Kmm
		int ii = i-N;
		if(ii==j){
			K[ii+j*M] = 1.0;
		}else if(ii>j){
			K[ii+j*M] = 0.0;
		}else if(ii<j){
			tmp = sin((Ta[ii]-Ta[j])/2.0);
			sum += tmp*tmp/rho[0];
			K[ii+j*M] = exp(-sum);
		}
		Kinv[ii+j*M] = K[ii+j*M];
	if(j==(M-1)&&ii==(M-2)){
		printf("K[%d,%d]=%lf\n",ii,j, K[ii+j*N]);
	}
	}}
}


// with periodic
__device__ void getKernelMatsWithPer(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Knm, double* K, double* Kinv, double* KinvKmn){
	int k;
	double sum = 0.0, tmp;
	int i = blockIdx.x;
	int j = blockIdx.y;
	//printf("%d %d %d %d %d\n",i,j,N,M,Q);
	if(j<M){if(i<N){ // Knm
		tmp = sin((X[i]-Ta[j])/2.0);
		sum += tmp*tmp/rho[0];
		for(k=1; k<Q; k++){
			tmp = X[i+k*N]-Ta[j+k*M];
			sum += tmp*tmp/2.0/rho[k];
		}
		Knm[i+j*N] = exp(-sum);
		//printf("%lf\n", exp(-sum));
	}else if(i>=N && i<(N+M)){ // Kmm
		int ii = i-N;
		if(ii==j){
			K[ii+j*M] = 1.0;
		}else if(ii>j){
			K[ii+j*M] = 0.0;
		}else if(ii<j){
			tmp = sin((Ta[ii]-Ta[j])/2.0);
			sum += tmp*tmp/rho[0];
			for(k=1; k<Q; k++){
				tmp = Ta[ii+k*M]-Ta[j+k*M];
				sum += tmp*tmp/2.0/rho[k];
			}
			K[ii+j*M] = exp(-sum);
		}
		Kinv[ii+j*M] = K[ii+j*M];
	//if(j==(M-1)&&ii==(M-2)){
	//	printf("K[%d,%d]=%lf\n",ii,j, K[ii+j*N]);
	//}
	
	}}
}


// with periodic id specified by q
__device__ void getKernelMatsWithPerSpecificLV(int N, int P, int M, int Q, double* X, double* Ta, double* rho, int q, double* Knm, double* K, double* Kinv, double* KinvKmn){
	int k;
	double sum = 0.0, tmp;
	int i = blockIdx.x;
	int j = blockIdx.y;
	if(j<M){if(i<N){ // Knm
		if(q==0){
			tmp = sin((X[i]-Ta[j])/2.0);
			sum += tmp*tmp/rho[0];
		}else{
			k = q;
			tmp = X[i+k*N]-Ta[j+k*M];
			sum += tmp*tmp/2.0/rho[k];
		}
		Knm[i+j*N] = exp(-sum);
	}else if(i>=N && i<(N+M)){ // Kmm
		int ii = i-N;
		if(ii==j){
			K[ii+j*M] = 1.0;
		}else if(ii>j){
			K[ii+j*M] = 0.0;
		}else if(ii<j){
			if(q==0){
			tmp = sin((Ta[ii]-Ta[j])/2.0);
			sum += tmp*tmp/rho[0];
			}else{
				k = q;
				tmp = Ta[ii+k*M]-Ta[j+k*M];
				sum += tmp*tmp/2.0/rho[k];
			}
			K[ii+j*M] = exp(-sum);
		}
		Kinv[ii+j*M] = K[ii+j*M];
	}}
}



__global__ void cholKmm(int M, double* Kinv){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	//printM_d(Kinv,10,M,M,0);
	if(i==0) choldc(Kinv, M);
}

__global__ void getKinvKmnDense(int N, int P, int M, double* Kinv, double* Z, double* KinvKmn){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(i<N){
		double *Knm; Knm = Z + N*(P-M);
		cholslg(Kinv, M, Knm+i, N, KinvKmn+i*M, 1);
	}
}


__global__ void getKinvKmn(double* Kinv, DataFrame* Z, double* KinvKmn, double* titsias){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	int N = Z->N;
	if(i<N){
		int M = Z->M;
		cholslg(Kinv, M, Z->Knm+i, N, KinvKmn+i*M, 1);
		titsias[i] = ( 1.0 - nk_fdot(Z->Knm+i, N, KinvKmn+i*M, 1, M) )/2.0;
	}
}


__global__ void getKinvKmnSparse(int p, int* nh, int N, int P, int M, double* Kinv, SparseMat* Z, double* KinvKmn){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(i<N){
		double *Knm; Knm = Z->x + Z->p[nh[p-1]];
		cholslg(Kinv, M, Knm+i, N, KinvKmn+i*M, 1);
	}
}

__global__ void printK(int N, double* K){
		printM_d(K, 10, 10, N, 0);
}

// skipping cell cycle kernel
__global__ void getKernelTarget(int N, int M, int Q, double *X, double* Ta, double* rho, double* Knmta, double* Kta, double* Ktainv){
	int k;
	double sum = 0.0, tmp;
	//double *Knm; Knm = Z + N*(P-M);
	int i = blockIdx.x;
	int j = blockIdx.y;
	if(j<M){if(i<N){ // Knm
		for(k=1; k<Q; k++){
			tmp = X[i+k*N]-Ta[j+k*M];
			sum += tmp*tmp/2.0/rho[k];
		}
		Knmta[i+j*N] = exp(-sum);
	}else if(i>=N && i<(N+M)){ // Kmm
		int ii = i-N;
		if(ii==j){
			Kta[ii+j*M] = 1.0;
		}else if(ii>j){
			Kta[ii+j*M] = 0.0;
		}else if(ii<j){
			for(k=1; k<Q; k++){
				tmp = Ta[ii+k*M]-Ta[j+k*M];
				sum += tmp*tmp/2.0/rho[k];
			}
			Kta[ii+j*M] = exp(-sum);
		}
	Ktainv[ii+j*M] = Kta[ii+j*M];
	}}
}

__global__ void getKnmtaKtainvh(int N, int M, double* Ktainv, double* Knmta, double* doxgen){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(i<N){
		forwardslg(Ktainv, M, Knmta+i, N);
		for(int k=0; k<M; k++){
			Knmta[i+k*N] *= doxgen[i];
		}
		Knmta[i+M*N] = doxgen[i];
	}
}




__device__ void getKernelMatsOnlyARDSE(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* K, double* Kinv, double* KinvKmn){
	int k;
	double sum = 0.0, tmp;
	//double *Knm; Knm = Z + N*(P-M);
	int i = blockIdx.x;
	int j = blockIdx.y;
	if(j<M){if(i<N){ // Knm
		//if(i==0&&j==0)printM_d(rho,1,Q,1,0);
		//if(i==0&&j==0)printM_d(Ta,10,Q,M,0);
		//if(i==0&&j==0)printM_d(X,10,Q,N,0);
		for(k=0; k<Q; k++){
			tmp = X[i+k*N]-Ta[j+k*M];
			sum += tmp*tmp/2.0/rho[k];
		}
		Z[N*(P-M) + i+j*N] = exp(-sum);
	}else if(i>=N && i<(N+M)){ // Kmm
		int ii = i-N;
	//if(j==(M-1)){printf("%d,",ii);}
		if(ii==j){
			K[ii+j*M] = 1.0;
		}else if(ii>j){
			K[ii+j*M] = 0.0;
		}else if(ii<j){
			for(k=0; k<Q; k++){
				tmp = Ta[ii+k*M]-Ta[j+k*M];
				sum += tmp*tmp/2.0/rho[k];
			}
			K[ii+j*M] = exp(-sum);
		}
	Kinv[ii+j*M] = K[ii+j*M];
	//if(j==(M-1)&&ii==(M-2)){
	//	printf("K[%d,%d]=%lf\n",ii,j, K[ii+j*N]);
	//}
	}}
}


