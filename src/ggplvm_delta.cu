#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "DataFrame.cuh"
#include "SparseMat.cuh"
//#include "dsyev.cuh"
#include "util.h"
#include "util_cuda.cuh"
#include "init.cuh"
#include "kernel_funcs.cuh"

#include "f2c.h"
#include "blaswrap.h"
#include "clapack.h"

double wt = 0.0; // weight for titsias penalty


// P
__global__ void getZeta(int J, int P, double* Beta, double* phi, double *zeta){
	int k = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(k<P){
		double sum = 0.0;
		double den = 0.0;
		int j;
		for(j=0; j<J; j++){
			sum += Beta[k+j*P]/phi[j];
			den += 1.0/phi[j];
		}
		zeta[k] = sum/den;
		//printf("%d:%lf\n",k,zeta[k]);
	}
}

// J x P
__global__ void solvePsi(int J, int P, double* IdenP, double* Psi){
	int j = blockIdx.x; // 1..J
	int k = blockIdx.y; // 1..P

	if(j<J && k<P){
		cholsl(Psi+j*P*P, P, IdenP+k*P, Psi+J*P*P+j*P*P+k*P);
	}
}

// M x M
__global__ void getC2(int J, DataFrame *Z, double* Psi, double* W, double* delta, double* tBeta, double* phi, double* K, double* Kinv, double* KinvKmn, double wt, double* Work, double* C2){
	int j = blockIdx.x;
	int k = blockIdx.y;
	if(j<=k){
		int N = Z->N;
		int P = Z->P;
		int p = Z->p;
		int *nh; nh = Z->nh;
		int i;
		int M = Z->M;
		double *Oinvt; Oinvt = W+N*J;
		double *ev, *ev2;
		//if(N<(j+k*M+1)){printf("memory leak!!!!!!!\n!"); return;}
		ev  = Work + (j+k*M)*2*P;
		ev2 = Work + (j+k*M)*2*P + P;
		double sum = 0.0;
		for(i=0; i<N; i++){// titsias
			sum += KinvKmn[j+i*M]*KinvKmn[k+i*M]*Oinvt[i];
		}
		C2[j+k*M] = - wt * delta[p-1]*sum/2.0;

		sum = 0.0;
		for(i=0; i<J; i++){
			sum += tBeta[j+nh[p-1]+i*P]*tBeta[k+nh[p-1]+i*P]/phi[i];
		}
		C2[j+k*M] -= sum/delta[p-1]/2.0;
		
		sum = 0.0;
		fill0_d(ev,P); ev[k+nh[p-1]] = 1.0;
		for(i=0; i<J; i++){
			//cholsl(Psi+i*P*P, P, ev, ev2);
			//sum += ev2[j+nh[p-1]];
			sum += Psi[J*P*P + i*P*P + (k+nh[p-1])*P + (j+nh[p-1])];
		}
		C2[j+k*M] -= sum/delta[p-1]/2.0;

		fill0_d(ev,M); ev[k] = 1.0;
		cholsl(Kinv, M, ev, ev2);
		C2[j+k*M] += ((double)J)*ev2[j]/2.0;
		
		C2[j+k*M] *= K[j+k*M];
		if(j<k) C2[k+j*M] = C2[j+k*M];
	}
}

// M x Q
__global__ void getGradTau(int Q, DataFrame *Z, double* Xi, double* Ta, double* rho, double* C1, double* C2, double* grad){
	int m = blockIdx.x;
	int q = blockIdx.y;
	int N = Z->N;
	int p = Z->p;
	int M = Z->M;
	double* gT; gT = grad + N + p + N*Q;
	int i;
	double sum, cm1;
	if(m<M){if(q==0){
		sum = 0.0; cm1 = 0.0;
		double tmp;
		for(i=0; i<N; i++){ // dKnm/dTmq
			tmp = (Xi[i] - Ta[m])/2.0;
			sum += C1[i+m*N] * sin(tmp)*cos(tmp); // minus
		}
		gT[m+q*M] = sum/rho[0];
		
		sum = 0.0; cm1 = 0.0;
		for(i=0; i<M; i++){ // dK/dTmq
			tmp = (Ta[i] - Ta[m])/2.0;
			sum += C2[m+i*M]* sin(tmp)*cos(tmp);
		}
		gT[m+q*M] += 2.0*sum/rho[0]; // x 2.0 for row and col wise derivative (not for Knm)
		//gT[m] = 0.0;
	}else if(q>0 && q<Q){
		sum = 0.0; cm1 = 0.0;
		for(i=0; i<N; i++){
			sum += C1[i+m*N]*Xi[i+q*N];
			cm1 += C1[i+m*N];
		}
		gT[m+q*M] = (sum - cm1*Ta[m+q*M])/rho[q];

		sum = 0.0; cm1 = 0.0;
		for(i=0; i<M; i++){
			sum += C2[m+i*M]*Ta[i+q*M];
			cm1 += C2[m+i*M];
		}
		gT[m+q*M] += 2.0*(sum - cm1*Ta[m+q*M])/rho[q];
		gT[m+q*M] -= Ta[m+q*M]; // normal prior
	}}

}

// Q
__global__ void collectGradLogRho(int Q, DataFrame *Z, double* gradLogRhoTmp, double* grad){
	int q = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	int N = Z->N;
        int p = Z->p;
        int M = Z->M;
	if(q<Q){
		double *gr; gr = grad + N + p + N*Q + M*Q;
		int k;
		double sum = 0.0;
		for(k=0; k<M; k++){
			sum += gradLogRhoTmp[q+k*Q];
		}
		gr[q] = sum;
	}
}

// M x Q
__global__ void getGradLogRho(int Q, DataFrame *Z, double* Xi, double* Ta, double* rho, double* C1, double* C2, double* gradLogRhoTmp){
	int j = blockIdx.x;// 1..M
	int q = blockIdx.y;// 1..Q
	int i;
	int N = Z->N;
        int M = Z->M;
	double sum = 0.0;
	double tmp;
	//double *gr; gr = grad + N + p + N*Q + M*Q;
	if(q==0){
		for(i=0; i<N; i++){
			//for(j=0; j<M; j++){
				tmp = sin((Xi[i+q*N]-Ta[j+q*M])/2.0);
				sum += C1[i+j*N]*tmp*tmp/rho[q]/rho[q];
			//}
		}
		for(i=0; i<M; i++){
			//for(j=0; j<M; j++){
				tmp = sin((Ta[i+q*M]-Ta[j+q*M])/2.0);
				sum += C2[i+j*M]*tmp*tmp/rho[q]/rho[q];
			//}
		}
		gradLogRhoTmp[q+j*Q] = sum*rho[q]; // log rho
	}else if(q>0 && q<Q){
		//if(q==0){printM_d(C1,10,M,N,0);printM_d(C2,10,M,M,0);}
		for(i=0; i<N; i++){
			//for(j=0; j<M; j++){
				tmp = Xi[i+q*N]-Ta[j+q*M];
				sum += C1[i+j*N]*tmp*tmp/rho[q]/rho[q]/2.0;
			//}
		}
		for(i=0; i<M; i++){
			//for(j=0; j<M; j++){
				tmp = Ta[i+q*M]-Ta[j+q*M];
				sum += C2[i+j*M]*tmp*tmp/rho[q]/rho[q]/2.0;
			//}
		}
		gradLogRhoTmp[q+j*Q] = sum*rho[q]; // log rho
	}

}

// gradient for Delta
// J x P
__global__ void getGradLogDelta(int J, DataFrame *Z, double *DeltaInv, double *delta, double* K, double *Psi, double *tBeta, double *phi, double* Work, double *gradDeltaTmp){
	int j = blockIdx.x; // 1..J
	int k = blockIdx.y; // 1..P
        int p = Z->p;
        int P = Z->P;
        int M = Z->M;
	int *nh; nh = Z->nh;
	if(j<J && k<P){
		//printf("%d %d %d\n", k, nh[k], nh[k+1]);
		int l;
		double dk, sum;
		double* ek;	ek   = Work + 2*(j*P+k)*P ; //(double*)malloc(sizeof(double)*P);
		double* psijk; psijk = Work + 2*(j*P+k)*P + P; //(double*)malloc(sizeof(double)*P);
		double *tbj;   tbj = &(tBeta[j*P]); // bj must be iteratively updated after Wj updated
		sum = 0.0;
		for(l=0; l<p; l++){
			if(nh[l] <= k && k < nh[l+1]){
				dk = delta[l];
				break;
			}
		}
		fill0_d(ek, P); 
		if(k<nh[p-1]){
			ek[k] = 1.0; // (0,...,0,1,0,...,0) kth element to be 1
		}else{ // k=nh[p-1] .. nh[p]-1
			for(l=0; l<(k-nh[p-1]); l++) ek[l+nh[p-1]] = K[l+(k-nh[p-1])*M];
			for(l=(k-nh[p-1]); l<M; l++) ek[l+nh[p-1]] = K[(k-nh[p-1])+l*M];
		}
		fill0_d(psijk, P);	 // kth vector of Psij
		cholsl(Psi+j*P*P, P, ek, psijk);
		sum -= (dk-psijk[k])/dk/dk/2.0;	// for eigen : // sum -= (dk-Psij[k+k*P])/dk/dk/2.0;
		if(k<nh[p-1]){
			sum += tbj[k]*tbj[k]/dk/dk/phi[j]/2.0;
		}else if(k==nh[p-1]){
			sum += quadform(K, tbj+nh[p-1], M)/dk/dk/phi[j]/2.0;
			//dinvtbjk = 0.0;
			//for(l=0; l<k; l++) dinvtbjk += DeltaInv[l+k*P]*tbj[l];
			//for(l=k; l<M; l++) dinvtbjk += DeltaInv[k+l*P]*tbj[l];
			//sum += dinvtbjk*tbj[k]/dk/phi[j]/2.0;
		}
		gradDeltaTmp[k+j*P] = sum*dk;
		//if(j==0){printf("%d:%lf\n",k,sum*dk);}
	}
}

// p
__global__ void collectGradLogDelta(int J, DataFrame *Z, double* delta, double *gradDeltaTmp, double* W, double* titsias, double wt, double *grad){

	int l = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y); // l = 0,...,p-1
	int p = Z->p;
	if(l<p){
		int i, j, k;
		int N = Z->N;
		int P = Z->P;
		int *nh; nh = Z->nh;
		double *gd; gd = grad; 
		double* Oinvt; Oinvt = W + N*J;
		double sum = 0.0;
		for(j=0; j<J; j++){
			for(k=nh[l]; k<nh[l+1]; k++){
				sum += gradDeltaTmp[k+j*P];
			}
		}
		// titsias
		if(l==(p-1)){
			for(i=0; i<N; i++){
				sum -= wt * Oinvt[i]* titsias[i] * delta[p-1]; // dk multiplication for log delta
			}
		}
		gd[l] = sum;
	}
}

// N
__global__ void getOinvt(int N, int J, double *Omega, double *W){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);// sample
	if(i<N){
		int j;
		double Oii = 0.0;
		for(j=0; j<J; j++){
			Oii += W[i+j*N]/Omega[i];
		}
		W[N*J+i] = Oii; // sum(Oijinv)
	}
}


// gradient for Omega
// later will be grad for X
// N
__global__ void getGradLogOmega(int J, int Q, DataFrame *Z, double *Omega, double *delta, double *phi, double *Nu, double *Psi, double *W, double *tBeta, 
		double* Xi, double* Ta, double* rho, double* KinvKmn, double* titsias, double wt, double* Work, double* C1, double *grad){
	int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);// sample
	int N = Z->N;
	if(i<N){
		double *gO, *gX;
                int p = Z->p;
                int P = Z->P;
                int *nh; nh = Z->nh;
		int M = Z->M;
		gO = grad;
		gX = grad + N + p;
		double sum = 0.0;
		double tmp;
		double *Knm;  Knm = Z->Knm;
		int j, k;
		double* zi;	 zi     = Work + i*2*P; //(double*)malloc(sizeof(double)*P);
		double* Psijzi; Psijzi = Work + i*2*P + P; //(double*)malloc(sizeof(double)*P);
		fill0_d(zi,P);
		for(k=0; k<P; k++){
			zi[k] = getModelMatrix(Z, i, k);
		}
		double* ci;	 ci = C1 + i; fill0_dg(ci, M, N);
		double Oii = 0.0;
		double *tbj;
		double Oijinv = 0.0;
		double tnij;
		for(j=0; j<J; j++){
			//Psij = Psi + j*P*P; //&(Psi[j*P*P]);
			//Wj   = W   + j*N; // &(W[j*N]);
			//nj   = Nu  + j*N; //&(Nu[j*N]);
			tbj  = tBeta + j*P; // &(tBeta[j*P]);
			Oijinv = W[i+j*N]/Omega[i];
			Oii += Oijinv;
			cholsl(Psi+j*P*P, P, zi, Psijzi);
			tnij = Oijinv*Nu[i+j*N] - Oijinv*nk_fdot(zi,1,tbj,1,P);
			sum -= (1.0/Omega[i] - nk_fdot(zi,1,Psijzi,1,P)/Omega[i]*Oijinv)/2.0; // diag( Wj^-1Vj^-1 )
			sum += tnij*tnij/W[i+j*N]/phi[j]/2.0; // tnj^2/Wj/phij/2.0
			//if(isnan(sum)>0){
			//	printf("sum[%d,%d]2 is nan: tnij=%lf Wji=%lf phij=%lf\n",i,j, tnij, Wj[i], phi[j]);
			//	return;
			//}
			for(k=0; k<M; k++){
				ci[k*N] += ( -Psijzi[k+nh[p-1]]*Oijinv + tnij*tbj[k+nh[p-1]]/phi[j] );
			}
		}
		//double sum2 = 0.0;
		//for(k=0; k<M; k++){// titsas for Omega
		//	sum2 += Knm[i+k*N]*KinvKmn[k+i*M];
		//}
		// titsias for omega
		sum += wt * delta[p-1]*Oii/Omega[i]*titsias[i];
		gO[i] = sum*Omega[i] + 30.0/Omega[i] - 30.0; // gradient for log(omega_i) with inv gamma prior * omegai

		// Oinvt
		W[N*J+i] = Oii; // sum(Oijinv)

		// creating C1
		for(k=0; k<M; k++){
			ci[k*N] += wt * delta[p-1]*Oii*KinvKmn[k+i*M]; // titsias
			ci[k*N] *= Knm[i+k*N]; // ci odot ki
			//gX[i+k*N] = ci[k*N] - X[i+k*N]; // linear kernel
		}

		double ci1 = 0.0; gX[i] = 0.0; 
		for(k=0; k<M; k++){// periodic
			ci1 += ci[k*N]; // for ard-se
			tmp = (Xi[i] - Ta[k])/2.0;
			gX[i] -= ci[k*N] * sin(tmp)*cos(tmp)/rho[0];
		}
		// ard-se
		for(int q=1; q<Q; q++){ //old code for Tau: // sum = 0.0; for(k=0; k<M; k++){sum += ci[k*N]*Ta[k+q*M];} // ci %*% T
			gX[i+q*N] = ( nk_fdot(ci, N, Ta+q*M, 1, M) - ci1*Xi[i+q*N] )/rho[q] - Xi[i+q*N]; // normal prior
		}
	}
}

__global__ void printParams(int N, int J, int p, int M, int Q, double* Omega, double* delta, double* rho, double* Xi, double* Ta, double* zeta, double* phi){
	printf("Omega:");for(int i=0; i<5; i++){printf("%lf, ", Omega[i]);};  printf("\n");
	printf("phi:  ");for(int i=0; i<5; i++){printf("%lf, ", phi[i]);};  printf("\n");
	printf("zeta: ");for(int i=0; i<5; i++){printf("%lf, ", zeta[i]);};  printf("\n");
	printf("delta:");for(int i=0; i<p; i++){printf("%lf, ", delta[i]);};   printf("\n");
	printf("rho:  ");for(int i=0; i<Q; i++){printf("%lf, ", rho[i]);};	 printf("\n");
	printf("Xi:\n"); printM_d(Xi,5,Q,N,0);  printf("\n");
	printf("Tau:\n");printM_d(Ta,5,Q,M,0);  printf("\n");
}

// J
__global__ void getBetaFromTildeBeta(int J, DataFrame *Z, double* tBeta, double *zeta, double *Beta){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int P = Z->P;
		for(int k=0; k<P; k++){
			Beta[k+j*P] = tBeta[k+j*P] + zeta[k];
		}
	}
}


// J
__global__ void getTildeBeta(int J, DataFrame *Z, double *ls, double *Psi, double *Omega, double* Nu, double* W, double* Work, double *tBeta){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int N = Z->N;
		int P = Z->P;
		getZtOinvjnj(Z, W+j*N, Omega, Nu+j*N, Work+j*P);
		cholsl(Psi+j*P*P, P, Work+j*P, tBeta+j*P);
	}
}

// J
__global__ void getBetaPhi(int J, DataFrame *Z, double *ls, double *Psi, double *Omega, double *zeta, double* Nu, double* W, double* Work, double *Beta, double *phi){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int N = Z->N;
		int P = Z->P;
		double phij = 0.0;
		int i, k;
		getZtOinvjnj(Z, W+j*N, Omega, Nu+j*N, Work+j*P);
		for(i=0; i<N; i++){
			phij += Nu[i+j*N]*Nu[i+j*N]*W[i+j*N]/Omega[i];
		}

		cholsl(Psi+j*P*P, P, Work+j*P, Beta+j*P);
	
		for(k=0; k<P; k++){
			phij	   -= Beta[k+j*P]*Work[k+j*P];
			Beta[k+j*P] += zeta[k];
		}
		phi[j] = phij/((double)N);

		if(isnan(phi[j])>0){printf("phi[%d] is nan!\n", j); phi[j]=1.0;}
		if(phi[j]<0.0){ printf("phi[%d]=%lf is negative!\n", j, phi[j]);phi[j]=100.0;}
		if(phi[j]<0.01){ printf("phi[%d]=%lf is too small!\n", j, phi[j]);phi[j]=0.01;}
		if(phi[j]>100.0){printf("phi[%d]=%lf is too large!\n", j, phi[j]);phi[j]=100.0;}
	}
}

// J
__global__ void getWForBeta(SparseMat *Y, DataFrame *Z, double *ls, double *Beta, double *zeta, double* delta, double* titsias, double* W, double* Nu){
	int J = Y->J;
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int N = Y->N;
		int P = Z->P;
		int p = Z->p;
		int* nh; nh = Z->nh;
		double *bj; bj  = &(Beta[j*P]);
		double *Wj; Wj  = &(W[j*N]);
		double *nj; nj  = &(Nu[j*N]);

		int* pi;
		double* pd;

		int i, k, m;
		int l = Y->p[j];
		int lmax = Y->p[j+1];
		double sum, sum2;
		for(i=0; i<N; i++){
			sum = 0.0;
			sum2= 0.0;
			for(k=0; k<p; k++){
				if(Z->c[k]==1){// numerical
					pd = (double*)(Z->x + Z->b[k]);
					for(m=0; m<Z->v[k]; m++){
						sum  += pd[i+m*N] *  bj[ nh[k] + m ];
						sum2 += pd[i+m*N] * (bj[ nh[k] + m ] - zeta[ nh[k] + m ]);
					}
				}else{// categorical
					pi = (int*)(Z->x + Z->b[k]);
					sum  +=  bj[ nh[k] + pi[i] ];
					sum2 += (bj[ nh[k] + pi[i] ] - zeta[ nh[k] + pi[i] ]);
				}
			}
			Wj[i] = exp(sum)*ls[i]*exp(delta[p-1]*titsias[i]);
			if(i==Y->i[l] && l<lmax){
				nj[i] = sum2 + (Y->x[l]-Wj[i])/Wj[i];
				l++;
			}else{
				nj[i] = sum2 - 1.0;
			}
		}
	}
}



// J
__global__ void getW(SparseMat *Y, DataFrame *Z, double *ls, double *Beta, double *zeta, double* delta, double* titsias, double* W, double* Nu){
	int J = Y->J;
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int N = Y->N;
		int P = Z->P;
		int p = Z->p;
		int* nh; nh = Z->nh;
		double *bj; bj  = &(Beta[j*P]);
		double *Wj; Wj  = &(W[j*N]);
		double *nj; nj  = &(Nu[j*N]);

		int* pi;
		double* pd;

		int i, k, m;
		int l = Y->p[j];
		int lmax = Y->p[j+1];
		double sum, sum2, mui;
		for(i=0; i<N; i++){
			sum = 0.0;
			sum2= 0.0;
			for(k=0; k<p; k++){
				if(Z->c[k]==1){// numerical
					pd = (double*)(Z->x + Z->b[k]);
					for(m=0; m<Z->v[k]; m++){
						sum  += pd[i+m*N] *  bj[ nh[k] + m ];
						sum2 += pd[i+m*N] * (bj[ nh[k] + m ] - zeta[ nh[k] + m ]);
					}
				}else{// categorical
					pi = (int*)(Z->x + Z->b[k]);
					sum  +=  bj[ nh[k] + pi[i] ];
					sum2 += (bj[ nh[k] + pi[i] ] - zeta[ nh[k] + pi[i] ]);
				}
			}
			mui   = exp(sum)*ls[i];
			Wj[i] = mui*exp(delta[p-1]*titsias[i]);
			if(i==Y->i[l] && l<lmax){
				nj[i] = sum2 + (Y->x[l]-mui)/mui;
				l++;
			}else{
				nj[i] = sum2 - 1.0;
			}
		}
	}
}


// J x P x P
__global__ void getPsi(int J, DataFrame *Z, double *Omega, double *DeltaInv, double* W, double* Psi){
	int j = blockIdx.x;
	int k = blockIdx.y;
	int l = threadIdx.y;
	int N = Z->N;
	int P = Z->P;
	double sum=0.0;
	if(k<=l && l<P){
            if(Z->mc[k]==1){ // num 1
                double* pd1; pd1 = (double*)(Z->x + Z->mb[k]);
                if(Z->mc[l]==1){// num 2
                    double* pd2; pd2 = (double*)(Z->x + Z->mb[l]);
                    for(int i=0; i<Z->N; i++){
                        sum += pd1[i]*pd2[i]*W[i+j*N]/Omega[i];
                    }
                }else{// cate 2
                    int* pi2; pi2 = (int*)(Z->x + Z->mb[l]);
                    int mvl = Z->mv[l];
                    for(int i=0; i<Z->N; i++){
                        if(pi2[i]==mvl){
                            sum += pd1[i]*W[i+j*N]/Omega[i];
                        }
                    }
                }
            }else{ // cate 1
                int* pi1; pi1 = (int*)(Z->x + Z->mb[k]);
                int mvk = Z->mv[k];
                if(Z->mc[l]==1){ // num 2
                    double* pd2; pd2 = (double*)(Z->x + Z->mb[l]);
                    for(int i=0; i<Z->N; i++){
                        if(pi1[i]==mvk){
                            sum += pd2[i]*W[i+j*N]/Omega[i];
                        }
                    }
                }else{// cate 2
                    int* pi2; pi2 = (int*)(Z->x + Z->mb[l]);
                    int mvl = Z->mv[l];
                    if(Z->mk[k]==Z->mk[l]){// same var
                        if(k==l){
                            for(int i=0; i<Z->N; i++){
                                if(pi1[i]==mvk){
                                    sum += W[i+j*N]/Omega[i];
                                }
                            }
                        }
                    }else{ // diff vars
                        for(int i=0; i<Z->N; i++){
                            if(pi1[i]==mvk && pi2[i]==mvl){
                                sum += W[i+j*N]/Omega[i];
                            }
                        }
                    }
                }
            }
            Psi[j*P*P + k + l*P] = sum + DeltaInv[k + l*P];
        }else{
            Psi[j*P*P + k + l*P] = 0.0;
        }
}

// J
__global__ void cholPsi(int J, int P, double* Psi){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		//if(j==1577){printM_d(Psi+j*P*P,P,P,P,0);}
		// cholesky
		choldc(Psi+j*P*P, P);
		// eigen
		//choldc(Psi+j*P*P, P, Psi+J*P*P+j*P*4);
	}
}

// 1
__global__ void cholDeltaInv(int P, double* DeltaInv){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j==0){
		choldc(DeltaInv, P);
	}
}

// J
__global__ void getLkhdPois(SparseMat* Y, DataFrame* Z, double* W, double* delta, double* titsias, double *lkhds){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	int J = Y->J;
	if(j<J){
		int p = Z->p;
		int N = Y->N;
		double *Wj; Wj  = &(W[j*N]);
		int i, l, lmax;
		double sum = 0.0;
		l = Y->p[j];
		lmax = Y->p[j+1];
		for(i=0; i<N; i++){
			if(Y->i[l]==i && l<lmax){
				sum += (log(Wj[i])-delta[p-1]*titsias[i]) * Y->x[l] - Wj[i];
				l++;
			}else{
				sum -= Wj[i];
			}
		}
		lkhds[j] = sum;
	}
}

__device__ double getLkhdj(DataFrame *Z, double *Psij, double *Wj, double *Omega, double *nj, double *DeltaInv, double phij, double* tmp, double* tmp2){
	int i, k;
	double Oijinv;
	int N = Z->N;
	int P = Z->P;
	double lkhdj = - ((double)N)/2.0*log(phij);
	for(i=0; i<N; i++){
		lkhdj -= log(Omega[i]/Wj[i])/2.0;
	}
	for(k=0; k<P; k++){
		lkhdj -= log(Psij[k+k*P]); // Cholesky Decomp!!!!
		lkhdj += log(DeltaInv[k+k*P]);
	}
	fill0_d(tmp,  P);
	fill0_d(tmp2, P); // tilde bj actually
	getZtOinvjnj(Z, Wj, Omega, nj, tmp);
	for(i=0; i<N; i++){
		Oijinv = Wj[i]/Omega[i];
		lkhdj -= nj[i]*nj[i]*Oijinv/2.0/phij;
	}
	cholsl(Psij, P, tmp, tmp2);
	for(k=0; k<P; k++){
		lkhdj += tmp[k]*tmp2[k]/2.0/phij;
		if(isnan(lkhdj)>0){break;}
	}
	return lkhdj;
}


// J
__global__ void getLkhd(int J, DataFrame *Z, double *Omega, double *DeltaInv, double *phi, double* Nu, double* Psi, double* W, double* Work, double *lkhds){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j<J){
		int N = Z->N;
		int P = Z->P;
		double *nj; nj  = &(Nu[j*N]);
		double *Psij; Psij  = &(Psi[j*P*P]);
		double *Wj; Wj  = &(W[j*N]);
		lkhds[j] = getLkhdj(Z, Psij, Wj, Omega, nj, DeltaInv, phi[j], Work+2*j*P, Work+2*j*P+P);
	}
}


// 1
__global__ void collectLkhd(int N, int J, int Q, double* Omega, double* lkhds, int itr, int jtr, DataFrame *Z, double* delta, double* Xi, double* Ta, double* W, double* titsias, double wt, double* lkhd_d){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j==0){ // call after getGradLogOmega
		int i, k;
		double sum;
		//int N = Z->N;
		int p = Z->p;
		int M = Z->M;
		double *Oinvt; Oinvt = W + N*J;
		sum = 0.0;
		for(k=0; k<J; k++){
			sum += lkhds[k];
		}
		// normal prior on Xi and Ta
		for(i=0; i<N; i++){for(k=1; k<Q; k++){sum -= Xi[i+k*N]*Xi[i+k*N]/2.0;}}
		for(i=0; i<M; i++){for(k=1; k<Q; k++){sum -= Ta[i+k*M]*Ta[i+k*M]/2.0;}}
		// titsias
		for(i=0; i<N; i++){
			sum -= wt * delta[p-1]*Oinvt[i]*titsias[i];
		}
		//inv gamma prior for Omega
		for(i=0; i<N; i++){
			sum = sum - 30.0/Omega[i] - 30.0*log(Omega[i]);
		}
		printf("lkhd[%d,%d]=%lf\n", itr, jtr, sum);
		lkhd_d[jtr] = sum;
	}
}

// 1
__global__ void Downdate(int itr, int Q, DataFrame *Z, double* Omega, double* delta, double* K, double* Xi, double* Ta, double* rho, double* grad, double* grad0, double* SS, double* XX, double* ss, double r, int rhoflag){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j==0){
		int i, k;
		int N = Z->N;
                int M = Z->M;
                int p = Z->p;
		int tot = 1; //p; 
		int ptp1 = (itr) % 10;
		//logscale(Omega, N);
		printf("delta[%d]=", itr);
		for(k=p-2; k<p-1; k++){
			delta[k] = exp(log(delta[k+p]) - r*ss[0]);
			printf("%lf, ", delta[k]);
		}
		printf("\n");

		for(k=0; k<tot; k++){
			SS[k+ptp1*tot] = r*ss[k];
		}
	}
}
//SS : step size
//SS : grad diff
//ss : step size at itr
// do not reapeat (backup will be overwritten!)
// 1
__global__ void LBFGS(int itr, int Q, DataFrame *Z, double* Omega, double* delta, double* K, double* Xi, double* Ta, double* rho, double* grad, double* grad0, double* SS, double* XX, double* ss, double* Work, int rhoflag){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	if(j==0){
		int N = Z->N;
		int M = Z->M;
		int p = Z->p;
		int i, k;
		int pt   = (itr-1) % 10;
		int ptp1 = (itr) % 10;
		int tot = 1 ; //p;
		if(itr==0){
			for(k=0; k<tot; k++){
				//ss[k] = -grad[k]/10000.0;
printf("grad[%d]=%lf\n", p,grad[p-2]);
				ss[k] = -grad[p-2]/10000.0;
				if(ss[k]>1.0){ss[k]=0.01;}
				if(ss[k]<(-1.0)){ss[k]=-0.01;}
				grad0[p-2] = grad[p-2];
				//if(k<10){ printf("%lf ", ss[k]);}
				if(k>=0 && k<(p)){ printf("ss_delta=%lf ", ss[k]);}
			}
			printf("\n");
		}else{
			for(k=0; k<tot; k++){
				XX[k+pt*tot] = grad[p-2] - grad0[p-2];
				grad0[p-2] = grad[p-2];
			}
			lbfgs(SS, XX, grad, tot, 10, itr, Work, ss);
			printf("ss_delta_lbfgs=");printf("%lf \n", ss[0]);
		}
		//printM_d(SS,10,10,N+p,0);
		//printM_d(XX,10,10,N+p,0);
		//printM_d(ss,1,10,1,0);

		// delta update
		printf("delta[%d]=", itr);
		for(k=p-2; k<p-1; k++){
			delta[k+p] = delta[k]; // backup
			delta[k] = exp(log(delta[k]) - ss[0]); printf("%lf, ", delta[k]);
		}
		printf("\n");
		

		printf("\n");
		for(k=0; k<tot; k++){
			SS[k+ptp1*tot] = ss[k];
		}
	}
}

int main(int argc, char** argv){

	int ncudadev;
	cudaGetDeviceCount(&ncudadev);
	printf("n cuda devs=%d\n", ncudadev);



	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	printf("Heap Size=%ld\n", size);
	
	int verbose=0;
	if(verbose>0) printf("%s %s %s %s\n", argv[0], argv[1], argv[2], argv[3]);

	SparseMat *Y, *Y_h; Y_h=loadSparseMat(argv[1]);
	int N = Y_h->N;
	int J = Y_h->J;
	Y = newSparseMatOnDevice(Y_h);

	int M=50;
	int Q=16;

	M=100;
	Q=51;
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-M")==0){M=(int)atoi(argv[i+1]);break;}}
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-Q")==0){Q=(int)atoi(argv[i+1]);break;}}
	double initDeltaG = 0.01;
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"--init-deltaG")==0){initDeltaG=(double)atof(argv[i+1]);break;}}

	DataFrame *Z_h, *Z;
	Z_h = loadDataFrame(argv[2], M); //printMetaDataFrame(Z_h);
	Z = newDataFrameOnDevice(Z_h); // Knm fixed
	int P = Z_h->P;
	int p = Z_h->p;

		
	printf("N=%d, J=%d, P=%d, p=%d, M=%d, Q=%d\n", N, J, P, p, M, Q);

	printMetaDataFrame(Z_h);

	int ITRMAX=100, JTRMAX=1000, itr=0, jtr=0;

	double *Omega, *Omega_h;
	double *DeltaInv, *delta, *delta_h;
	double *zeta;
	double *ls_h, *ls;
	double *phi;
	double *Psi;
	double *W;
	double *Nu;
	double *Beta, *tBeta;
	double *lkhds, *lkhd_h, *lkhd_d;
	double *grad, *grad0, *SS, *XX, *ss, *gradDeltaTmp, *gradLogRhoTmp, *Work;

	Omega_h  = (double*)malloc(sizeof(double)*N); // init with Z
	delta_h  = (double*)malloc(sizeof(double)*p); // init with Z

	double *rho, *Xi, *Ta, *K, *Kinv, *KinvKmn, *C1, *C2, *IdenP, *titsias;
	double *Xi_h;  Xi_h  = (double*)malloc(sizeof(double)*N*Q); // init with Z
	double *Ta_h;  Ta_h  = (double*)malloc(sizeof(double)*M*Q); initTaWithPer(M, Q, Ta_h); 
	double *rho_h;  rho_h  = (double*)malloc(sizeof(double)*Q); // for save purpos 
	double *phi_h;  phi_h  = (double*)malloc(sizeof(double)*J); // for save purpos 
	FILE* ft; ft = fopen("Tau.bin","wb"); fwrite(Ta_h, sizeof(double), M*Q, ft); fclose(ft);
	//double *rho_h; rho_h = (double*)malloc(sizeof(double)*Q); int k; for(k=0; k<Q; k++){rho_h[k]=(7.0-(double)k)*3;}

	ls_h	= (double *)malloc(sizeof(double)*N); if(ls_h==NULL){fprintf(stderr, "ls_h not allocated...\n"); return 1;}
	lkhd_h  = (double *)malloc(sizeof(double)*JTRMAX);
	FILE *fls; fls = fopen(argv[3], "rb");
	FILE *fXi; fXi = fopen(argv[4], "rb");
	fread(Xi_h,  sizeof(double), N*Q, fXi);
	fread(ls_h,  sizeof(double), N,   fls);

	if(cudaMallocManaged((void **) &Psi,	sizeof(double)*J*P*P*2)  == cudaErrorMemoryAllocation){printf("Psi not allocated.\n"); return 1;}; // chol
	if(cudaMallocManaged((void **) &W,	sizeof(double)*N*(J+1))  == cudaErrorMemoryAllocation){printf("W not allocated.\n"); return 1;};
	if(cudaMallocManaged((void **) &Nu,   sizeof(double)*N*J)  == cudaErrorMemoryAllocation){printf("Nu not allocated.\n"); return 1;}; // pseudo data
	if(cudaMallocManaged((void **) &grad,  sizeof(double)*(N+p+(N+M+1)*Q))	== cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &grad0, sizeof(double)*(N+p+(N+M+1)*Q))	== cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &SS,	sizeof(double)*(N+p+(N+M+1)*Q)*10) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &XX,	sizeof(double)*(N+p+(N+M+1)*Q)*10) == cudaErrorMemoryAllocation){return 1;}; 
	if(cudaMallocManaged((void **) &ss,	sizeof(double)*(N+p+(N+M+1)*Q))    == cudaErrorMemoryAllocation){return 1;};
	int lwork = (N<M*M ? M*M : N)*(2*P);
        if(N<J*P){lwork = (J*P<M*M ? M*M : J*P)*(2*P);}
	if(cudaMallocManaged((void **) &Work,  sizeof(double)*lwork)   == cudaErrorMemoryAllocation){printf("Work not allocated.\n"); return 1;};;
	if(cudaMallocManaged((void **) &C1,  sizeof(double)*N*M)   == cudaErrorMemoryAllocation){printf("Work not allocated.\n"); return 1;};;
	if(cudaMallocManaged((void **) &C2,  sizeof(double)*M*M)   == cudaErrorMemoryAllocation){printf("Work not allocated.\n"); return 1;};;
	if(cudaMallocManaged((void **) &IdenP,  sizeof(double)*P*P)   == cudaErrorMemoryAllocation){printf("Work not allocated.\n"); return 1;};;
	if(cudaMallocManaged((void **) &ls,	 sizeof(double)*N) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &Omega,  sizeof(double)*N*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &DeltaInv,  sizeof(double)*P*P) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &delta,  sizeof(double)*p*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &phi,	sizeof(double)*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &gradDeltaTmp, sizeof(double)*P*J)   == cudaErrorMemoryAllocation){return 1;};;
	if(cudaMallocManaged((void **) &gradLogRhoTmp, sizeof(double)*M*Q)   == cudaErrorMemoryAllocation){return 1;};;
	
	if(cudaMallocManaged((void **) &zeta, sizeof(double)*P) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &Beta, sizeof(double)*P*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &tBeta, sizeof(double)*P*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &lkhds, sizeof(double)*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &lkhd_d, sizeof(double)*JTRMAX) == cudaErrorMemoryAllocation){return 1;};

	if(cudaMallocManaged((void **) &rho, sizeof(double)*Q*2)   == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &Xi,  sizeof(double)*N*Q*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &Ta,  sizeof(double)*M*Q*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &K, sizeof(double)*M*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &Kinv, sizeof(double)*M*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &KinvKmn, sizeof(double)*N*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMallocManaged((void **) &titsias, sizeof(double)*N) == cudaErrorMemoryAllocation){return 1;};

	cudaMemcpy(ls,	ls_h,    sizeof(double)*N,	 cudaMemcpyHostToDevice);
	cudaMemcpy(Xi,	Xi_h,    sizeof(double)*N*Q,       cudaMemcpyHostToDevice);
	cudaMemcpy(Ta,	Ta_h,    sizeof(double)*M*Q,       cudaMemcpyHostToDevice);

	printf("Memory allocated...\n");

	dim3 gridJ(J,1); // J
	dim3 gridJJ(J,J); // JxJ
	dim3 gridJP(J,P); // JxP

	dim3 gridPsi(J,P); // J*P*P
	dim3 blockPsi(1,P);

	dim3 gridPP(P,P); // P*P
	
	dim3 gridP(P,1); // P

	int sN = (int)(sqrt((double)N)/4)+1;
	dim3 gridN(sN,sN); // N
	dim3 blockN(4,4);
	printf("N=%d sN^2*16=%d\n", N, sN*sN*16);
	//dim3 gridN(100,50); // N
	//dim3 blockN(2,2);
	
	dim3 gridDel(p,1); // p

	dim3 block1(1,1);

	dim3 gridNQ(N,Q);
	dim3 gridNpMM(N+M,M);

	dim3 gridMM(M,M);
	dim3 gridMQ(M,Q);
	dim3 gridQ(Q,1);

	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
	float milliseconds = 0;

	printf("N=%d, M=%d, Q=%d, p=%d\n", N, M, Q, Z_h->p);
	initParams<<<1,1>>>(Y_h->N, Y_h->J, Z_h->P, Z_h->p, Q, delta, Omega, phi, rho, IdenP);cudaDeviceSynchronize();
	//double *rho, *Xi, *Ta1, *Ta, *pT, *K, *Kinv, *KinvKmn, *titsias;
        //double *Xi_h;  Xi_h   = (double*)malloc(sizeof(double)*N*Q); // init with Z
        //double *Ta1_h; Ta1_h   = (double*)malloc(sizeof(double)*Mfull*Q);
        //double *Ta_h; Ta_h  = (double*)malloc(sizeof(double)*Mredc*Q);
        //double *rho_h;  rho_h  = (double*)malloc(sizeof(double)*Q); // for save purpos

        FILE *fpa; fpa = fopen("params.bin", "rb");
        fread(Omega_h, sizeof(double), N, fpa);
        fread(delta_h, sizeof(double), (Z_h->p)-1, fpa);delta_h[(Z_h->p)-1]=delta_h[(Z_h->p)-2];delta_h[(Z_h->p)-2]=initDeltaG;
        fread(Xi_h, sizeof(double), N*Q, fpa);
        fread(Ta_h, sizeof(double), M*Q, fpa);
        fread(rho_h, sizeof(double), Q, fpa);
        fclose(fpa);
	cudaMemcpy(Omega, Omega_h, sizeof(double)*N, cudaMemcpyHostToDevice);
        cudaMemcpy(delta, delta_h, sizeof(double)*((Z_h->p)-1), cudaMemcpyHostToDevice);
        cudaMemcpy(Xi, Xi_h, sizeof(double)*N*Q, cudaMemcpyHostToDevice);
        cudaMemcpy(Ta, Ta_h, sizeof(double)*M*Q, cudaMemcpyHostToDevice);
        cudaMemcpy(rho, rho_h, sizeof(double)*Q, cudaMemcpyHostToDevice);	
	// start with X
	cudaEventRecord(start);
	//printf("init Kernel mats...\n");
	getKernelMats<<<gridNpMM,block1>>>(Q, Xi, Ta, rho, Z, K, Kinv, KinvKmn);cudaDeviceSynchronize();
	cholKmm<<<1,1>>>(M, Kinv);cudaDeviceSynchronize();
	getKinvKmn<<<gridN,blockN>>>(Kinv, Z, KinvKmn, titsias);cudaDeviceSynchronize();
	
	expandDelta_k<<<1,1>>>(Z, delta, K, DeltaInv);cudaDeviceSynchronize();
	initPsi<<<gridPP, block1>>>(Z, DeltaInv, Psi);cudaDeviceSynchronize();
	initPsi2<<<1,1>>>(P, Psi);cudaDeviceSynchronize();
	initBeta<<<gridJ, block1>>>(Y, ls, Z, Psi, Work, Beta);cudaDeviceSynchronize(); 
	cudaEventRecord(stop);cudaEventSynchronize(stop);cudaEventElapsedTime(&milliseconds, start, stop);
	printf("initBeta: %f\n", milliseconds);
	getZeta<<<gridP, block1>>>(J, P, Beta, phi, zeta);cudaDeviceSynchronize();
	
	int rhoflag = 0;
	printf("iteration start...\n");
	for(itr=0; itr<ITRMAX; itr++){
		printf("###################\n");
		printf("#	  ITR=%d      \n", itr);
		printf("###################\n");
		printf("1.\n");
		//printParams<<<1,1>>>(N, J, p, M, Q, Omega, delta, rho, Xi, Ta, zeta, phi);	
		if(itr==0){JTRMAX=30;}else{JTRMAX=30;}
		for(jtr=0; jtr<JTRMAX; jtr++){

			cudaEventRecord(start);
			getWForBeta<<<gridJ, block1>>>(Y, Z, ls, Beta, zeta, delta, titsias, W, Nu);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("W Nu: %f\n", milliseconds);
			
			cudaEventRecord(start);
			getPsi<<<gridPsi, blockPsi>>>(J, Z, Omega, DeltaInv, W, Psi);cudaDeviceSynchronize();
			cholPsi<<<gridJ, block1>>>(J, P, Psi);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Psi: %f\n", milliseconds);
			
			cudaEventRecord(start);
			getBetaPhi<<<gridJ, block1>>>(J, Z, ls, Psi, Omega, zeta, Nu, W, Work, Beta, phi);cudaDeviceSynchronize();
			getZeta<<<gridP, block1>>>(J, P, Beta, phi, zeta);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("get Beta Phi Zeta: %f\n", milliseconds);
			
			cudaEventRecord(start);
			getLkhdPois<<<gridJ, block1>>>(Y, Z, W, delta, titsias, lkhds);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("getLkhd: %f\np", milliseconds);
			collectLkhd<<<1,1>>>(0, Y_h->J, 0, Omega, lkhds, itr, jtr, Z, delta, Xi, Ta, W, titsias, wt, lkhd_d);cudaDeviceSynchronize();
			cudaMemcpy(lkhd_h, lkhd_d, sizeof(double)*(jtr+1), cudaMemcpyDeviceToHost);
			if(jtr>0){
				if(jtr==(JTRMAX-1) || fabs(lkhd_h[jtr-1]-lkhd_h[jtr])/fabs(lkhd_h[jtr])<1e-5){
					//lkhd_h[0] = lkhd_h[jtr];
					//cudaMemcpy(lkhd_d, lkhd_h, sizeof(double), cudaMemcpyHostToDevice);
					break;
				}
			}	
		}

		printf("2.\n");
		cudaEventRecord(start);
		
		//getW<<<gridJ, block1>>>(Y, Z, ls, Beta, zeta, wt, titsias, W, Nu);cudaDeviceSynchronize();
		//getBetaPhi<<<gridJ, block1>>>(J, Z, ls, Psi, Omega, zeta, Nu, W, Work, Beta, phi);cudaDeviceSynchronize();
		//getZeta<<<gridP, block1>>>(J, P, Beta, phi, zeta);cudaDeviceSynchronize();

		getOinvt<<<gridN, blockN>>>(Y_h->N, Y_h->J, Omega, W);cudaDeviceSynchronize();
		cholDeltaInv<<<1,1>>>(Z_h->P, DeltaInv);cudaDeviceSynchronize();
		getLkhd<<<gridJ, block1>>>(Y_h->J, Z, Omega, DeltaInv, phi, Nu, Psi, W, Work, lkhds);cudaDeviceSynchronize();
		cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
		printf("getLkhd: %f\n", milliseconds);
		collectLkhd<<<1,1>>>(Y_h->N, Y_h->J, Q, Omega, lkhds, itr, 0, Z, delta, Xi, Ta, W, titsias, wt, lkhd_d);cudaDeviceSynchronize();
		if(itr>10){rhoflag=1;}
		for(jtr=0; jtr<100; jtr++){
			expandDelta_k<<<1,1>>>(Z, delta, K, DeltaInv);cudaDeviceSynchronize();
			getOinvt<<<gridN, blockN>>>(Y_h->N, Y_h->J, Omega, W);cudaDeviceSynchronize();

			printf("\njtr=%d\n", jtr);
			cudaEventRecord(start);
			getTildeBeta<<<gridJ, block1>>>(Y_h->J, Z, ls, Psi, Omega, Nu, W, Work, tBeta);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("getTildeBeta: %f\n", milliseconds);

			cudaEventRecord(start);	
			getGradLogDelta<<<gridJP, block1>>>(Y_h->J, Z, DeltaInv, delta, K, Psi, tBeta, phi, Work, gradDeltaTmp);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			collectGradLogDelta<<<gridDel, block1>>>(Y_h->J, Z, delta, gradDeltaTmp, W, titsias, wt, grad);cudaDeviceSynchronize();
			printf("getGradDel: %f\n", milliseconds);
		
			cudaEventRecord(start);
			LBFGS<<<1,1>>>(jtr, Q, Z, Omega, delta, K, Xi, Ta, rho, grad, grad0, SS, XX, ss, Work, rhoflag);
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Update: %f\n", milliseconds);

			int ktr;
			double sr=1.0;
			for(ktr=1; ktr<8; ktr++){
				getOinvt<<<gridN, blockN>>>(Y_h->N, Y_h->J, Omega, W);cudaDeviceSynchronize();

				getKernelMats<<<gridNpMM,block1>>>(Q, Xi, Ta, rho, Z, K, Kinv, KinvKmn);cudaDeviceSynchronize();
				expandDelta_k<<<1,1>>>(Z, delta, K, DeltaInv);cudaDeviceSynchronize();
				cholKmm<<<1,1>>>(Z_h->M, Kinv);cudaDeviceSynchronize();
				getKinvKmn<<<gridN,blockN>>>(Kinv, Z, KinvKmn, titsias);cudaDeviceSynchronize();
				
				cudaEventRecord(start);
				getPsi<<<gridPsi, blockPsi>>>(Y_h->J, Z, Omega, DeltaInv, W, Psi);cudaDeviceSynchronize();
				cholPsi<<<gridJ, block1>>>(Y_h->J, Z_h->P, Psi);cudaDeviceSynchronize();
				cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
				printf("getPsi: %f\n", milliseconds);

				cudaEventRecord(start);
				cholDeltaInv<<<1,1>>>(Z_h->P, DeltaInv);
				getLkhd<<<gridJ, block1>>>(Y_h->J, Z, Omega, DeltaInv, phi, Nu, Psi, W, Work, lkhds);
				cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
				printf("getLkhd: %f\n", milliseconds);
				collectLkhd<<<1,1>>>(Y_h->N, Y_h->J, Q, Omega, lkhds, itr, jtr+1, Z, delta, Xi, Ta, W, titsias, wt, lkhd_d);cudaDeviceSynchronize();
				cudaMemcpy(lkhd_h, lkhd_d, sizeof(double)*(jtr+2), cudaMemcpyDeviceToHost);

				if(lkhd_h[jtr+1] > lkhd_h[jtr] || ktr==7){
					break;
				}else if(isnan(lkhd_h[jtr+1])||isinf(lkhd_h[jtr+1])){
					printf("likelihood becomes nan/inf...donwdating parameters\n");
					sr *= 0.01;
				}else if(lkhd_h[jtr+1] < lkhd_h[jtr]){
					printf("likelihood becomes lower...donwdating parameters\n");
					sr *= 0.1;
				}
				Downdate<<<1,1>>>(jtr, Q, Z, Omega, delta, K, Xi, Ta, rho, grad, grad0, SS, XX, ss, sr, rhoflag);cudaDeviceSynchronize();
			}

			if(jtr>0 && fabs(lkhd_h[jtr+1]-lkhd_h[jtr])<fabs(lkhd_h[jtr+1])*1e-8){
				printf("likelihood becomes converged\n");
				break;	
			}

		}
		//return 0;

		cudaEventRecord(start);
		expandDelta_k<<<1,1>>>(Z, delta, K, DeltaInv);cudaDeviceSynchronize();
		/*initPsi<<<gridPP, block1>>>(Z, DeltaInv, Psi);cudaDeviceSynchronize();
		initPsi2<<<1,1>>>(Z_h->P, Psi);cudaDeviceSynchronize();
		initBeta<<<gridJ, block1>>>(Y, ls, Z, Psi, Work, Beta);cudaDeviceSynchronize(); 
		getZeta<<<gridP, block1>>>(Y_h->J, Z_h->P, Beta, phi, zeta);cudaDeviceSynchronize();*/
		getBetaFromTildeBeta<<<gridJ,1>>>(Y_h->J, Z, tBeta, zeta, Beta);cudaDeviceSynchronize();
		getZeta<<<gridP, block1>>>(Y_h->J, Z_h->P, Beta, phi, zeta);cudaDeviceSynchronize();
		cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
		printf("revive Beta: %f\n", milliseconds);
	}	
	printf("Iteration end...\n");

	cudaMemcpy(Omega_h, Omega, sizeof(double)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(delta_h, delta, sizeof(double)*p, cudaMemcpyDeviceToHost);
	cudaMemcpy(Xi_h, Xi, sizeof(double)*N*Q, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ta_h, Ta, sizeof(double)*M*Q, cudaMemcpyDeviceToHost);
	cudaMemcpy(rho_h, rho, sizeof(double)*Q, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_h, phi, sizeof(double)*J, cudaMemcpyDeviceToHost);
	/*FILE *fp; fp = fopen("params.bin", "wb");
	fwrite(Omega_h, sizeof(double), N, fp);
	fwrite(delta_h, sizeof(double), p, fp);
	fwrite(Xi_h, sizeof(double), N*Q, fp);
	fwrite(Ta_h, sizeof(double), M*Q, fp);
	fwrite(rho_h, sizeof(double), Q, fp);
	fwrite(phi_h, sizeof(double), J, fp);
	fclose(fp);*/

	free(Y_h); free(Z_h); free(ls_h); free(lkhd_h); free(Xi_h);
	cudaFree(Y); cudaFree(Z); cudaFree(Omega); cudaFree(DeltaInv); cudaFree(delta); cudaFree(ls); cudaFree(phi); 
	cudaFree(Psi); cudaFree(W); cudaFree(Nu); cudaFree(zeta); cudaFree(lkhds); cudaFree(Beta); cudaFree(tBeta); 
	cudaFree(ss); cudaFree(SS); cudaFree(XX);

	return 0;
}







