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


//1
__global__ void getDesignMatrix(DataFrame *Z, double* dZ){
	for(int i=0; i<Z->N; i++){
		for(int k=0; k<Z->P; k++){
			dZ[i+k*Z->N] = getModelMatrix(Z, i, k);
		}
	}
}


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
		double* ek;	ek    = Work + 2*j*P; //(double*)malloc(sizeof(double)*P);
		double* psijk; psijk = Work + 2*j*P + P; //(double*)malloc(sizeof(double)*P);
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
		double *gd; gd = grad + N; 
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
		
		if(j==5075)printM_d(Beta+5075*P,1,P,1,0);
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


// J x L
// G: N x L matrix
__global__ void getScoreStats(int L, SparseMat *Y, DataFrame *Z, double* G, double* Psi, double* W, double* Omega, double* phi, double* Work, double* S){
	int J = Y->J;
	int j = blockIdx.x;
	int lg= blockIdx.y;
	if(j<J && lg<L){
		int i;
		int N = Y->N;
		int P = Z->P;
		double *Wj; Wj  = &(W[j*N]);
		double *gl; gl  = &(G[lg*N]);
		double *tmp;  tmp  = Work + 2*(j+lg*J)*P; // Zt Oinvj gl
		double *tmp2; tmp2 = Work + 2*(j+lg*J)*P + P; // Psij Zt Oinvj gl
		double *Psij; Psij  = &(Psi[j*P*P]);

		double gVinvg = 0.0;
		fill0_d(tmp,  P);
		fill0_d(tmp2, P);
		getZtOinvjnj(Z, Wj, Omega, gl, tmp);
		cholsl(Psij, P, tmp, tmp2);

		for(i=0; i<N; i++){
			gVinvg += gl[i]*gl[i]*Wj[i]/Omega[i];
		}
		for(i=0; i<P; i++){
			gVinvg -= tmp[i]*tmp2[i];
		}	

		int l = Y->p[j];
		int lmax = Y->p[j+1];
		double ss=0.0;
		for(i=0; i<N; i++){
			if(i==Y->i[l] && l<lmax){
				// nj - Z tbj = (Yj-Wj)/Wj
				ss += gl[i]/Omega[i]*(Y->x[l]-Wj[i]); // Wji/Wji cancelled
				l++;
			}else{
				ss += gl[i]*Wj[i]/Omega[i]*(-1.0);
			}
		}
		S[j+lg*J] = ss/sqrt(phi[j]*gVinvg);
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
__global__ void getLkhdPois(SparseMat* Y, DataFrame* Z, double* W, double* delta, double* titsias, double* Psi, double* Omega, double* phi, double *lkhds){
	int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
	int J = Y->J;
	if(j<J){
		int p = Z->p;
		int P = Z->P;
		int N = Y->N;
		double *Wj; Wj  = &(W[j*N]);
		int i, k, l, lmax;
		double sum = 0.0;
		double *Psij; Psij  = &(Psi[j*P*P]);
		l = Y->p[j];
		lmax = Y->p[j+1];
		for(i=0; i<N; i++){
			if(Y->i[l]==i && l<lmax){
				sum += ((log(Wj[i])-delta[p-1]*titsias[i]) * Y->x[l] - Wj[i])/Omega[i];
				l++;
			}else{
				sum -= Wj[i]/Omega[i];
			}
		}
		for(k=0; k<P; k++){
			sum -= log(Psij[k+k*P]);
		}
		lkhds[j] = sum/phi[j] + log(2*3.1415926535)*((double)P)/2.0;
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
		int tot = N+p+N*Q+M*Q+Q*rhoflag;
		int ptp1 = (itr) % 10;
		for(k=0; k<N; k++){
			Omega[k] = exp(log(Omega[k+N]) - r*ss[k]);
		}
		//logscale(Omega, N);
		printf("delta[%d]=", itr);
		for(k=0; k<p; k++){
			delta[k] = exp(log(delta[k+p]) - r*ss[N+k]);
			printf("%lf, ", delta[k]);
		}
		printf("\n");
		for(i=0; i<N; i++){
			for(k=0; k<Q; k++){
				Xi[i+k*N] = Xi[i+k*N+N*Q] - r*ss[N+p+i+k*N];
			}
		}
		for(i=0; i<M; i++){
			for(k=0; k<Q; k++){
				Ta[i+k*M] = Ta[i+k*M+M*Q] - r*ss[N+p+N*Q+i+k*M];
			}
		}
		printf("rho[%d]=", itr);
		for(k=0; k<Q; k++){
			rho[k] = exp(log(rho[k+Q]) - r*ss[N+p+N*Q+M*Q+k]*((double)rhoflag));
			printf("%lf, ", rho[k]);
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
		int tot = N+p+N*Q+M*Q+Q*rhoflag; 
		if(itr==0){
			printf("ss_delta=");
			for(k=0; k<tot; k++){
				ss[k] = -grad[k]/10000.0;
				if(ss[k]>1.0){ss[k]=1.0;}
				if(ss[k]<(-1.0)){ss[k]=-1.0;}
				grad0[k] = grad[k];
				//if(k<10){ printf("%lf ", ss[k]);}
				if(k>=N && k<(N+p)){ printf("%lf ", ss[k]);}
				if(k>=N+p+N*Q+M*Q){ printf("%lf ", ss[k]);}
			}
			printf("\n");
		}else{
			for(k=0; k<tot; k++){
				XX[k+pt*tot] = grad[k] - grad0[k];
				grad0[k] = grad[k];
			}
			lbfgs(SS, XX, grad, tot, 10, itr, Work, ss);
		}
		//printM_d(SS,10,10,N+p,0);
		//printM_d(XX,10,10,N+p,0);
		//printM_d(ss,1,10,1,0);

		// omega update
		for(k=0; k<N; k++){
			Omega[k+N] = Omega[k]; // backup
			Omega[k] = exp(log(Omega[k]) - ss[k]);
		}
		//logscale(Omega, N);
		// delta update
		printf("delta[%d]=", itr);
		for(k=0; k<p; k++){
			delta[k+p] = delta[k]; // backup
			delta[k] = exp(log(delta[k]) - ss[N+k]); printf("%lf, ", delta[k]);
		}
		printf("\n");
		for(i=0; i<N; i++){
			for(k=0; k<Q; k++){
				Xi[i+k*N+N*Q] = Xi[i+k*N]; // backup
				Xi[i+k*N]	 = Xi[i+k*N] - ss[N+p+i+k*N];
			}
		}
		for(i=0; i<M; i++){
			for(k=0; k<Q; k++){
				Ta[i+k*M+M*Q] = Ta[i+k*M]; // backup
				Ta[i+k*M]	 = Ta[i+k*M] - ss[N+p+N*Q+i+k*M];
			}
		}
		printf("rho[%d]=", itr);
		for(k=0; k<Q; k++){
			rho[k+Q] = rho[k]; // backup
			rho[k] = exp(log(rho[k]) - ss[N+p+N*Q+M*Q+k]*((double)rhoflag)); printf("%lf, ", rho[k]);
		}
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
	if(verbose>0) printf("%s %s %s %s %s %s\n", argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);

	SparseMat *Y, *Y_h; Y_h=loadSparseMat(argv[1]);
	int N = Y_h->N;
	int J = Y_h->J;
	Y = newSparseMatOnDevice(Y_h);

	int Mredc = 5;
	int Mfull = 50;
	int M, L=0;
	int Q=16;

	M=100;// PBMC
	Q=51; // PBMC

	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-Mf")==0){Mfull=(int)atoi(argv[i+1]);break;}}
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-Mr")==0){Mredc=(int)atoi(argv[i+1]);break;}}
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-L")==0){L=(int)atoi(argv[i+1]);break;}}
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"-Q")==0){Q=(int)atoi(argv[i+1]);break;}} // N. LVs
        //FILE* fp; // init parameters
	//for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"--init-params")==0){fp=fopen(argv[i+1],"rb");break;}} // N. LVs
	M=Mfull;

	DataFrame *Z_h, *Z, *Z0_h, *Z0, *Z1_h, *Z1, *pZ, *pZ_h;
	Z_h = loadDataFrame(argv[2], Mredc); //printMetaDataFrame(Z_h);
	Z0_h = loadDataFrame(argv[2]); //printMetaDataFrame(Z_h);
	Z1_h = loadDataFrame(argv[2], Mfull); //printMetaDataFrame(Z_h);
	Z    = newDataFrameOnDevice(Z_h); // Knm fixed
	Z0   = newDataFrameOnDevice(Z0_h); // Knm fixed
	Z1   = newDataFrameOnDevice(Z1_h); // Knm fixed
	int P; 
	int P0 = Z0_h->P;// null
	int P1 = Z1_h->P;// full
	int p = Z_h->p;  // full & reduced
	P = P1;
	double *dZ, *dZ_h; dZ_h = (double*)malloc(sizeof(double)*N*P1);
		
	printf("N=%d, J=%d, P=%d, p=%d, M=%d, Q=%d, L=%d\n", N, J, P, p, M, Q, L);
	//printMetaDataFrame(Z_h);

	int JTRMAX=1000, q=0, jtr=0;

	double *Omega, *Omega_h;
	double *DeltaInv, *delta, *delta_h;
	double *zeta;
	double *ls_h, *ls;
	double *phi;
	double *Psi;
	double *W;
	double *Nu;
	double *Beta, *tBeta, *Beta_h;
	double *BFs, *BFs_h, *lkhds, *lkhd_h, *lkhd_d, *Knm_h, *Kmm_h;
	double *Work;

	Omega_h  = (double*)malloc(sizeof(double)*N); // init with Z
	delta_h  = (double*)malloc(sizeof(double)*p); // init with Z
	BFs_h  = (double*)malloc(sizeof(double)*J*(Q+2)); // init with Z
	Beta_h  = (double*)malloc(sizeof(double)*J*P1); // init with Z
	Knm_h  = (double*)malloc(sizeof(double)*N*Mfull); // init with Z
	Kmm_h  = (double*)malloc(sizeof(double)*Mfull*Mfull); // init with Z

	double *rho, *Xi, *Ta1, *Ta, *pT, *K, *Kinv, *KinvKmn, *titsias;
	double *Xi_h;  Xi_h   = (double*)malloc(sizeof(double)*N*Q); // init with Z
	double *Ta1_h; Ta1_h   = (double*)malloc(sizeof(double)*Mfull*Q); 
	double *Ta_h; Ta_h  = (double*)malloc(sizeof(double)*Mredc*Q); 
	double *rho_h;  rho_h  = (double*)malloc(sizeof(double)*Q); // for save purpos 
	double *phi_h;  phi_h  = (double*)malloc(sizeof(double)*J); // for save purpos 

	//FILE *fp; fp = fopen("params.bin", "rb");
	//fread(Omega_h, sizeof(double), N, fp);
	//fread(delta_h, sizeof(double), p, fp);
	//fread(Xi_h, sizeof(double), N*Q, fp);
	//fread(Ta1_h, sizeof(double), Mfull*Q, fp);
	//fread(rho_h, sizeof(double), Q, fp);
	//fclose(fp);
	initTaEquispaced(Mredc, Q, Ta_h);
	
	ls_h	= (double *)malloc(sizeof(double)*N); if(ls_h==NULL){fprintf(stderr, "ls_h not allocated...\n"); return 1;}
	lkhd_h  = (double *)malloc(sizeof(double)*JTRMAX);
	FILE *fls; fls = fopen(argv[3], "rb");
	fread(ls_h,  sizeof(double), N,   fls);

	double *G_h, *G;
	double* S_h, *S;
	double* d_h; // dox
	if(L>0){
		G_h = (double*)malloc(sizeof(double)*N*L);
		d_h = (double*)malloc(sizeof(double)*N);
		S_h = (double*)malloc(sizeof(double)*J*L);
		FILE *fg; fg = fopen(argv[4], "rb");
		fread(G_h,  sizeof(double), N*L,   fg);
		fclose(fg);
		FILE *fd; fd = fopen(argv[5], "rb");// dox
		fread(d_h,  sizeof(double), N,   fd);
		fclose(fd);
		for(int l=0; l<L; l++){
			for(int i=0; i<N; i++){
				G_h[i+l*N] *= d_h[i]; // dox * gen
			}
			scale(G_h+l*N,N);
		}
	}

	if(cudaMalloc((void **) &Psi,	sizeof(double)*J*P*P*2)  == cudaErrorMemoryAllocation){printf("Psi not allocated.\n"); return 1;}; // chol
	if(cudaMalloc((void **) &W,	sizeof(double)*N*(J+1))  == cudaErrorMemoryAllocation){printf("W not allocated.\n"); return 1;};
	if(cudaMalloc((void **) &Nu,   sizeof(double)*N*J)  == cudaErrorMemoryAllocation){printf("Nu not allocated.\n"); return 1;}; // pseudo data
	int lwork = (N<M*M ? M*M : N);
	lwork = (lwork<L*J ? L*J : lwork);
	lwork = lwork*(2*P);
	if(cudaMalloc((void **) &Work,  sizeof(double)*lwork)   == cudaErrorMemoryAllocation){printf("Work not allocated.\n"); return 1;};;
	if(cudaMalloc((void **) &ls,	 sizeof(double)*N) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Omega,  sizeof(double)*N*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &DeltaInv,  sizeof(double)*P*P) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &delta,  sizeof(double)*p*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &phi,	sizeof(double)*J) == cudaErrorMemoryAllocation){return 1;};
	
	if(cudaMalloc((void **) &zeta, sizeof(double)*P) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Beta, sizeof(double)*P*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &tBeta, sizeof(double)*P*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &lkhds, sizeof(double)*J) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &BFs, sizeof(double)*J*(Q+2)) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &G,   sizeof(double)*N*L) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &S,   sizeof(double)*J*L) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &lkhd_d, sizeof(double)*JTRMAX) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &dZ, sizeof(double)*N*P1) == cudaErrorMemoryAllocation){return 1;};
	cudaMemcpy(G, G_h, sizeof(double)*N*L, cudaMemcpyHostToDevice); // doxgen

	if(cudaMalloc((void **) &rho, sizeof(double)*Q*2)   == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Xi,  sizeof(double)*N*Q*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Ta1, sizeof(double)*Mfull*Q*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Ta,  sizeof(double)*Mredc*Q*2) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &K, sizeof(double)*M*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &Kinv, sizeof(double)*M*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &KinvKmn, sizeof(double)*N*M) == cudaErrorMemoryAllocation){return 1;};
	if(cudaMalloc((void **) &titsias, sizeof(double)*N) == cudaErrorMemoryAllocation){return 1;};

	cudaMemcpy(ls,	ls_h,    sizeof(double)*N,	 cudaMemcpyHostToDevice);
	
	for(int i=0; i<argc-1; i++){if(strcmp(argv[i],"--init-params")==0){
                readParams(argv[i+1], N, J, M, P, p, Q, Omega, delta, Xi, Ta1, rho, phi, Omega_h, delta_h, Xi_h, Ta1_h, rho_h, phi_h);
                break;
        }}
	//cudaMemcpy(Omega, Omega_h, sizeof(double)*N, cudaMemcpyHostToDevice);
	//cudaMemcpy(delta, delta_h, sizeof(double)*p, cudaMemcpyHostToDevice);
	//cudaMemcpy(Xi, Xi_h, sizeof(double)*N*Q, cudaMemcpyHostToDevice);
	//cudaMemcpy(Ta1, Ta1_h, sizeof(double)*Mfull*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(Ta, Ta_h, sizeof(double)*Mredc*Q, cudaMemcpyHostToDevice);
	//cudaMemcpy(rho, rho_h, sizeof(double)*Q, cudaMemcpyHostToDevice);
	printParams<<<1,1>>>(N, J, p, Mredc, Q, Omega, delta, rho, Xi, Ta, zeta, phi);
	
	printf("Memory allocated...\n");

	dim3 gridJ(J,1); // J
	
	dim3 gridPsi0(J,P0); // J*P*P
	dim3 blockPsi0(1,P0);
	dim3 gridPP0(P0,P0); // P0*P0
	
	int sN = (int)(sqrt((double)N)/4)+1;
	dim3 gridN(sN,sN); // N
	dim3 blockN(4,4);
	printf("N=%d sN^2*16=%d\n", N, sN*sN*16);
	
	dim3 gridNpMM(N+M,M);

	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
	float milliseconds = 0;

	printf("N=%d, M=%d, Q=%d, P=%d, P0=%d\n", N, M, Q, P, P0);
	// start with X
	cudaEventRecord(start);
	initPhi<<<1,1>>>(J, phi);cudaDeviceSynchronize();
	initTitsias<<<1,1>>>(N, titsias);cudaDeviceSynchronize();
	
	expandDeltaWOK_k<<<1,1>>>(Z0, delta, DeltaInv);cudaDeviceSynchronize();
	initPsi<<<gridPP0, 1>>>(Z0, DeltaInv, Psi);cudaDeviceSynchronize();
	initPsi2<<<1,1>>>(P0, Psi);cudaDeviceSynchronize();
	initBeta<<<gridJ, 1>>>(Y, ls, Z0, Psi, Work, Beta);cudaDeviceSynchronize(); 
	cudaEventRecord(stop);cudaEventSynchronize(stop);cudaEventElapsedTime(&milliseconds, start, stop);
	printf("initBeta: %f\n", milliseconds);
	getZeta<<<P0, 1>>>(J, P0, Beta, phi, zeta);cudaDeviceSynchronize();
	for(jtr=0; jtr<JTRMAX; jtr++){

		cudaEventRecord(start);
		getWForBeta<<<gridJ, 1>>>(Y, Z0, ls, Beta, zeta, delta, titsias, W, Nu);cudaDeviceSynchronize();
		printf("W Nu: %f\n", milliseconds);
			
		getPsi<<<gridPsi0, blockPsi0>>>(J, Z0, Omega, DeltaInv, W, Psi);cudaDeviceSynchronize();
		cholPsi<<<gridJ, 1>>>(J, P0, Psi);cudaDeviceSynchronize();
		printf("Psi: %f\n", milliseconds);
			
		getBetaPhi<<<gridJ, 1>>>(J, Z0, ls, Psi, Omega, zeta, Nu, W, Work, Beta, phi);cudaDeviceSynchronize();
		getZeta<<<P0, 1>>>(J, P0, Beta, phi, zeta);cudaDeviceSynchronize();
		printf("get Beta Phi Zeta: %f\n", milliseconds);
			
		getLkhdPois<<<gridJ, 1>>>(Y, Z0, W, delta, titsias, Psi, Omega, phi, lkhds);cudaDeviceSynchronize();
		cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
		printf("getLkhd: %f\np", milliseconds);
		collectLkhd<<<1,1>>>(0, Y_h->J, 0, Omega, lkhds, q, jtr, Z0, delta, Xi, Ta, W, titsias, wt, lkhd_d);cudaDeviceSynchronize();
		cudaMemcpy(lkhd_h, lkhd_d, sizeof(double)*(jtr+1), cudaMemcpyDeviceToHost);
		if(jtr>0){
			printf("lkhd=%lf\n", lkhd_h[jtr]);
			if(isinf(lkhd_h[jtr])>0 || isnan(lkhd_h[jtr])>0){fprintf(stderr, "LKHD becomes NAN!!!\n"); break;}
			if(jtr==(JTRMAX-1) || fabs(lkhd_h[jtr-1]-lkhd_h[jtr])/fabs(lkhd_h[jtr])<1e-5){
				break;
			}
		}
	}
	cudaMemcpy(BFs, lkhds, sizeof(double)*J, cudaMemcpyDeviceToDevice);
	
	printf("iteration start...\n");
	for(q=0; q<Q+1; q++){// q=Q : full model
		printf("###################\n");
		printf("#	LVq=%d      \n", q);
		printf("###################\n");
	
		if(q==Q){// full
			pZ=Z1;
			pZ_h=Z1_h;
			pT=Ta1;
			P = P1;
			M=Mfull;
			printf("Pnull=%d Mfull=%d\n", P, M);
			dim3 gridNpMM(N+Mfull,Mfull);
			getKernelMats<<<gridNpMM,1>>>(Q, Xi, pT, rho, pZ, K, Kinv, KinvKmn);cudaDeviceSynchronize();
		}else{// reduced
			M=Mredc;
			P=Z_h->P;
			pZ=Z;
			pZ_h=Z_h;
			pT=Ta;
			dim3 gridNpMM(N+Mredc,Mredc);
			getKernelMats<<<gridNpMM,1>>>(Q, Xi, pT, rho, pZ, q, K, Kinv, KinvKmn);cudaDeviceSynchronize();	
		}
		printParams<<<1,1>>>(N, J, p, M, Q, Omega, delta, rho, Xi, pT, zeta, phi);
		cholKmm<<<1,1>>>(M, Kinv);cudaDeviceSynchronize();
		getKinvKmn<<<gridN,blockN>>>(Kinv, pZ, KinvKmn, titsias);cudaDeviceSynchronize();

		expandDelta_k<<<1,1>>>(pZ, delta, K, DeltaInv);cudaDeviceSynchronize();
		dim3 gridPP(pZ_h->P,pZ_h->P);
		initPsi<<<gridPP, 1>>>(pZ, DeltaInv, Psi);cudaDeviceSynchronize();
		initPsi2<<<1,1>>>(P, Psi);cudaDeviceSynchronize();
		initBeta<<<gridJ, 1>>>(Y, ls, pZ, Psi, Work, Beta);cudaDeviceSynchronize();
		getZeta<<<pZ_h->P, 1>>>(J, P, Beta, phi, zeta);cudaDeviceSynchronize();

		for(jtr=0; jtr<JTRMAX; jtr++){

			cudaEventRecord(start);
			getWForBeta<<<gridJ, 1>>>(Y, pZ, ls, Beta, zeta, delta, titsias, W, Nu);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("W Nu: %f\n", milliseconds);
			
			cudaEventRecord(start);
			dim3 gridPsi(J,P);
			dim3 blockPsi(1,P);
			getPsi<<<gridPsi, blockPsi>>>(J, pZ, Omega, DeltaInv, W, Psi);cudaDeviceSynchronize();
			cholPsi<<<gridJ, 1>>>(J, P, Psi);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Psi: %f\n", milliseconds);
			
			cudaEventRecord(start);
			getBetaPhi<<<gridJ, 1>>>(J, pZ, ls, Psi, Omega, zeta, Nu, W, Work, Beta, phi);cudaDeviceSynchronize();
			getZeta<<<pZ_h->P, 1>>>(J, P, Beta, phi, zeta);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("get Beta Phi Zeta: %f\n", milliseconds);
			
			cudaEventRecord(start);
			getLkhdPois<<<gridJ, 1>>>(Y, pZ, W, delta, titsias, Psi, Omega, phi, lkhds);cudaDeviceSynchronize();
			cudaEventRecord(stop);cudaEventSynchronize(stop);milliseconds = 0;cudaEventElapsedTime(&milliseconds, start, stop);
			printf("getLkhd: %f\np", milliseconds);
			collectLkhd<<<1,1>>>(0, Y_h->J, 0, Omega, lkhds, q, jtr, pZ, delta, Xi, pT, W, titsias, wt, lkhd_d);cudaDeviceSynchronize();
			cudaMemcpy(lkhd_h, lkhd_d, sizeof(double)*(jtr+1), cudaMemcpyDeviceToHost);
			if(jtr>0){
				if(isinf(lkhd_h[jtr])>0 || isnan(lkhd_h[jtr])>0){fprintf(stderr, "LKHD becomes NAN!!!\n"); break;}
				if(jtr==(JTRMAX-1) || fabs(lkhd_h[jtr-1]-lkhd_h[jtr])/fabs(lkhd_h[jtr])<1e-8){
					//lkhd_h[0] = lkhd_h[jtr];
					//cudaMemcpy(lkhd_d, lkhd_h, sizeof(double), cudaMemcpyHostToDevice);
					break;
				}
			}	
		}
		// Beta
		cudaMemcpy(Beta_h, Beta, sizeof(double)*J*P, cudaMemcpyDeviceToHost);
		gzFile f = gzopen("Beta_de.gz","a6h");
		for(int i=0; i<J; i++){for(int k=0; k<P; k++){gzprintf(f, "%lf,", Beta_h[i*P+k]);};gzprintf(f, "\n");}
		gzclose(f);
		// Knm
		DataFrame *pZ_h; pZ_h = (DataFrame*)malloc(sizeof(DataFrame));
		cudaMemcpy(pZ_h, pZ, sizeof(DataFrame), cudaMemcpyDeviceToHost);
		cudaMemcpy(Knm_h, pZ_h->Knm, sizeof(double)*N*M, cudaMemcpyDeviceToHost);
		cudaMemcpy(Kmm_h, K, sizeof(double)*M*M, cudaMemcpyDeviceToHost);
		f = gzopen("Knm_de.gz","a6h");
		for(int i=0; i<N; i++){for(int k=0; k<M; k++){gzprintf(f, "%lf,", Knm_h[i+k*N]);};gzprintf(f, "\n");}
		for(int i=0; i<M; i++){for(int k=0; k<M; k++){gzprintf(f, "%lf,", Kmm_h[i+k*M]);};gzprintf(f, "\n");}
		gzclose(f);
		// BFs
		cudaMemcpy(BFs+J*(q+1), lkhds, sizeof(double)*J, cudaMemcpyDeviceToDevice);
	}	
	printf("Iteration end...\n");

	cudaMemcpy(BFs_h, BFs, sizeof(double)*J*(Q+2), cudaMemcpyDeviceToHost);
	gzFile f = gzopen("BFs_de.gz","w6h");
	int i,k;
	for(i=0; i<J; i++){for(k=0; k<Q+2; k++){gzprintf(f, "%lf,", BFs_h[i+k*J]);};gzprintf(f, "\n");}
	gzclose(f);


	// genotype score stats
	if(L>0){
		printf("Score Stats calculation...");
		dim3 gridJL(J,L);
		getScoreStats<<<gridJL,1>>>(L, Y, pZ, G, Psi, W, Omega, phi, Work, S);cudaDeviceSynchronize();
		cudaMemcpy(S_h, S, sizeof(double)*J*L, cudaMemcpyDeviceToHost);
		f = gzopen("ScoreStats.gz","w6h");
		for(int i=0; i<J; i++){for(int k=0; k<L; k++){gzprintf(f, "%lf,", S_h[i+k*J]);};gzprintf(f, "\n");}
		gzclose(f);
		printf("Done.\n");

		getDesignMatrix<<<1,1>>>(pZ, dZ);cudaDeviceSynchronize();
		cudaMemcpy(dZ_h, dZ, sizeof(double)*N*P1, cudaMemcpyDeviceToHost);
		f = gzopen("DesignMatrix.gz","w6h");
		double* W_h; W_h = (double*)malloc(sizeof(double)*N*2);
		double* Omega_h; Omega_h = (double*)malloc(sizeof(double)*N);
		cudaMemcpy(W_h, W, sizeof(double)*N*2, cudaMemcpyDeviceToHost);
		cudaMemcpy(Omega_h, Omega, sizeof(double)*N, cudaMemcpyDeviceToHost);
		for(int i=0; i<N; i++){for(int k=0; k<P1; k++){gzprintf(f, "%lf,", dZ_h[i+k*N]);};gzprintf(f, "%lf,%lf\n", W_h[i+N], Omega_h[i]);}
		gzclose(f);
	}
	free(Y_h); free(Z_h); free(ls_h); free(lkhd_h); free(Xi_h);
	cudaFree(Y); cudaFree(Z); cudaFree(Omega); cudaFree(DeltaInv); cudaFree(delta); cudaFree(ls); cudaFree(phi); 
	cudaFree(Psi); cudaFree(W); cudaFree(Nu); cudaFree(zeta); cudaFree(lkhds); cudaFree(Beta); cudaFree(tBeta); 

	return 0;
}







