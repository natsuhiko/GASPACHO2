#include <stdio.h>
//#include "DataFrame.cuh"
#include "SparseMat.cuh"

#include "util_cuda.cuh"
#include "chol.cuh"

/*
// too slow
// D^-1 + Z^T Oinv Z
// upper tri only
// with Cholesky decomp (lower tri only with diagonal elements)
__device__ void getPsij(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *Delta){
	int i, k, l;
	double Oijinv;
	fill0_d(Psij, P*P);
	for(i=0; i<N; i++){
		Oijinv = Wj[i]/Omega[i];
		for(k=0; k<P; k++){
			for(l=k; l<P; l++){
				Psij[k+l*P] += Z[i+k*N]*Oijinv*Z[i+l*N];
			}
		}
	}
	for(k=0; k<P; k++){ Psij[k+k*P] += 1.0/Delta[k]; }
	choldc(Psij, P);
}
*/

//Z, Psij, Wj, Omega, Delta, nj, phij
__device__ double getLkhdjDense(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *nj, double *DeltaInv, double phij, double* tmp, double* tmp2){
	int i, k;
	double Oijinv;

	double lkhdj = - ((double)N)/2.0*log(phij);
	//if(isnan(lkhdj)>0)printf("lkhd nan error 1\n");
	for(i=0; i<N; i++){
		lkhdj -= log(Omega[i]/Wj[i])/2.0;
	}
	//if(isnan(lkhdj)>0)printf("lkhd nan error 2\n");
	for(k=0; k<P; k++){
		lkhdj -= log(Psij[k+k*P]); // Cholesky Decomp!!!!
		//if(evals[k]>0.0) lkhdj -= log(evals[k])/2.0;
		lkhdj += log(DeltaInv[k+k*P]);
	}
	//if(isnan(lkhdj)>0)printf("lkhd nan error 3\n");
	//double* tmp;  tmp  = (double*)malloc(sizeof(double)*P); 
	fill0_d(tmp,  P);
	//double* tmp2; tmp2 = (double*)malloc(sizeof(double)*P); 
	fill0_d(tmp2, P); // tilde bj actually
	for(i=0; i<N; i++){
		Oijinv = Wj[i]/Omega[i];
		lkhdj -= nj[i]*nj[i]*Oijinv/2.0/phij;
		for(k=0; k<P; k++){ // Z^T (Wj/Omega) nj
			tmp[k] += Z[i+k*N]*nj[i]*Oijinv;
			if(isinf(tmp[k])>0){
				//printf("zk^T Ojnv nj is inf! nij=%lf Oi=%lf Wij=%lf", nj[i], Omega[i], Wj[i]); 
				return -log(-1.0);
			}
		}
	}
	cholsl(Psij, P, tmp, tmp2);
	for(k=0; k<P; k++){
		lkhdj += tmp[k]*tmp2[k]/2.0/phij;
		if(isnan(lkhdj)>0){ //printf("phi=%lf lkhd nan error 4\n", phij); 
			//printM_d(Psij, P,P,P,0);
			//printM_d(evals,1,P,1,0);
			//printM_d(tmp,1,P,1,0);
			//printM_d(tmp2,1,P,1,0);
			break;}
	}
	//free(tmp); free(tmp2);
	return lkhdj;
}


// Z, bj, ls, zeta -> Wj
// Z, (bj-zeta), yj, Wj -> nj
__device__ void getWj(int N, int P, double *Z, double *bj, double *ls, double *zeta, double *yj, double *Wj, double *nj){
	int i, k;
	double sum, sum2;
	for(i=0; i<N; i++){
		//Wj[i] = nj[i] = 0.0;
		sum = 0.0;
		sum2= 0.0;
		for(k=0; k<P; k++){
			sum += Z[i+k*N] *  bj[k];
			sum2 += Z[i+k*N] * (bj[k]-zeta[k]);
		}
		Wj[i] = exp(sum)*ls[i];
		nj[i] = sum2 + (yj[i]-Wj[i])/Wj[i];
	}
}

//Z, Psij, nj, Wj, Omega -> tbj not bj!
__device__ void gettbj(int N, int P, double *Z, double *Psij,  double *Wj, double *Omega, double *nj, double* tmp, double *tbj){
	int i, k;

	//double* tmp; tmp = (double*)malloc(sizeof(double)*P); 
	fill0_d(tmp, P);
	double Oijinv;
	for(i=0; i<N; i++){
		Oijinv = Wj[i]/Omega[i];
		for(k=0; k<P; k++){
			tmp[k] += Z[i+k*N]*nj[i]*Oijinv;
		}
	}
	cholsl(Psij, P, tmp, tbj);
	//free(tmp);
}


//Z, Psij, nj, Wj, Omega -> bj
__device__ double getbjphij(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *nj, double *zeta, double* tmp, double *bj){
	int i, k;

	double phij = 0.0;
	double Oijinv;
	//double* tmp; tmp = (double*)malloc(sizeof(double)*P); 
	fill0_d(tmp, P);
	for(i=0; i<N; i++){
		Oijinv = Wj[i]/Omega[i];
		for(k=0; k<P; k++){
			tmp[k] += Z[i+k*N]*nj[i]*Oijinv;
		}
		phij += nj[i]*nj[i]*Oijinv;
	}
	cholsl(Psij, P, tmp, bj);
	if(isnan(bj[0])>0){
		//printf("bj is nan!\n"); 
		fill0_d(bj, P);
	}
	for(k=0; k<P; k++){
		phij  -= bj[k]*tmp[k];
		bj[k] += zeta[k];
	}
	
	//free(tmp);

	return phij/((double)N);
}


__device__ void fill0_dg(double* x, int n, int ldx){
	int i;
	for(i=0; i<n; i++){
		x[i*ldx] = 0.0;
	}
}


__device__ void fill0_d(double* x, int n){
	int i;
	for(i=0; i<n; i++){
		x[i] = 0.0;
	}
}

__device__ void fill1_d(double* x, int n){
	int i;
	for(i=0; i<n; i++){
		x[i] = 1.0;
	}
}

__device__ double nk_fdot(double* x, int ldx, double* y, int ldy, int n){
	double sum=0.0;
	int i;
	for(i=0; i<n; i++){
		sum += x[i*ldx]*y[i*ldy];
	}
	return sum;
}



__device__ void nk_fcopy(double cons, double* a, int lda, double* b, int ldb, int n){
	int i;
	for(i=0; i<n; i++){
		b[i*ldb] = cons * a[i*lda];
	}
}

// g = grad at itr
// Omega, delta, Xi, Ta, rho
// N + p + NxQ + M*Q + Q
__device__ void lbfgs(double* S, double* Y, double* g, int N, int M, int itr, double* Work, double* z){
	int pt = (itr-1) % M;
	int i, ii, k;
	double num, den;
	double* q; q = Work; //(double*)malloc(sizeof(double)*N); 
	nk_fcopy(1.0, g, 1, q, 1, N);
	double* a; a = Work + N; //(double*)malloc(sizeof(double)*M);
	//double* b; b = (double*)malloc(sizeof(double)*M);
	for(i=pt; i>=(itr>M ? pt-M+1 : 0); i--){
		ii = (i+M) % M;
		//printf("%d %d\n", i, ii);
		num = den = 0.0;
		for(k=0; k<N; k++){ // a[ii] = sum(S[,ii]*q)/sum(S[,ii]*Y[,ii]) on R
			num += S[k+ii*N]*q[k];
			den += S[k+ii*N]*Y[k+ii*N];
		}
		a[ii] = num/den;
		for(k=0; k<N; k++){ // q = q - a[ii]*Y[,ii] on R
			q[k] -= a[ii]*Y[k+ii*N];
		}
	}
	num = den = 0.0;
	for(k=0; k<N; k++){
		num += S[k+pt*N]*Y[k+pt*N];
		den += Y[k+pt*N]*Y[k+pt*N];
	}
	nk_fcopy(num/den, q, 1, z, 1, N);
	for(i=(itr>M ? pt-M+1 : 0); i<=pt; i++){
		ii = (i+M) % M;
		num = den = 0.0;
		for(k=0; k<N; k++){ // b[ii] = sum(Y[,ii]*z)/sum(S[,ii]*Y[,ii]) on R
			num += Y[k+ii*N]*z[k];
			den += S[k+ii*N]*Y[k+ii*N];
		}
		//b[ii] = num/den;
		for(k=0; k<N; k++){ // z = z + S[,i]*(a[i]-bi) on R
			z[k] += S[k+ii*N] * (a[ii]-num/den);
		}
	}
	for(k=0; k<N; k++){
		if(z[k]>1.0){z[k]=1.0;}
		if(z[k]<(-1.0)){z[k]=-1.0;}
		z[k] = -z[k];
	}
	//free(q); free(a);
}


__global__ void expandDeltaWOK_k(DataFrame *Z, double *delta, double* DeltaInv){
	int i, j;
	int p = Z->p;
	int* nh; nh = Z->nh;
	int P = Z->nh[p];
	fill0_d(DeltaInv, P*P);
	for(i=0; i<p; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			DeltaInv[j+j*P] = 1.0/delta[i];
		}
	}
}


__global__ void expandDelta_k(DataFrame* Z, double *delta, double* Kmm, double* DeltaInv){
	int i, j, k;
	int p = Z->p;
	int P = Z->P;
	int M = Z->M;
	fill0_d(DeltaInv, P*P);
	for(i=0; i<p-1; i++){
		for(j=Z->nh[i]; j<Z->nh[i+1]; j++){
			DeltaInv[j+j*P] = 1.0/delta[i];
		}
	}
	// i = p-1
	for(j=0; j<M; j++){
		int jj = j+Z->nh[p-1];
		for(k=j; k<M; k++){
			int kk = k+Z->nh[p-1];
			DeltaInv[jj+kk*P] = Kmm[j+k*M]/delta[p-1];
		}
	}
	//printM_d(DeltaInv,P,P,P,0);
}


__global__ void expandDelta_k(int p, int* nh, double *delta, double* Kmm, double* DeltaInv){
	int i, j, k;
	int P = nh[p];
	int M = nh[p] - nh[p-1];
	fill0_d(DeltaInv, P*P);
	for(i=0; i<p-1; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			DeltaInv[j+j*P] = 1.0/delta[i];
		}
	}
	// i = p-1
	for(j=0; j<M; j++){
		int jj = j+nh[p-1];
		for(k=j; k<M; k++){
			int kk = k+nh[p-1];
			DeltaInv[jj+kk*P] = Kmm[j+k*M]/delta[p-1];
		}
	}
	//printM_d(DeltaInv,P,P,P,0);
}


__device__ void expandDelta_d(int p, int* nh, double *delta, double* Kmm, double* DeltaInv){
	int i, j, k;
	int P = nh[p];
	int M = nh[p] - nh[p-1];
	fill0_d(DeltaInv, P*P);
	for(i=0; i<p-1; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			DeltaInv[j+j*P] = 1.0/delta[i];
		}
	}
	// i = p-1
	for(j=0; j<M; j++){
		int jj = j+nh[p-1];
		for(k=j; k<M; k++){
			int kk = k+nh[p-1];
			DeltaInv[jj+kk*P] = Kmm[j+k*M]/delta[p-1];
		}
	}
}


__global__ void expandDelta_cdg0_k(int p, int* nh, double *delta, double* K, double* DeltaInv){
	// C | D | G 1
	// P | M | M+1
	int i, j, k, jj, kk;
	int P = nh[p];
	int M = nh[p] - nh[p-1];
	int ldd = P + 1;
	fill0_d(DeltaInv, ldd*ldd);
	for(i=0; i<p-1; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			DeltaInv[j+j*ldd] = 1.0/delta[i];
		}
	}
	// i = p-1 context
	for(j=0; j<M; j++){
		jj = j+nh[p-1];
		for(k=j; k<M; k++){
			kk = k+nh[p-1];
			DeltaInv[jj+kk*ldd] = K[j+k*M]/delta[p-1];
		}
	}
	DeltaInv[ldd*ldd-1] = 1.0 / delta[p];
}


__global__ void expandDelta_cdg_k(int p, int* nh, double *delta, double* K, double* Kta, double* DeltaInv){
	// C | D | G 1
	// P | M | M+1
	int i, j, k, jj, kk;
	int P = nh[p];
	int M = nh[p] - nh[p-1];
	int ldd = P + M + M+1;
	fill0_d(DeltaInv, ldd*ldd);
	for(i=0; i<p-1; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			DeltaInv[j+j*ldd] = 1.0/delta[i];
		}
	}
	// i = p-1 context
	for(j=0; j<M; j++){
		jj = j+nh[p-1];
		for(k=j; k<M; k++){
			kk = k+nh[p-1];
			DeltaInv[jj+kk*ldd] = K[j+k*M]/delta[p-1];
		}
	}
	for(j=0; j<M; j++){
		jj = j+nh[p-1]+M;
		for(k=j; k<M; k++){
			kk = k+nh[p-1]+M;
			DeltaInv[jj+kk*ldd] = Kta[j+k*M]/delta[p];
		}
	}
	for(j=0; j<M; j++){
		jj = j+nh[p-1]+2*M;
		for(k=j; k<M; k++){
			kk = k+nh[p-1]+2*M;
			DeltaInv[jj+kk*ldd] = Kta[j+k*M]/delta[p+1];
		}
	}
	DeltaInv[ldd*ldd-1] = 1.0 / delta[p+1];
}


__global__ void printM_k(double *Y, int n, int m, int ldy, int integ){
        int i, j;
        for(i=0; i<n; i++){
                for(j=0; j<m; j++){
                        if(integ==0){
                                printf("%lf,", Y[i+ldy*j]);
                        }else{
                                printf("%.1lf,", Y[i+ldy*j]);
                        }
                }
                printf("\n");
        }
}


__device__ void printM_d(double *Y, int n, int m, int ldy, int integ){
	int i, j;
	for(i=0; i<n; i++){
		for(j=0; j<m; j++){
			if(integ==0){
				printf("%lf,", Y[i+ldy*j]);
			}else{
				printf("%.1lf,", Y[i+ldy*j]);
			}
		}
		printf("\n");
	}
}

__device__ void checkNA(double *x, int n, char* mes){
	int i;
	for(i=0; i<n; i++){
		if(isnan(x[i])>0){
			printf("%s: x[%d] is nan!\n", mes, i);
		}
	}
}

__device__ void scale_d(double *x, int n){
	int i;
	double sum=0.0;
	for(i=0; i<n; i++){
		sum += (x[i]);
	}
	sum = sum/((double)n);
	for(i=0; i<n; i++){
		x[i] -= (sum);
	}
}


__device__ void logscale(double *x, int n){
	int i;
	double sum=0.0;
	for(i=0; i<n; i++){
		sum += log(x[i]);
	}
	sum = sum/((double)n);
	for(i=0; i<n; i++){
		x[i] /= exp(sum);
	}
}

__device__ double quadform(double *A, double *b, int N){
	double sum = 0.0;
	for(int i=0; i<N; i++){
		sum += A[i+i*N]*b[i]*b[i];
		for(int j=i+1; j<N; j++){
			sum += 2.0*A[i+j*N]*b[i]*b[j];
		}
	}
	return sum;
}

__device__ double getYij(double* Y, int* iv, int *pv, int i, int j){
	for(int k=pv[j]; k<pv[j+1]; k++){
		if(iv[k]==i){
			return Y[k];
		}
	}
	return 0.0;
}

__device__ void getZtOinvjnj(DataFrame *Z, double* Wj, double* Omega, double* nj, double* bj){
	double sum;
	int N = Z->N;
	int P = Z->P;
	int *nh; nh = Z->nh;
	int* pi;
	double* pd;
	fill0_d(bj, P);
	for(int k=0; k<Z->p; k++){
		if(Z->c[k]==1){
			pd = (double*)(Z->x + Z->b[k]);
			for(int m=0; m<Z->v[k]; m++){
				sum = 0.0;
				for(int i=0; i<N; i++){
					sum += pd[i+m*N]*Wj[i]/Omega[i]*nj[i];
				}
				bj[nh[k]+m] = sum;
			}
		}else{ // cate
			pi = (int*)(Z->x + Z->b[k]);
			for(int i=0; i<N; i++){
				bj[nh[k]+pi[i]] += Wj[i]/Omega[i]*nj[i];
			}
		}
	}
}


__device__ void getZtOinvjnj(SparseMat *Z, double* Wj, double* Omega, double* nj, double* b){
	int i, j, l;
	double sum;
	for(j=0; j<Z->J; j++){
		sum = 0.0;
		for(l=Z->p[j]; l<Z->p[j+1]; l++){
			i = Z->i[l];
			sum += Z->x[l]*Wj[i]/Omega[i]*nj[i];
		}
		b[j] = sum;
	}
}

__device__ void bsort(int* a, int* b, double* c, int n)
{
    int i, j, temp;
    double dtmp;

    for(i=0; i<n-1; i++){
        for(j=0; j<n-1; j++){
            if(a[j+1] < a[j]){
                temp = a[j]; a[j] = a[j+1]; a[j+1] = temp;
                temp = b[j]; b[j] = b[j+1]; b[j+1] = temp;
                dtmp = c[j]; c[j] = c[j+1]; c[j+1] = dtmp;
            }
        }
    }
}


void readParams(char* fname, int N, int J, int M, int P, int p, int Q, double* Omega, double* delta, double* Xi, double* Ta, double* rho, double* phi,
                double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h, double* phi_h){
        FILE *fpa; fpa = fopen(fname, "rb");
	//int tmp;
	/*fread(&tmp, sizeof(int), 1, fpa); if(N!=tmp){fprintf(stderr, "N is inconsistent!\n");}
	fread(&tmp, sizeof(int), 1, fpa); if(J!=tmp){fprintf(stderr, "J is inconsistent!\n");}
	fread(&tmp, sizeof(int), 1, fpa); if(M!=tmp){fprintf(stderr, "M is inconsistent!\n");}
	fread(&tmp, sizeof(int), 1, fpa); if(P!=tmp){fprintf(stderr, "P is inconsistent!\n");}
	fread(&tmp, sizeof(int), 1, fpa); if(p!=tmp){fprintf(stderr, "p is inconsistent!\n");}
	fread(&tmp, sizeof(int), 1, fpa); if(Q!=tmp){fprintf(stderr, "Q is inconsistent!\n");}*/
        fread(Omega_h, sizeof(double), N, fpa);
        fread(delta_h, sizeof(double), p, fpa);
        fread(Xi_h,    sizeof(double), N*Q, fpa);
        fread(Ta_h,    sizeof(double), M*Q, fpa);
        fread(rho_h,   sizeof(double), Q, fpa);
        fread(phi_h,   sizeof(double), J, fpa);
        fclose(fpa);
        printf("Init param=%s\n", fname);
        cudaMemcpy(Omega, Omega_h,  sizeof(double)*N,   cudaMemcpyHostToDevice);
        cudaMemcpy(delta, delta_h,  sizeof(double)*p,   cudaMemcpyHostToDevice);
        cudaMemcpy(Xi,    Xi_h,     sizeof(double)*N*Q, cudaMemcpyHostToDevice);
        cudaMemcpy(Ta,    Ta_h,     sizeof(double)*M*Q, cudaMemcpyHostToDevice);
        cudaMemcpy(rho,   rho_h,    sizeof(double)*Q,   cudaMemcpyHostToDevice);
        cudaMemcpy(phi,   phi_h,    sizeof(double)*J,   cudaMemcpyHostToDevice);
}

__device__ void mscale_d(int Q, double* rho, double m){
	double mh=0.0;
	for(int i=0; i<Q; i++){
		mh += rho[i];
	}
	mh /= ((double)Q);
	for(int i=0; i<Q; i++){
		rho[i] *= m/mh;
	}
}

void writeParams(char* fname, int N, int J, int M, int P, int p, int Q, double* Omega, double* delta, double* Xi, double* Ta, double* rho, double* phi,
                double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h, double* phi_h){
        cudaMemcpy(Omega_h, Omega,  sizeof(double)*N,   cudaMemcpyDeviceToHost);
        cudaMemcpy(delta_h, delta,  sizeof(double)*p,   cudaMemcpyDeviceToHost);
        cudaMemcpy(Xi_h,    Xi,     sizeof(double)*N*Q, cudaMemcpyDeviceToHost);
        cudaMemcpy(Ta_h,    Ta,     sizeof(double)*M*Q, cudaMemcpyDeviceToHost);
        cudaMemcpy(rho_h,   rho,    sizeof(double)*Q,   cudaMemcpyDeviceToHost);
        cudaMemcpy(phi_h,   phi,    sizeof(double)*J,   cudaMemcpyDeviceToHost);
        FILE *fpa; fpa = fopen(fname, "wb");
	fwrite(&N, sizeof(int), 1, fpa);
	fwrite(&J, sizeof(int), 1, fpa);
	fwrite(&M, sizeof(int), 1, fpa);
	fwrite(&P, sizeof(int), 1, fpa);
	fwrite(&p, sizeof(int), 1, fpa);
	fwrite(&Q, sizeof(int), 1, fpa);
        fwrite(Omega_h, sizeof(double), N, fpa);
        fwrite(delta_h, sizeof(double), p, fpa);
        fwrite(Xi_h,    sizeof(double), N*Q, fpa);
        fwrite(Ta_h,    sizeof(double), M*Q, fpa);
        fwrite(rho_h,   sizeof(double), Q, fpa);
        fwrite(phi_h,   sizeof(double), J, fpa);
        fclose(fpa);
        printf("Output param=%s\n", fname);
}
