#include "chol.cuh"
#include "DataFrame.cuh"
__device__ void getPsij(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *Delta);
__device__ double getLkhdjDense(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *nj, double *Delta, double phij, double* tmp, double* tmp2);
__device__ void getWj(int N, int P, double *Z, double *bj, double *ls, double *zeta, double *yj, double *Wj, double *nj);
__device__ double getbjphij(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *nj, double *Delta, double* tmp, double *bj);
__device__ void gettbj(int N, int P, double *Z, double *Psij, double *Wj, double *Omega, double *nj, double* tmp, double *tbj);
__device__ void fill0_d(double* x, int n);
__device__ void fill0_dg(double* x, int n, int ldx);
__device__ void fill1_d(double* x, int n);
__device__ double nk_fdot(double* x, int ldx, double* y, int ldy, int n);
__device__ void nk_fcopy(double cons, double* a, int lda, double* b, int ldb, int n);
__device__ void lbfgs(double* S, double* Y, double* g, int N, int M, int itr, double* Work, double* z);
//__device__ void expandDelta_d(double *Delta, double *delta, int *nh, int p);
__device__ void expandDelta_d(int p, int* nh, double *delta, double* Kmm, double* DeltaInv);
__global__ void expandDelta_k(int p, int* nh, double *delta, double* Kmm, double* DeltaInv);
__global__ void expandDeltaWOK_k(DataFrame *Z, double *delta, double* DeltaInv);
__global__ void printM_k(double *Y, int n, int m, int ldy, int integ);
__device__ void printM_d(double *Y, int n, int m, int ldy, int integ);
__device__ void checkNA(double *x, int n, char *mes);
__device__ void logscale(double *x, int n);
__device__ void scale_d(double *x, int n);
__device__ double quadform(double *A, double *b, int N);

__device__ void mscale_d(int Q, double* rho, double m);

__global__ void expandDelta_cdg0_k(int p, int* nh, double *delta, double* K, double* DeltaInv);
__global__ void expandDelta_cdg_k(int p, int* nh, double *delta, double* K, double* Kta, double* DeltaInv);

__device__ void getZtOinvjnj(SparseMat *Z, double* Wj, double* Omega, double* nj, double* b);
__device__ void bsort(int* a, int* b, double* c, int n);

__global__ void expandDelta_k(DataFrame* Z, double *delta, double* Kmm, double* DeltaInv);
__device__ void getZtOinvjnj(DataFrame *Z, double* Wj, double* Omega, double* nj, double* b);

void readParams(char* fname, int N, int J, int M, int P, int p, int Q, double* Omega, double* delta, double* Xi, double* Ta, double* rho, double* phi,
                double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h, double* phi_h);
void writeParams(char* fname, int N, int J, int M, int P, int p, int Q, double* Omega, double* delta, double* Xi, double* Ta, double* rho, double* phi,
                double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h, double* phi_h);
