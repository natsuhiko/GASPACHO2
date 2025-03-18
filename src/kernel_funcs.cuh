__global__ void getKernelMatsDense(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* Kmm, double* Kinv, double* KinvKmn);
__global__ void cholKmm(int M, double* Kinv);
__global__ void getKinvKmnDense(int N, int P, int M, double* Kinv, double* Z, double* KinvKmn);
__global__ void printK(int M, double* Kmm);
__device__ void getKernelMatsWithPer(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* K, double* Kinv, double* KinvKmn);
__device__ void getKernelMatsWithPerSpecificLV(int N, int P, int M, int Q, double* X, double* Ta, double* rho, int q, double* Z, double* K, double* Kinv, double* KinvKmn);
__device__ void getKernelMatsOnlyARDSE(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Z, double* K, double* Kinv, double* KinvKmn);

__global__ void getKernelTarget(int N, int M, int Q, double *X, double* Ta, double* rho, double* Knmta, double* Kta, double* Ktainv);
__global__ void getKnmtaKtainvh(int N, int M, double* Ktainv, double* Knmta, double* doxgen);


__global__ void getKernelMatsPer(int N, int P, int M, int Q, double* X, double* Ta, double* rho, double* Knm, double* K, double* Kinv, double* KinvKmn);
__global__ void getKernelMatsSparse(int p, int* nh, int M, int Q, double* X, double* Ta, double* rho, SparseMat *Z, double* K, double* Kinv, double* KinvKmn);
__global__ void getKinvKmnSparse(int p, int* nh, int N, int P, int M, double* Kinv, SparseMat* Z, double* KinvKmn);

__global__ void getKernelMats(int Q, double* Xi, double* Ta, double* rho, DataFrame *Z, double* K, double* Kinv, double* KinvKmn);
__global__ void getKernelMats(int Q, double* Xi, double* Ta, double* rho, DataFrame *Z, int q, double* K, double* Kinv, double* KinvKmn);
__global__ void getKinvKmn(double* Kinv, DataFrame* Z, double* KinvKmn, double* titsias);
