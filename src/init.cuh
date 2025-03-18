__global__ void initPhi(int J, double* phi);
__global__ void initTitsias(int N, double* titsias);
__global__ void initParams(int N, int J, int P, int p, int Q, double* delta, double* Omega, double* phi, double* rho, double* IdenP);
__global__ void initParamsOld(int N, int J, int P, int p, int* nh, int Q, double* delta, double* DeltaInv, double* Omega, double* phi, double* Kmm, double* rho, double* IdenP);
__global__ void initPsiDense(int N, int J, int P, double* Z, double* Delta, double* Psi); // psi for log linear model
__global__ void initPsi2(int P, double* Psi); // cholesky of psi for log linear model
__global__ void initBetaDense(int N, int J, int P, double* Y, double* ls, double* Z, double *Psi, double* tmp, double *Beta); // log linear model
__global__ void initBeta0(int N, int J, int P, double* Y, double* ls, double *Beta); // E[Y/ls]
__global__ void getNu0Dense(int N, int J, int P, double* Y, double* ls, double* Z, double* Beta, double *Nu); // E[Y/ls]
__global__ void getNu0(SparseMat* Y, double* ls, DataFrame* Z, double* Beta, double *Nu); // E[Y/ls]
__global__ void getNuTNu(int N, int J, double *Nu, double* NuTNu);
__global__ void getX(int N, int J, int M, double *Nu, double* eval, double* evec, double* Z);

__global__ void initPsiSparse(SparseMat* Z, double* DeltaInv, double* Psi);
__global__ void initPsi(DataFrame* Z, double* DeltaInv, double* Psi);
__global__ void initBetaSparse(SparseMat *Y, double* ls, SparseMat *Zt, double *Psi, double* Work, double *Beta);
__global__ void initBeta(SparseMat *Y, double* ls, DataFrame *Zt, double *Psi, double* Work, double *Beta);
