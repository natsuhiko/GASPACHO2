__device__ void getRb(double* Lmat, double* b, int n);
__device__ void choldc(double *a, int n);
__device__ void cholsl(double *a, int n, double* b, double* x);
__device__ void cholslg(double *a, int n, double* b, int ldb, double* x, int ldx);
__device__ void forwardslg(double *a, int n, double* x, int ldx);

