void printM(double *Y, int n, int m, int ldy, int integ);
void fill1(double* v, int n);
void fill0(double* v, int n);
void expandDelta(double *Delta, double *delta, int *nh, int p);
void initTa(int M, int Q, double* Ta);
void initTaWithPer(int M, int Q, double* Ta);
void loadParams(char* fname, int N, int p, int M, int Q, double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h);

void initTaEquispaced(int M, int Q, double* Ta);

void scale(double *x, int n);

