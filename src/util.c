#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void printM(double *Y, int n, int m, int ldy, int integ){
	int i, j;
	for(i=0; i<n; i++){
		for(j=0; j<m; j++){
			if(integ==0){
				printf("%lf, ", Y[i+ldy*j]);
			}else{
				printf("%.1lf, ", Y[i+ldy*j]);
			}
		}
		printf("\n");
	}
}

void fill1(double* v, int n){
	int i;
	for(i=0; i<n; i++){
		v[i] = 1.0;
	}
}

void fill0(double* v, int n){
	int i;
	for(i=0; i<n; i++){
		v[i] = 0.0;
	}
}

void expandDelta(double *Delta, double *delta, int *nh, int p){
	int i, j;
	for(i=0; i<p; i++){
		for(j=nh[i]; j<nh[i+1]; j++){
			Delta[j] = delta[i];
		}
	}
}

void initTa(int M, int Q, double* Ta){
	int i, k;
	gsl_rng *r;
	unsigned long seed = 0;
	gsl_rng_env_setup();
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, seed);
	for(k=0; k<Q; k++){
		for(i=0; i<M; i++){
			Ta[i+k*M] = gsl_ran_gaussian(r, 1.0);
		}
	}
	gsl_rng_free(r);
}

void initTaEquispaced(int M, int Q, double* Ta){
	int i, k;
	gsl_rng *r;
	unsigned long seed = 0;
	gsl_rng_env_setup();
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, seed);
	for(i=0; i<M; i++){ Ta[i] = ((double)(i+1)/(double)(M+1))*2.0*3.14159265358979; }
	for(k=1; k<Q; k++){
		for(i=0; i<M; i++){
			Ta[i+k*M] = ((double)(i+1)/(double)(M+1)-0.5)*6.0;
		}
	}
	gsl_rng_free(r);
}

void initTaWithPer(int M, int Q, double* Ta){
	int i, k;
	gsl_rng *r;
	unsigned long seed = 0;
	gsl_rng_env_setup();
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, seed);
	for(i=0; i<M; i++){ Ta[i] = ((double)(i+1)/(double)(M+1))*2.0*3.14159265358979; }
	for(k=1; k<Q; k++){
		for(i=0; i<M; i++){
			Ta[i+k*M] = gsl_ran_gaussian(r, 1.0);
		}
	}
	gsl_rng_free(r);
}

void loadParams(char* fname, int N, int p, int M, int Q, double* Omega_h, double* delta_h, double* Xi_h, double* Ta_h, double* rho_h){
	FILE* f; f=fopen(fname, "rb");
	fread(Omega_h,  sizeof(double), N, f);
	fread(delta_h,  sizeof(double), p, f);
	fread(Xi_h,  sizeof(double), N*Q, f);
	fread(Ta_h,  sizeof(double), M*Q, f);
	fread(rho_h,  sizeof(double), Q, f);
	fclose(f);
}


void scale(double *x, int n){
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

