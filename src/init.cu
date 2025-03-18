#include "SparseMat.cuh"
//#include "DataFrame.cuh"
#include "util_cuda.cuh"


__global__ void initTitsias(int N, double* titsias){
        int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
        if(j==0){
                for(j=0; j<N; j++){
                        titsias[j] = 0.0;
                }
        }
}



__global__ void initPhi(int J, double* phi){
        int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
        if(j==0){
                for(j=0; j<J; j++){
                        phi[j] = 1.0;
                }
        }
}


__global__ void initParams(int N, int J, int P, int p, int Q, double* delta, double* Omega, double* phi, double* rho, double* IdenP){
        int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
        if(j==0){
                int i, j, k;
                delta[0] = 1.5;
		for(i=1; i<p-1; i++){
			delta[i] = 0.01;
		}
                delta[p-1] = .3; // theta for Kmm
                for(i=0; i<N; i++){
                        Omega[i] = 1.0;
                }
                for(j=0; j<J; j++){
                        phi[j] = 1.0;
                }
		rho[0] = 5.;
		for(k=1; k<Q; k++){rho[k]= 10.; } //((double)k)*5+3.;}

		fill0_d(IdenP,P*P);
		for(i=0; i<P; i++){IdenP[i+i*P]=1.0;}
        }
}


__global__ void initParamsOld(int N, int J, int P, int p, int* nh, int Q, double* delta, double* DeltaInv, double* Omega, double* phi, double* Kmm, double* rho, double* IdenP){
        int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
        if(j==0){
                int i, j, k;
                delta[0] = 1.0;
                delta[1] = 0.1; // batch/cond/dox
                delta[2] = 0.05; // mt
                delta[3] = 0.05; // ribo
                delta[4] = 0.5; // theta for Kmm
                //expandDelta_d(p, nh, delta, Kmm, DeltaInv);
                for(i=0; i<N; i++){
                        Omega[i] = 1.0;
                }
                for(j=0; j<J; j++){
                        phi[j] = 1.0;
                }
		rho[0] = 0.5;
		for(k=1; k<Q; k++){rho[k]=((double)k)+3.;}

		fill0_d(IdenP,P*P);
		for(i=0; i<P; i++){IdenP[i+i*P]=1.0;}
        }
}

__global__ void initPsiDense(int N, int J, int P, double* Z, double* DeltaInv, double* Psi){ // psi for log linear model
    int j = blockIdx.x;
    int k = blockIdx.y;
    if((j<=k && j<P) && k<P){
        double sum = 0.0;
        int i;
        for(i=0; i<N; i++){
            sum += Z[i+j*N]*Z[i+k*N];
        }
        //if(j==k){ sum += 1.0/Delta[j]; }
        Psi[j+k*P] = sum + DeltaInv[j+k*P];
    }else{
        Psi[j+k*P] = 0.0;
    }
}


__global__ void initPsiSparse(SparseMat* Z, double* DeltaInv, double* Psi){ // psi for log linear model
    int j = blockIdx.x;
    int k = blockIdx.y;
    int N = Z->N;
    int P = Z->J;
    if((j<=k && j<P) && k<P){
        double sum = 0.0;
        int i;
	int l1=Z->p[j],      l2=Z->p[k];
        int l1max=Z->p[j+1], l2max=Z->p[k+1];
        int i1, i2;
        for(i=0; i<N; i++){
                    if(l1==Z->L || l2==Z->L){break;}
                    i1 = Z->i[l1];
                    i2 = Z->i[l2];
                    if(i1==i && l1<(l1max)){
                                if(i2==i && l2<(l2max)){
                                        sum += Z->x[l1]*Z->x[l2];
                                        l2++;
                                }
                                l1++;
                    }else{
                                if(i2==i && l2<(l2max)){
                                        l2++;
                                }
                    }
        }
        Psi[j+k*P] = sum + DeltaInv[j+k*P];
    }else{
        Psi[j+k*P] = 0.0;
    }
}

__global__ void initPsi(DataFrame* Z, double* DeltaInv, double* Psi){
	int k = blockIdx.x;
	int l = blockIdx.y;
	int P = Z->P;
	double sum = 0.0;
	if(k<=l && l<P){
            if(Z->mc[k]==1){ // num 1
                double* pd1; pd1 = (double*)(Z->x + Z->mb[k]);
                if(Z->mc[l]==1){// num 2
                    double* pd2; pd2 = (double*)(Z->x + Z->mb[l]);
                    for(int i=0; i<Z->N; i++){
                        sum += pd1[i]*pd2[i];
                    }
                }else{// cate 2
                    int* pi2; pi2 = (int*)(Z->x + Z->mb[l]);
                    int mvl = Z->mv[l];
                    for(int i=0; i<Z->N; i++){
                        if(pi2[i]==mvl){
                            sum += pd1[i];
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
                            sum += pd2[i];
                        }
                    }
                }else{// cate 2
                    int* pi2; pi2 = (int*)(Z->x + Z->mb[l]);
                    int mvl = Z->mv[l];
                    if(Z->mk[k]==Z->mk[l]){// same var
                        if(k==l){
                            for(int i=0; i<Z->N; i++){
                                if(pi1[i]==mvk){
                                    sum += 1.0;
                                }
                            }
                        }
                    }else{ // diff vars
                        for(int i=0; i<Z->N; i++){
                            if(pi1[i]==mvk && pi2[i]==mvl){
                                sum += 1.0;
                            }
                        }
                    }
                }
            }
	    Psi[k+l*P] = sum + DeltaInv[k+l*P];
	}else{
	    Psi[k+l*P] = 0.0;
	}
}



__global__ void initPsi2(int P, double* Psi){ // cholesky of psi for log linear model
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j==0){
	//printM_d(Psi,P,P,P,0);
        choldc(Psi,P);
    }
}

__global__ void initBeta(SparseMat *Y, double* ls, DataFrame* Z, double *Psi, double* Work, double *Beta){ // log linear model
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<(Y->J)){
	int P = Z->P;
        //double* tmp; tmp=(double*)malloc(sizeof(double)*P); 
	double* tmp; tmp = Work+j*P;
	fill0_d(tmp,P);
        int i, m;
	int l = Y->p[j];
	int lmax = Y->p[j+1];
	int N=Y->N;
	double yi;
        for(i=0; i<N; i++){
	    if(i==Y->i[l] && l<lmax){
                yi = log(((Y->x[l])+0.05)/ls[i]);
		l++;
	    }else{
		yi = log(0.05/ls[i]);
	    }
            for(m=0; m<P; m++){
                tmp[m] += getModelMatrix(Z,i,m)*yi;
            }
        }
	//if(j==0)printM_d(tmp,1,P,1,0);
        cholsl(Psi, P, tmp, Beta+j*P);
	if(j==242)printM_d(Beta+j*P,1,P,1,0);
        //free(tmp);
    }
}


__global__ void initBetaSparse(SparseMat *Y, double* ls, SparseMat* Zt, double *Psi, double* Work, double *Beta){ // log linear model
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<(Y->J)){
	int P = Zt->N;
        //double* tmp; tmp=(double*)malloc(sizeof(double)*P); 
	double* tmp; tmp = Work+j*P;
	fill0_d(tmp,P);
        int i, k, m;
	int l = Y->p[j];
	int lmax = Y->p[j+1];
	int N=Y->N;
	double yi;
        for(i=0; i<N; i++){
	    if(i==Y->i[l] && l<lmax){
                yi = log(((Y->x[l])+0.5)/ls[i]);
		l++;
	    }else{
		yi = log(0.5/ls[i]);
	    }
            for(m=Zt->p[i]; m<Zt->p[i+1]; m++){
		k = Zt->i[m];
                tmp[k] += Zt->x[m]*yi;
            }
        }
	//if(j==1577)printM_d(tmp,1,P,1,0);
        cholsl(Psi, P, tmp, Beta+j*P);
	//if(j==1577)printM_d(Beta+j*P,1,P,1,0);
        //free(tmp);
    }
}


__global__ void initBetaDense(int N, int J, int P, double* Y, double* ls, double* Z, double *Psi, double* Work, double *Beta){ // log linear model
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<J){
        //double* tmp; tmp=(double*)malloc(sizeof(double)*P); 
	double* tmp; tmp = Work+j*P;
	fill0_d(tmp,P);
        int i, k;
        for(i=0; i<N; i++){
            double yi = log((Y[i+j*N]+0.5)/ls[i]);
            for(k=0; k<P; k++){
                tmp[k] += Z[i+k*N]*yi;
            }
        }
	//if(j==1577)printM_d(tmp,1,P,1,0);
        cholsl(Psi, P, tmp, Beta+j*P);
	//if(j==1577)printM_d(Beta+j*P,1,P,1,0);
        //free(tmp);
    }
}

__global__ void initBeta0(int N, int J, int P, double* Y, double* ls, double *Beta){ // E[Y/ls]
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<J){
        double sum = 0.0;
        int i;
        for(i=0; i<N; i++){
            sum += Y[i+j*N]/ls[i];
        }
        Beta[0+j*P] = log(sum/((double)N));
    }
}

__global__ void getNu0Dense(int N, int J, int P, double* Y, double* ls, double* Z, double* Beta, double *Nu){ // E[Y/ls]
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<J){
        int i, k;
        for(i=0; i<N; i++){
            double yi = log((Y[i+j*N]+0.5)/ls[i]);
            for(k=0; k<P; k++){
                yi -= Z[i+k*N]*Beta[k+j*P];
            }
            Nu[i+j*N] = yi;
	    //if(j==10 && i<10){printf("%lf %lf %lf \n",Y[i+j*N], ls[i], Nu[i+j*N]);}
        }
    }
}


__global__ void getNu0(SparseMat* Y, double* ls, DataFrame* Z, double* Beta, double *Nu){ // E[Y/ls]
    int j = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(j<Y->J){
	int P=Z->P;
	int N=Y->N;
        int i, k;
	int l = Y->p[j];
	int lmax = Y->p[j+1];
	double yi;
        for(i=0; i<N; i++){
            if(Y->i[l]==i && l<lmax){
                yi = log((Y->x[l]+0.5)/ls[i]);
	    	l++;
	    }else{
		yi = log((0.5)/ls[i]);
	    }
            for(k=0; k<P; k++){
                yi -= getModelMatrix(Z,i,k)*Beta[k+j*P];
            }
            Nu[i+j*N] = yi;
	    //if(j==10 && i<10){printf("%lf %lf %lf \n",Y[i+j*N], ls[i], Nu[i+j*N]);}
        }
    }
}

__global__ void getNuTNu(int N, int J, double *Nu, double* NuTNu){
    int j = blockIdx.x;
    int k = blockIdx.y;
    if(j<=k){
        int i;
        double sum = 0.0;
        for(i=0; i<N; i++){
            sum += Nu[i+j*N]*Nu[i+k*N];
        }
        NuTNu[j+k*J] = sum;
    }
}


__global__ void getX(int N, int J, int M, double *Nu, double* eval, double* evec, double* X){
    int i = (blockDim.x*blockIdx.x+threadIdx.x)*gridDim.y*blockDim.y+(blockDim.y*blockIdx.y+threadIdx.y);
    if(i<N){
        int j, k;
        //int pX = nh[p]-nh[p-1];
        //double* X; X = Z+nh[p-1]*N;
        for(k=0; k<M; k++){
            double sum = 0.0;
            for(j=0; j<J; j++){
                sum += Nu[i+j*N]*evec[j+k*J];
            }
            X[i+(M-k-1)*N] = sum/sqrt(eval[k])*sqrt((double)N);
        }
    }
}



