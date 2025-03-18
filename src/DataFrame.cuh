typedef struct{
        char* x; // length of L
        int p; // n variables (compatible with n delta parameters)
        int N; // n samples
	int B; // total bytes
        int* c; // n levels of a categorical variable (1=numerical variable)
        int* v; // n subvariables for one variable (1=categorical variable)
    
        int* b; // bytes in the start position of a variable
        int* nh; // column No of a variable in model matrix

        int* mc; // n levels (1=numerical) of a column of model matrix
        int* mv; // XXth level of a column of model matrix
        int* mb; // bytes in the start position of a column of model matrix
        int* mk; // which variable tht column corresponds
    
        int P;

	double* Knm;
	int M;
    
} DataFrame;

__global__ void printModelMatrix(DataFrame* Y, int N, int P);
__device__ double getModelMatrix(DataFrame* Y, int i, int k);
DataFrame* loadDataFrame(char* fname);
DataFrame* loadDataFrame(char* fname, int M);
DataFrame* newDataFrameOnDevice(DataFrame *Z_h);
__global__ void printMetaDataFrame(DataFrame* Y);
void printMetaDataFrameHost(DataFrame* Y);
void printModelMatrixHost(DataFrame* Y, int N, int P);

