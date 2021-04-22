#include "Kernels/spgemm.h"
#include "../Include/GraphBLAS.h"
#include "GB.h"
#include <string.h>
#include <iostream>
#include <cstdio>
#include "simple_timer.h"
using namespace std;


GB_PUBLIC
void simple_tic         // returns current time in seconds and nanoseconds
(
    double tic [2]      // tic [0]: seconds, tic [1]: nanoseconds
)
{

    #if defined ( _OPENMP )

        // OpenMP is available; use the OpenMP timer function
        tic [0] = omp_get_wtime ( ) ;
        tic [1] = 0 ;

    #elif defined ( __linux__ ) || defined ( __GNU__ )

        // Linux has a very low resolution clock() function, so use the high
        // resolution clock_gettime instead.  May require -lrt
        struct timespec t ;
        clock_gettime (CLOCK_MONOTONIC, &t) ;
        tic [0] = (double) t.tv_sec ;
        tic [1] = (double) t.tv_nsec ;

    #elif defined ( __MACH__ ) && defined ( __APPLE__ )

        // otherwise, on the Mac, use the MACH timer
        clock_serv_t cclock ;
        mach_timespec_t t ;
        host_get_clock_service (mach_host_self ( ), SYSTEM_CLOCK, &cclock) ;
        clock_get_time (cclock, &t) ;
        mach_port_deallocate (mach_task_self ( ), cclock) ;
        tic [0] = (double) t.tv_sec;
        tic [1] = (double) t.tv_nsec;

    #else

        // The ANSI C11 clock() function is used instead.  This gives the
        // processor time, not the wallclock time, and it might have low
        // resolution.  It returns the time since some unspecified fixed time
        // in the past, as a clock_t integer.  The clock ticks per second are
        // given by CLOCKS_PER_SEC.  In Mac OSX this is a very high resolution
        // clock, and clock ( ) is faster than clock_get_time (...) ;
        clock_t t = clock ( ) ;
        tic [0] = ((double) t) / ((double) CLOCKS_PER_SEC) ;
        tic [1] = 0 ;

    #endif

}

//------------------------------------------------------------------------------
// simple_toc: return the time since the last simple_tic
//------------------------------------------------------------------------------

GB_PUBLIC
double simple_toc           // returns time since last simple_tic
(
    const double tic [2]    // tic from last call to simple_tic
)
{
    double toc [2] ;
    simple_tic (toc) ;
    return ((toc [0] - tic [0]) + 1e-9 * (toc [1] - tic [1])) ;
}



#define print_int(ptr, n, name) \
    printf("%s: ", name); \
    for(int i = 0; i<n; i++) { \
        printf("\t%d  ", ptr[i]); \
    } \
    printf("\n");

#define print_long(ptr, n, name) \
    printf("%s: ", name); \
    for(int i = 0; i<n; i++) { \
        printf("\t%ld  ", ptr[i]); \
    } \
    printf("\n");
#define CUDA_WARN(XXX) \
do { if (XXX != cudaSuccess) cerr << "CUDA Error: " << \
    cudaGetErrorString(XXX) << ", at line " << __LINE__ \
    << endl; cudaDeviceSynchronize(); } while (0)
void print_info(GrB_Matrix A, char name){
    char ap[] = "A->p";
    char ai[] = "A->i";
    char ax[] = "A->x";
    ap[0] = name;
    ai[0] = name;
    ax[0] = name;
    print_long(A->p, A->plen+1, ap);
    print_long(A->i, A->nzmax, ai);
    bool *Ax = static_cast<bool*>(A->x);
    print_int(Ax, A->nzmax, ax);
}

typedef bool TYPE;

extern "C" GrB_Info GB_transpose           // C=A', C=(ctype)A or C=op(A')
(
    GrB_Matrix *Chandle,            // output matrix C, possibly modified in place
    GrB_Type ctype,                 // desired type of C; if NULL use A->type.
                                    // ignored if op is present (cast to op->ztype)
    const bool C_is_csc,            // desired CSR/CSC format of C
    const GrB_Matrix A_in,          // input matrix
    // no operator is applied if both op1 and op2 are NULL
    const GrB_UnaryOp op1_in,       // unary operator to apply
    const GrB_BinaryOp op2_in,      // binary operator to apply
    const GxB_Scalar scalar,        // scalar to bind to binary operator
    bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    GB_Context Context
 );


extern "C" GrB_Info GB_mxm_gpu
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for C=A*B
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    const bool B_transpose,         // if true, use B' instead of B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const GrB_Desc_Value AxB_method,// for auto vs user selection of methods
    int64_t ** A_csrRowPtr,
    int64_t **  A_csrColInd,
    TYPE ** A_csrVal,
    // const TYPE * A_csrVal,
    int if_valid_A,
    GB_Context Context
)

{
    double tic [2], t1, t2, t3, t4, t5 ;
    simple_tic (tic) ;
    int A_nrows = A->plen;
    int B_ncols = B->plen;
    
    GrB_Info info ;
    // if (A_transpose) {
    //     GB_transpose(NULL, NULL, false, A, NULL, NULL, NULL, false, Context); // 是否inplace
    // } else if (A->is_csc == true) {
    //     printf("A csc not implemented yet__------------------------------\n");
    //     return GrB_NO_VALUE;
    //     // csc => csr 参考 csc2csr的实现
    // }
    // if (B_transpose) {
	// GB_transpose(NULL, NULL, true, B, NULL, NULL, NULL, false, Context); // 是否inplace
    // } else if (B->is_csc == false) {
    //     printf("B csr not implemented yet--------------------------------\n");
    //     return GrB_NO_VALUE;
    //     // csr => csc
    // }
    t1 = simple_toc (tic) ;

    simple_tic (tic) ;

    TYPE* C_csrVal;
    cudaMalloc(&C_csrVal, M->nzmax * A->type->size);

    if (*A_csrRowPtr==NULL) {
        // printf("ship A_csrrowptr from gpu to cpu\n");
        cudaMalloc(A_csrRowPtr, (A->plen + 1) * sizeof(int64_t));
        cudaMemcpy(*A_csrRowPtr,  A->p, (A->plen + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    }
    

    if (*A_csrColInd==NULL) {
        cudaMalloc(A_csrColInd, A->nzmax * sizeof(int64_t));
        cudaMemcpy(*A_csrColInd, A->i, A->nzmax * sizeof(int64_t), cudaMemcpyHostToDevice);
    }

    if (*A_csrVal == NULL) {
        cudaMalloc(A_csrVal, A->nzmax * A->type->size);
        cudaMemcpy(*A_csrVal, A->x, A->nzmax * A->type->size, cudaMemcpyHostToDevice);
    }

    t2 = simple_toc (tic) ;

    simple_tic (tic) ;

    int64_t* B_cscColPtr;   // GPU CSR format
    int64_t* B_cscRowInd;
    TYPE* B_cscVal;        // TODO: Need Correct A TYPE
    cudaMalloc(&B_cscColPtr, (B->plen + 1) * sizeof(int64_t));
    cudaMalloc(&B_cscRowInd, B->nzmax * sizeof(int64_t));
    cudaMalloc(&B_cscVal, B->nzmax * A->type->size);

    // Alloc space in GPU and transfer memory from CPU to GPU
    cudaMemcpy(B_cscColPtr, B->p, (B->plen + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B_cscRowInd, B->i, B->nzmax * sizeof (int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B_cscVal,    B->x, B->nzmax * A->type->size, cudaMemcpyHostToDevice);

    int64_t* M_csrRowPtr;       // GPU CSR format
    int64_t* M_csrColInd;
    int32_t* M_csrVal;             // TODO: Need Correct A TYPE

    cudaMalloc(&M_csrRowPtr, (M->plen + 1) * sizeof(int64_t));
    cudaMalloc(&M_csrColInd, M->nzmax * sizeof(int64_t));
    cudaMalloc(&M_csrVal, M->nzmax * M->type->size);

    cudaMemcpy(M_csrRowPtr, M->p, (M->plen + 1) * sizeof (int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(M_csrColInd, M->i, M->nzmax * sizeof (int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(M_csrVal,    M->x, M->nzmax * M->type->size, cudaMemcpyHostToDevice);

    t3 = simple_toc (tic) ;

    simple_tic (tic) ;
    
    const int nt = 128;  // GrB_128
    // printf("the number of thread is: %d\n", nt);

    if (M->is_csc) {
        
        dim3 NT(nt), NB(ceil(float(A_nrows)/nt));

        spgemmMaskedKernel<<<NB, NT>>>(C_csrVal,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            *A_csrRowPtr, 
            *A_csrColInd, 
            *A_csrVal,
            B_cscColPtr, 
            B_cscRowInd, 
            B_cscVal,
            A_nrows, 
            B_ncols, 
            1, 
            1);
        t4 = simple_toc (tic) ;
        simple_tic(tic);
        TYPE* temp = (TYPE*)malloc(M->nzmax * sizeof(TYPE));
        cudaMemcpy(temp, C_csrVal, M->nzmax * A->type->size, cudaMemcpyDeviceToHost);
                    
        C->p = M->p;
        C->i = M->i;
        C->x = (void *) temp;
        C->nzmax = M->nzmax;
        C->is_csc = true;
        t5 = simple_toc (tic) ;
    } else {
        dim3 NT(nt), NB(ceil(float(A_nrows)/nt));

        spgemmMaskedKernel<<<NB, NT>>>(
            C_csrVal,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            *A_csrRowPtr, 
            *A_csrColInd, 
            *A_csrVal,
            B_cscColPtr, 
            B_cscRowInd, 
            B_cscVal,
            A_nrows, 
            B_ncols, 
            1, 
            0);
        TYPE* temp = (TYPE*)malloc(M->nzmax * sizeof(TYPE));
        cudaMemcpy(temp, C_csrVal, M->nzmax * A->type->size, cudaMemcpyDeviceToHost);
                    
        C->p = M->p;
        C->i = M->i;
        C->x = (void *) temp;
        C->nzmax = M->nzmax;
        C->is_csc = false;
    }
    printf ("load time in seconds:                %14.6f\n", t1) ;
    printf ("ship A time in seconds:              %14.6f\n", t2) ;
    printf ("ship other vectors time in seconds:  %14.6f\n", t3) ;
    printf ("kernel function time in seconds:     %14.6f\n", t4) ;
    printf ("ship results back time in seconds:   %14.6f\n", t5) ;
    
    return (info) ;
}
