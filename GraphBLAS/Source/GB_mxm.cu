#include "Kernels/spgemm.h"
#include "../Include/GraphBLAS.h"
#include "GB.h"
#include <string.h>
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
    GB_Context Context
)

{
    int A_nrows = A->plen;
    int B_ncols = B->plen;
    
    GrB_Info info ;
    if (A_transpose) {
        GB_transpose(NULL, NULL, false, A, NULL, NULL, NULL, false, Context); // 是否inplace
    } else if (A->is_csc == true) {
        printf("A csc not implemented yet__------------------------------\n");
        return GrB_NO_VALUE;
        // csc => csr 参考 csc2csr的实现
    }
    if (B_transpose) {
	GB_transpose(NULL, NULL, true, B, NULL, NULL, NULL, false, Context); // 是否inplace
    } else if (B->is_csc == false) {
        printf("B csr not implemented yet--------------------------------\n");
        return GrB_NO_VALUE;
        // csr => csc
    }


    TYPE* C_csrVal;
    cudaMalloc(&C_csrVal, M->nzmax * A->type->size);

    int64_t* A_csrRowPtr;   // GPU CSR format
    int64_t* A_csrColInd;
    TYPE* A_csrVal;        // TODO: Need Correct A TYPE
    cudaMalloc(&A_csrRowPtr, (A->plen + 1) * sizeof(int64_t));
    cudaMalloc(&A_csrColInd, A->nzmax * sizeof(int64_t));
    cudaMalloc(&A_csrVal, A->nzmax * A->type->size);

    // Alloc space in GPU and transfer memory from CPU to GPU
    cudaMemcpy(A_csrRowPtr, A->p, (A->plen + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(A_csrColInd, A->i, A->nzmax * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(A_csrVal,    A->x, A->nzmax * A->type->size, cudaMemcpyHostToDevice);


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
    
    const int nt = 1;  // GrB_128
    // printf("the number of thread is: %d\n", nt);

    if (M->is_csc) {

        dim3 NT(nt), NB(ceil(A_nrows/nt));

        spgemmMaskedKernel<<<NB, NT>>>(C_csrVal,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            A_csrRowPtr, 
            A_csrColInd, 
            A_csrVal,
            B_cscColPtr, 
            B_cscRowInd, 
            B_cscVal,
            A_nrows, 
            B_ncols, 
            1, 
            1);
        TYPE* temp = (TYPE*)malloc(M->nzmax * sizeof(TYPE));
            cudaMemcpy(temp, C_csrVal, M->nzmax * A->type->size, cudaMemcpyDeviceToHost);
                    
        C->p = M->p;
        C->i = M->i;
        C->x = (void *) temp;
        C->nzmax = M->nzmax;
        C->is_csc = true;
        
    } else {
        dim3 NT(nt), NB(ceil(A_nrows/nt));
        spgemmMaskedKernel<<<NB, NT>>>(C_csrVal,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            A_csrRowPtr, 
            A_csrColInd, 
            A_csrVal,
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

    return (info) ;
}

