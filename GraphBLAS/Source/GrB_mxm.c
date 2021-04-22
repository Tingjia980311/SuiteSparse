#include "GB_mxm.h"

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

static void print_info(GrB_Matrix A, char name){
    char ap[] = "A->p";
    char ai[] = "A->i";
    char ax[] = "A->x";
    ap[0] = name;
    ai[0] = name;
    ax[0] = name;
    print_long(A->p, A->plen+1, ap);
    print_long(A->i, A->nzmax, ai);
    bool *Ax = (bool *) (A->x);
    print_int(Ax, A->nzmax, ax);
}


extern GrB_Info GB_mxm_gpu 
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
    int64_t ** A_csrColInd,
    bool ** A_csrVal,
    // const int64_t *  A_csrColInd,
    // const TYPE * A_csrVal,
    int if_valid_A,
    GB_Context Context
);

extern GrB_Info GrB_vxm_
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for matrix multiply
    const GrB_Vector u,             // first input:  vector u
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc,       // descriptor for w, M, and A
    int64_t ** A_csrRowPtr,
    int64_t ** A_csrColInd,
    bool ** A_csrVal
);

GrB_Info GrB_mxm
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for T=A*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B,
)
{
    fprintf (stderr, "mxm running\n") ;
    GB_WHERE ("GrB_mxm (C, M, accum, semiring, A, B, desc)") ;
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
                       A_transpose, B_transpose, AxB_method);
    info = GB_mxm (C, C_replace,
                       M, Mask_comp, Mask_struct,
                       accum,
                       semiring,
                       A, A_transpose,
                       B, B_transpose,
                       false,
                       AxB_method,
                       Context) ;
    printf("Calling ended\n");
    return (info);
}



GrB_Info GrB_vxm_                    // w'<M> = accum (w, u'*A)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for matrix multiply
    const GrB_Vector u,             // first input:  vector u
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc,       // descriptor for w, M, and A
    int64_t ** A_csrRowPtr,
    int64_t ** A_csrColInd,
    bool ** A_csrVal
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_vxm_ (w, M, accum, semiring, u, A, desc)") ;
    GB_BURBLE_START ("GrB_vxm_") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M == NULL || GB_VECTOR_OK (M)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx, A_transpose, AxB_method) ;

    //--------------------------------------------------------------------------
    // w'<M'> = accum (w',u'*A) and variations, using the mxm kernel
    //--------------------------------------------------------------------------

    // w, M, and u are passed as matrices to GB_mxm
    // A and u are swapped, and A_transpose is negated:
    //      u'*A  == A'*u
    //      u'*A' == A*u
    // Since A and u are swapped, in all the matrix multiply kernels
    // fmult(y,x) must be used instead of fmult(x,y).

    // int64_t *A_csrRowPtr;
    // int64_t *A_csrColInd;
    // bool * A_csrVal;

    // cudaMalloc(&A_csrRowPtr, (A->plen + 1) * sizeof(int64_t));
    // cudaMemcpy(A_csrRowPtr,  A->p, (A->plen + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    // cudaMalloc(&A_csrColInd, A->nzmax * sizeof(int64_t));
    // cudaMemcpy(A_csrColInd, A->i, A->nzmax * sizeof(int64_t), cudaMemcpyHostToDevice);

    // cudaMalloc(&A_csrVal, A->nzmax * A->type->size);
    // cudaMemcpy(A_csrVal, A->x, A->nzmax * A->type->size, cudaMemcpyHostToDevice);

    // print_info(A,'A');
    // print_info(u, 'u');
    // print_info(M, 'M');
     // stack
    // int64_t * A_csrColInd;
    // bool *A_csrVal;

    info = GB_mxm_gpu (
        (GrB_Matrix) w,     C_replace,      // w and its descriptor
        (GrB_Matrix) M, Mask_comp, Mask_struct, // mask and its descriptor
        accum,                              // for accum (w,t)
        semiring,                           // definition of matrix multiply
        A,                  !A_transpose,   // allow A to be transposed
        (GrB_Matrix) u,     false,          // u is never transposed
        true,                               // flipxy: fmult(y,x)
        AxB_method,                         // algorithm selector
        A_csrRowPtr,
        A_csrColInd,
        A_csrVal,
	    0,
        Context) ;
    // print_info(w, 'w')

    GB_BURBLE_END ;
    return (info) ;
}
