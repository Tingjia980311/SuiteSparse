//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/tri_demo.c: count triangles
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Read a graph from a file and count the # of triangles using two methods.
// Usage:
//
//  tri_demo   < infile
//  tri_demo 1 < infile
//  tri_demo 0 nrows ncols ntuples method
//  tri_demo 1 nx ny method
//
// Where infile has one line per edge in the graph; these have the form
//
//  i j x
//
// where A(i,j)=x is performed by GrB_Matrix_build, to construct the matrix.
// The default file format is 0-based, but with "tri_demo 1 < infile" the
// matrix is assumed to be 1-based.

// The dimensions of A are assumed to be the largest row and column indices,
// plus one if the matrix is 1-based.  This is done in read_matrix.c.
//
// For the second usage (tri_demo 0 ...), a random symmetric matrix is created
// of size nrows-by-ncols with ntuples edges (some will be duplicates so the
// actual number of edges will be slightly less).  The method is 0 for
// setElement and 1 for build.  The matrix will not have any self-edges, which
// cause the tricount method to fail.
//
// The 3rd usage (tri_demo 1 ...) creates a finite-element matrix on an
// nx-by-ny grid.  Method is 0 to 3; refer to wathen.c for details.

// macro used by OK(...) to free workspace if an error occurs
#define FREE_ALL                    \
    GxB_Scalar_free (&Thunk) ;      \
    GrB_Matrix_free (&C) ;          \
    GrB_Matrix_free (&A) ;          \
    GrB_Matrix_free (&L) ;          \
    GrB_Matrix_free (&U) ;

#include "graphblas_demos.h"

int main (int argc, char **argv)
{
    GrB_Matrix C = NULL, A = NULL, L = NULL, U = NULL ;
    GxB_Scalar Thunk = NULL ;
    GrB_Info info ;
    double tic [2], r1, r2 ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "tri_demo: nthreads %d\n", nthreads) ;
    printf ("--------------------------------------------------------------\n");

    //--------------------------------------------------------------------------
    // get a symmetric matrix with no self edges
    //--------------------------------------------------------------------------

    // get_matrix reads in a boolean matrix.  It could easily be changed to
    // read in int32 matrix instead, but this would affect the other GraphBLAS
    // demos.  So the time to typecast A = (int32) C is added to the read
    // time, not the prep time for triangle counting.
    simple_tic (tic) ;
    OK (get_matrix (&C, argc, argv, true, true)) ;
    GrB_Index n, nedges ;
    OK (GrB_Matrix_nrows (&n, C)) ;

    GrB_Type ctype = GrB_INT32 ;

    // A = spones (C), and typecast to int32
    OK (GrB_Matrix_new (&A, ctype, n, n)) ;
    OK (GrB_Matrix_apply (A, NULL, NULL, GxB_ONE_INT32, C, NULL)) ;
    double t_read = simple_toc (tic) ;
    printf ("\ntotal time to read A matrix: %14.6f sec\n", t_read) ;
    GrB_Matrix_free (&C) ;

    OK (GxB_Scalar_new (&Thunk, GrB_INT64)) ;

    // U = triu (A,1)
    simple_tic (tic) ;
    OK (GxB_Scalar_setElement_INT64 (Thunk, (int64_t) 1)) ;
    OK (GrB_Matrix_new (&U, ctype, n, n)) ;
    OK (GxB_Matrix_select (U, NULL, NULL, GxB_TRIU, A, Thunk, NULL)) ;
    OK (GrB_Matrix_nvals (&nedges, U)) ;
    printf ("\nn %.16g # edges %.16g\n", (double) n, (double) nedges) ;
    double t_U = simple_toc (tic) ;
    printf ("U=triu(A) time:  %14.6f sec\n", t_U) ;

    // L = tril (A,-1)
    simple_tic (tic) ;
    OK (GrB_Matrix_new (&L, ctype, n, n)) ;
    OK (GxB_Scalar_setElement_INT64 (Thunk, (int64_t) (-1))) ;
    OK (GxB_Matrix_select (L, NULL, NULL, GxB_TRIL, A, Thunk, NULL)) ;
    double t_L = simple_toc (tic) ;
    printf ("L=tril(A) time:  %14.6f sec\n", t_L) ;
    OK (GrB_Matrix_free (&A)) ;

    int nthreads_max = 1 ;
    #if defined ( _OPENMP )
    nthreads_max = omp_get_max_threads ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // count the triangles via C<L> = L*U' (dot-produt)
    //--------------------------------------------------------------------------

    printf ("\n------------------------------------- dot product method:\n") ;

    #define NTHREADS_MAX 2048
    nthreads_max = MIN (nthreads_max, NTHREADS_MAX) ;

    int64_t ntri2 [NTHREADS_MAX+1], nt = -1 ;
    double t1 ;

    for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads *= 2)
    {
        GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads) ;

        double t_dot [2] ;
        OK (tricount (&(ntri2 [nthreads]), 5, NULL, NULL, L, U, t_dot)) ;

        if (nthreads == 1)
        {
            printf ("# triangles %.16g\n", (double) ntri2 [nthreads]) ;
            fprintf (stderr, "# triangles %.16g\n", (double) ntri2 [nthreads]) ;
            nt = ntri2 [1] ;
        }
        if (ntri2 [nthreads] != nt)
        {
            printf ("error 5!\n") ;
            fprintf (stderr, "error!\n") ;
            exit (1) ;
        }

        printf ("L*U' time (dot):   %14.6f sec", t_dot [0]) ;
        if (nthreads == 1)
        {
            t1 = t_dot [0] ;
        }
        else
        {
            printf (" (nthreads: %d speedup %g)", nthreads, t1 / t_dot [0]) ;
        }

        printf ("\ntricount time:   %14.6f sec (dot product method)\n",
            t_dot [0] + t_dot [1]) ;
        printf ("tri+prep time:   %14.6f sec (incl time to compute L and U)\n",
            t_dot [0] + t_dot [1] + t_U + t_L) ;

        printf ("compute C time:  %14.6f sec\n", t_dot [0]) ;
        printf ("reduce (C) time: %14.6f sec\n", t_dot [1]) ;

        r1 = 1e-6*nedges / (t_dot [0] + t_dot [1] + t_U + t_L) ;
        r2 = 1e-6*nedges / (t_dot [0] + t_dot [1]) ;
        printf ("rate %8.2f million edges/sec (incl time for U=triu(A))\n",r1);
        printf ("rate %8.2f million edges/sec (just tricount itself)\n", r2);
        fprintf (stderr, "GrB: C<L>=L*U' (dot)   "
                "rate %8.2f (w/ prep), %8.2f (tri)", r1, r2) ;
        if (nthreads > 1) fprintf (stderr, " speedup: %6.2f", t1/ t_dot [0]) ;
        fprintf (stderr, "\n") ;
    }
    if (nthreads_max > 1) fprintf (stderr, "\n") ;

    //--------------------------------------------------------------------------
    // method 6:  C<U> = U*L' (dot)
    //--------------------------------------------------------------------------

//     for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads *= 2)
//     {
//         GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads) ;

//         double t_dot [2] ;
//         OK (tricount (&(ntri2 [nthreads]), 6, NULL, NULL, L, U, t_dot)) ;

// //      printf ("# triangles %.16g\n", (double) ntri2 [nthreads]) ;
// //      fprintf (stderr, "# triangles %.16g\n", (double) ntri2 [nthreads]) ;
//         if (ntri2 [nthreads] != nt)
//         {
//             printf ("error 6!\n") ;
//             fprintf (stderr, "error!\n") ;
//             exit (1) ;
//         }

//         printf ("L*U' time (dot):   %14.6f sec", t_dot [0]) ;
//         if (nthreads == 1)
//         {
//             t1 = t_dot [0] ;
//         }
//         else
//         {
//             printf (" (nthreads: %d speedup %g)", nthreads, t1 / t_dot [0]) ;
//         }

//         printf ("\ntricount time:   %14.6f sec (dot product method)\n",
//             t_dot [0] + t_dot [1]) ;
//         printf ("tri+prep time:   %14.6f sec (incl time to compute L and U)\n",
//             t_dot [0] + t_dot [1] + t_U + t_L) ;

//         printf ("compute C time:  %14.6f sec\n", t_dot [0]) ;
//         printf ("reduce (C) time: %14.6f sec\n", t_dot [1]) ;

//         r1 = 1e-6*nedges / (t_dot [0] + t_dot [1] + t_U + t_L) ;
//         r2 = 1e-6*nedges / (t_dot [0] + t_dot [1]) ;
//         printf ("rate %8.2f million edges/sec (incl time for U=triu(A))\n",r1);
//         printf ("rate %8.2f million edges/sec (just tricount itself)\n", r2);
//         fprintf (stderr, "GrB: C<U>=U*L' (dot)   "
//                 "rate %8.2f (w/ prep), %8.2f (tri)", r1, r2) ;
//         if (nthreads > 1) fprintf (stderr, " speedup: %6.2f", t1/ t_dot [0]) ;
//         fprintf (stderr, "\n") ;
//     }
//     if (nthreads_max > 1) fprintf (stderr, "\n") ;

//     //--------------------------------------------------------------------------
//     // count the triangles via C<L> = L*L (saxpy)
//     //--------------------------------------------------------------------------

//     printf ("\n----------------------------------- saxpy method:\n") ;

//     int64_t ntri1 [NTHREADS_MAX+1] ;

//     for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads *= 2)
//     {
//         GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads) ;

//         double t_mark [2] = { 0, 0 } ;
//         OK (tricount (&ntri1 [nthreads], 3, NULL, NULL, L, NULL, t_mark)) ;
//         printf ("triangles, method 3: %0.16g\n", (double) ntri1 [nthreads]) ;
//         if (ntri1 [nthreads] != nt)
//         {
//             printf ("error 3!\n") ;
//             fprintf (stderr, "error!\n") ;
//             exit (1) ;
//         }

//         printf ("C<L>=L*L time (saxpy):   %14.6f sec", t_mark [0]) ;
//         if (nthreads == 1)
//         {
//             t1 = t_mark [0] ;
//         }
//         else
//         {
//             printf (" (nthreads: %d speedup %g)", nthreads, t1 / t_mark [0]) ;
//         }

//         printf ("\ntricount time:   %14.6f sec (saxpy method)\n",
//             t_mark [0] + t_mark [1]) ;
//         printf ("tri+prep time:   %14.6f sec (incl time to compute L)\n",
//             t_mark [0] + t_mark [1] + t_L) ;

//         printf ("compute C time:  %14.6f sec\n", t_mark [0]) ;
//         printf ("reduce (C) time: %14.6f sec\n", t_mark [1]) ;

//         r1 = 1e-6*((double)nedges) / (t_mark [0] + t_mark [1] + t_L) ;
//         r2 = 1e-6*((double)nedges) / (t_mark [0] + t_mark [1]) ;
//         printf ("rate %8.2f million edges/sec (incl time for L=tril(A))\n",r1);
//         printf ("rate %8.2f million edges/sec (just tricount itself)\n", r2);
//         fprintf (stderr, "GrB: C<L>=L*L (saxpy)  "
//                 "rate %8.2f (w/ prep), %8.2f (tri)", r1, r2) ;
//         if (nthreads > 1) fprintf (stderr, " speedup: %6.2f", t1/ t_mark [0]) ;
//         fprintf (stderr, "\n") ;
//     }

//     //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    // FREE_ALL ;
    GrB_finalize ( ) ;
    printf ("\n") ;
    fprintf (stderr, "\n") ;
}


