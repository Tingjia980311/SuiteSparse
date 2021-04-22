#include <stdio.h>
#include "../../Include/GraphBLAS.h"
#include "utils.h"

// Sparse matrix-Sparse matrix multiplication with sparse matrix mask
// Strategy:
// 1) Loop through mask using 1 warp/row
// 2) For each nonzero (row, col) of mask:
//    i)   initialize each thread to identity
//    ii)  compute dot-product A(row, :) x B(:, col)
//    iii) use warp on each nonzero at float time
//    iv)  tally up accumulated sum using warp reduction
//    v)   write to global memory C_csrVal

typedef bool TYPE;

__global__ void spgemmMaskedKernel( TYPE*           C_csrVal,
                                    int64_t* mask_csrRowPtr,
                                    int64_t* mask_csrColInd,
                                    int32_t*           mask_csrVal,
                                    int64_t* A_csrRowPtr,
                                    int64_t* A_csrColInd,
                                    TYPE*     A_csrVal,
                                    int64_t* B_cscColPtr,
                                    int64_t* B_cscRowInd,
                                    TYPE*     B_cscVal,
                                    int        A_nrows,
                                    int        B_ncols,
                                    int if_mask,
                                    int mask_csc ) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id;
  // printf("thread_id = %d\n",thread_id); 
  
  
  if (mask_csc == 1) {
    if(warp_id < A_nrows) {
      // printf("thread_id = %d\n",thread_id); 
      int col_start = mask_csrRowPtr[0];
      int col_end = mask_csrRowPtr[1];
      
      // for (int edge = col_start; edge < col_end; ++edge) {
        int target_col = 0;
        int target_row = mask_csrColInd[warp_id];
        
        if (!mask_csrVal[warp_id]) {
          int a_col_start = A_csrRowPtr[target_row];
          int a_col_end = A_csrRowPtr[target_row + 1];
          // printf("a_col_start: %d, target row: %d\n", a_col_start, target_row);

          int b_row_start = B_cscColPtr[target_col];
          int b_row_end = B_cscColPtr[target_col + 1];
          bool accumulator = 0;
          int next_start = b_row_start;
          for(int iter = a_col_start; iter < a_col_end; iter++) {
            int col = A_csrColInd[iter];
            int B_ind = binarySearch(B_cscRowInd, col, next_start, b_row_end);
            next_start = B_ind;
            
            if (B_ind != -1) {
              accumulator += A_csrVal[iter] * B_cscVal[B_ind];
            }
          }
          C_csrVal[warp_id] = accumulator % 2;
        } else {
          C_csrVal[warp_id] = 0;
        }
      // }
    }

  } else {
      if(warp_id < A_nrows) {
        int row_start = mask_csrRowPtr[warp_id];
        int row_end = mask_csrRowPtr[warp_id + 1];

        for (int edge = row_start; edge < row_end; ++edge) {
          int target_row = warp_id;
          int target_col = mask_csrColInd[edge];

          if(!mask_csrVal[edge]) {
            int a_col_start = A_csrRowPtr[target_row];
            int a_col_end = A_csrRowPtr[target_row + 1];

            int b_row_start = B_cscColPtr[target_col];
            int b_row_end = B_cscColPtr[target_col + 1];
            bool accumulator = 0;
            for(int iter = a_col_start; iter < a_col_end; iter++) {
              int col = A_csrColInd[iter];
              int B_ind = binarySearch(B_cscRowInd, col, b_row_start, b_row_end);
              
              if (B_ind != -1) {
                accumulator += A_csrVal[iter] * B_cscVal[B_ind];
              }
            }

            C_csrVal[edge] = accumulator % 2;

          } else {
            C_csrVal[edge] = 0;
          }
        }
      }
  }

}


