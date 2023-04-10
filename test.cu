#include "a1_matrix_format.c"
#include "a2_set_up.cu"
#include "a3_expansion_and_sort.cu"
#include "z_print.cu"

#include <iostream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <iomanip>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#define N_WARPS_PER_BLOCK (1 << 2)
#define WARP_SIZE (1 << 5)
#define N_THREADS_PER_BLOCK (1 << 7)

#include <iostream>


void print(const thrust::device_vector<int>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << v[i];
    std::cout << "\n";
}

void print(const thrust::device_vector<float>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << std::fixed << std::setprecision(1) << v[i];
    std::cout << "\n";
}

void print(thrust::device_vector<int>& v1, thrust::device_vector<int>& v2)
{
    for (size_t i = 0; i < v1.size(); i++)
        std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
    std::cout << "\n";
}

void printVector(int* counting, int nrow) {
    for (int i = 0; i < nrow; i++) {
        std::cout << counting[i] << " ";
    }
    std::cout << "\n";
}

void printVector(float* counting, int nrow) {
    for (int i = 0; i < nrow; i++) {
        std::cout << counting[i] << " ";
    }
    std::cout << "\n";
}

__device__
void printVectorDevice(int* vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

__device__
void printVectorDevice(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
}

// get a sparse matrix 
int generateASparseMatrixRandomly(int nrow, int ncol, float** result_matrix) {
    float* A = (float*)malloc(sizeof(float) * nrow * ncol);
    int nnz = 0;
    int r;
    float* cur;
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            r = rand();
            cur = A + (i * ncol + j);
            if (r % 4 == 0) { *cur = 100.0 * (r / (double)RAND_MAX);; }
            else { *cur = 0.0f; }
            if (*cur != 0.0f) { nnz++; }
        }
    }
    *result_matrix = A;
    return nnz;
}


// present matrix
void presentAMatrix(float* mat, int nrow, int ncol, int present_row, int present_col) {
    for (int i = 0; i < present_row; i++) {
        for (int j = 0; j < present_col; j++) {
            printf("%6.2f ", mat[i * ncol + j]);
        }
        printf("...\n");
    }
    printf("...\n");
}


// // convert the matrix in CSR format
void convertToCSRFormat(float* mat, int nrow, int ncol, int nnz, int** ptr, int** indices, float** data) {
    int* row_ptr = (int*)malloc(sizeof(int) * (nrow + 1));
    int* col_ind = (int*)malloc(sizeof(int) * nnz);
    float* nz_val = (float*)malloc(sizeof(float) * nnz);
    float* cur;
    int count = 0;
    for (int i = 0; i < nrow; i++) {
        row_ptr[i] = count;
        for (int j = 0; j < ncol; j++) {
            cur = mat + (i * ncol + j);
            if (*cur != 0.0f) {
                col_ind[count] = j;
                nz_val[count] = *cur;
                count++;
            }
        }
    }
    row_ptr[nrow] = count;

    *ptr = row_ptr;
    *indices = col_ind;
    *data = nz_val;
    return;
};



// // present CSR matrix
void presentCSR(int* ptr, int* indices, float* data, int nnz, int nrow) {
    printf("ptr -  ");
    for (int i = 0; i <= nrow; i++) {
        printf("%+7d", ptr[i]);
    }
    printf("\n");
    printf("ind -  ");
    for (int i = 0; i < nnz; i++) {
        printf("%+7d", indices[i]);
    }
    printf("\n");
    printf("data - ");
    for (int i = 0; i < nnz; i++) {
        printf("%+7.2f", data[i]);
    }
    printf("\n");
}


__global__
void countingKernel(int n, int* d_counting, int* A_ptr, int* A_ind, int* B_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    if (idx < n) {
        int start = A_ptr[idx];
        int end = A_ptr[idx + 1];
        for (int i = start; i < end; i++) {
            int j = A_ind[i];
            d_counting[idx] = d_counting[idx] + B_ptr[j + 1] - B_ptr[j];
        }
    }
    __syncthreads();
}

__device__
void bubbleSortSwap(int* col_ind, int i, int j) {
    int temp = col_ind[i];
    col_ind[i] = col_ind[j];
    col_ind[j] = temp;
}

__device__
void bubbleSortNetwork(int* col_ind, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (col_ind[i] > col_ind[j]) {
                bubbleSortSwap(col_ind, i, j);
            }
        }
    }
}

__global__
void expansioAndSortingKernel(int* order, int* counting, int* operations,
    int* A_ptr, int* A_ind, float* A_val, int A_row, int A_col, int A_nnz,
    int* B_ptr, int* B_ind, float* B_val, int B_row, int B_col, int B_nnz,
    int* resulting_row, int* resulting_col, float* resulting_val) {

    __shared__ int resulting_index;

    int bid = blockIdx.x;
    if (bid < A_row)
    {
        int a_r = order[bid];
        int a_count = A_ptr[a_r + 1] - A_ptr[a_r];

        int tid = threadIdx.x;
        if (tid < a_count)
        {
            int   a_offset = A_ptr[a_r];
            int   a_c = A_ind[a_offset + tid];
            float a_v = A_val[a_offset + tid];

            int b_r = a_c;
            int b_count = B_ptr[b_r + 1] - B_ptr[b_r];
            int b_offset = B_ptr[b_r];
            for (int i = 0; i < b_count; ++i)
            {
                int   b_c = B_ind[b_offset + i];
                float b_v = B_val[b_offset + i];
                //printf("a_r = %d, a_c = %d, b_r = %d, b_c = %d, a_v = %6.2f, b_v = %6.2f, a_v * b_v: %.2f \r\n", a_r, a_c, b_r, b_c, a_v, b_v, a_v * b_v);
                int iIndex = atomicAdd(&resulting_index, 1);
                resulting_row[operations[bid] + iIndex] = a_r;
                resulting_col[operations[bid] + iIndex] = b_c;
                resulting_val[operations[bid] + iIndex] = a_v * b_v;
                printf("a_r = %d, a_c = %d, b_r = %d, b_c = %d, a_v = %6.2f, b_v = %6.2f, a_v * b_v: %.2f \r\n", a_r, a_c, b_r, b_c, a_v, b_v, a_v * b_v);
            }
        }
    }


    /*
    __shared__ int cache[];

    // Block
    int bid = blockIdx.x;
    if (bid < A_nnz)
    {
        int a_c = A_ind[bid];
        float a_v = A_val[bid];
        int a_r = 0;
        for (int i = 0; i < A_nnz; i++)
        {
            int a_nnz_to_r = A_ptr[i + 1];
            if (bid < a_nnz_to_r)
            {
                a_r = i;
                break;
            }
        }

        // Thread
        int tid = threadIdx.x;
        if (tid < B_nnz)
        {
            int b_c = B_ind[tid];
            float b_v = B_val[tid];
            int b_r = 0;
            for (int i = 0; i < B_nnz; i++)
            {
                int b_nnz_to_r = B_ptr[i + 1];
                if (tid < b_nnz_to_r)
                {
                    b_r = i;
                    break;
                }
            }

            if (a_c == b_r)
            {
                printf("a_r = %d, a_c = %d, b_r = %d, b_c = %d, a_v = %f, b_v = %f, a_v * b_v: %f \r\n", a_r, a_c, b_r, b_c, a_v, b_v, a_v * b_v);
                cache[0] = a_r;
                cache[0] = b_c;
                cache[0] = a_v * b_v;
            }
        }
    }
    */

    /*
    extern __shared__ int shared_mem[];
    int* C_row_ind = shared_mem;
    int* C_row_col = shared_mem + N_THREADS_PER_BLOCK;
    float* C_row_val = (float*)(shared_mem + N_THREADS_PER_BLOCK * 2);


    int tid = threadIdx.x; // the idx of row
    int num_C_row_nnz = 0;

    if (tid < A_row) {
        int row = order[tid];
        int A_row_start = A_ptr[row];
        int A_row_end = A_ptr[row + 1];
        num_C_row_nnz = A_row_end - A_row_start;

        // initialize the shared memory
        int* warp_C_row_ind = C_row_ind + tid;
        int* warp_C_row_col = C_row_col + tid;
        float* warp_C_row_val = C_row_val + tid;

        for (int entryIdx = tid + A_row_start; entryIdx < A_row_end; entryIdx += 32) {
            int k = A_ind[entryIdx];
            int A_ik = A_val[entryIdx];
            int B_rowk_start = B_ptr[k];
            int B_rowk_end = B_ptr[k + 1];
            for (int i = B_rowk_start; i < B_rowk_end; i++) {
                int j = B_ind[i];
                int B_kj = B_val[i];
                warp_C_row_val[*warp_C_row_ind] = A_ik * B_kj;
                warp_C_row_col[*warp_C_row_ind] = j;
                printf("    warp %d, lane %d, the warp_C_row_ind is %d, the col is %d, the val is %f \n", tid, tid, *warp_C_row_ind, warp_C_row_col[*warp_C_row_ind], warp_C_row_val[*warp_C_row_ind]);
                *warp_C_row_ind++;
            }
        }
    }
    __syncthreads();

    if (tid < A_row) {

        bubbleSortNetwork(C_row_col, num_C_row_nnz);
        // copy the shared memory to the resulting array
        for (int i = tid; i < num_C_row_nnz; i += 32) {
            resulting_row[operations[tid] + i] = row;
            resulting_col[operations[tid] + i] = warp_C_row_col[i];
            resulting_val[operations[tid] + i] = warp_C_row_val[i];
            printf(" i  = %d \n", i);
            printf("the idx on the resulting array is %d, the row is %d, the col is %d, the val is %f \n", operations[tid] + i, resulting_row[operations[tid] + i], resulting_col[operations[tid] + i], resulting_val[operations[tid] + i]);
        }
    }
    __syncthreads();
    */
}

int main() {
    // got the matrix A and B
    int size = 9;
    int A_row = size;
    int A_col = size;
    int B_row = size;
    int B_col = size;
    int present_row = size;
    int present_col = size;
    float *A;
    float *B;
    int A_nnz = generateASparseMatrixRandomly(A_row, A_col, &A);
    int B_nnz = generateASparseMatrixRandomly(B_row, B_col, &B);
    presentAMatrix(A, A_row, A_col, present_row, present_col);
    presentAMatrix(B, B_row, B_col, present_row, present_col);
    int *A_ptr;
    int *A_ind;
    float *A_val;
    convertToCSRFormat(A, A_row, A_col, A_nnz, &A_ptr, &A_ind, &A_val);
    presentCSR(A_ptr, A_ind, A_val, A_nnz, A_row);
    int *B_ptr;
    int *B_ind;
    float *B_val;
    convertToCSRFormat(B, B_row, B_col, B_nnz, &B_ptr, &B_ind, &B_val);
    presentCSR(B_ptr, B_ind, B_val, B_nnz, B_row);

    // allocate memory for d_A_ and d_B_
    int *d_A_ptr, *d_A_ind, *d_B_ptr, *d_B_ind;
    float *d_A_val, *d_B_val;
    cudaMalloc(&d_A_ptr, (A_row+1) * sizeof(int));
    cudaMalloc(&d_A_ind, A_nnz * sizeof(int));
    cudaMalloc(&d_A_val, A_nnz * sizeof(float));
    cudaMalloc(&d_B_ptr, (B_row+1) * sizeof(int));
    cudaMalloc(&d_B_ind, B_nnz * sizeof(int));
    cudaMalloc(&d_B_val, B_nnz * sizeof(float));
    cudaMemcpy(d_A_ptr, A_ptr, (A_row+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_ind, A_ind, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A_val, A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptr, B_ptr, (B_row+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ind, B_ind, B_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_val, B_val, B_nnz * sizeof(float), cudaMemcpyHostToDevice);
    
    // setup phase
    int * counting = (int *)malloc(A_row * sizeof(int));
    int * d_counting;
    cudaMalloc(&d_counting, A_row * sizeof(int));
    cudaMemset(d_counting, 0, A_row * sizeof(int));
    int set_up_n_blocks = (A_row + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
    
    // counting 
    countingKernel<<<set_up_n_blocks, N_THREADS_PER_BLOCK>>>(A_row, d_counting, d_A_ptr, d_A_ind, d_B_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(counting, d_counting, A_row * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "counting: ";
    printVector(counting, A_row);

    thrust::device_ptr<int> d_counting_ptr(d_counting);
    thrust::device_vector<int> d_counting_vec(d_counting_ptr, d_counting_ptr + A_row);
    thrust::device_vector<int> d_order_vec(A_row);
    thrust::sequence(d_order_vec.begin(), d_order_vec.end());
    thrust::stable_sort_by_key(d_counting_vec.begin(), d_counting_vec.end(), d_order_vec.begin(), thrust::greater<int>());
    thrust::device_vector<int> d_operations_vec(A_row+1);
    thrust::exclusive_scan(d_counting_vec.begin(), d_counting_vec.end(), d_operations_vec.begin());
    int tot_operations = d_counting_vec.back()+d_operations_vec[A_row-1];
    d_operations_vec[A_row] = tot_operations;

    std::cout << "d_counting_vec: ";
    print(d_counting_vec);
    std::cout << "d_order_vec: ";
    print(d_order_vec);
    std::cout << "d_operations_vec: ";
    print(d_operations_vec);

    int * order = (int *)malloc(A_row * sizeof(int));
    int * operations = (int *)malloc((A_row+1) * sizeof(int));
    thrust::copy(d_order_vec.begin(), d_order_vec.end(), order);
    thrust::copy(d_operations_vec.begin(), d_operations_vec.end(), operations);
    thrust::copy(d_counting_vec.begin(), d_counting_vec.end(), counting);
    std::cout << "from thrust back to cuda device ptr and vec:" << "\n";
    std::cout << "order: ";
    printVector(order, A_row);
    std::cout << "counting: ";
    printVector(counting, A_row);
    std::cout << "operations: ";
    printVector(operations, A_row+1);
    int * d_order;
    int * d_operations;
    cudaMalloc(&d_order, A_row * sizeof(int));
    cudaMalloc(&d_operations, (A_row+1) * sizeof(int));
    cudaMemcpy(d_order, order, A_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_operations, operations, (A_row+1) * sizeof(int), cudaMemcpyHostToDevice);

    // expansion and sorting phase
    int * d_resulting_row, * d_resulting_col; 
    float * d_resulting_val;
    cudaMalloc(&d_resulting_row, tot_operations * sizeof(int));
    cudaMalloc(&d_resulting_col, tot_operations * sizeof(int));
    cudaMalloc(&d_resulting_val, tot_operations * sizeof(float));

    //int expansion_and_sorting_n_warps = A_row;
    //int enpansion_and_sorting_n_blocks = (expansion_and_sorting_n_warps + N_WARPS_PER_BLOCK - 1) / N_WARPS_PER_BLOCK;
    //int shared_memory_size_per_block = N_WARPS_PER_BLOCK * (sizeof(int) + 32 * sizeof(int) + 32 * sizeof(float));
    //expansioAndSortingKernel<<<enpansion_and_sorting_n_blocks, N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_order, d_counting, d_operations,
    //                                                                                                                d_A_ptr, d_A_ind, d_A_val, A_row, A_col, A_nnz,
    //                                                                                                                d_B_ptr, d_B_ind, d_B_val, B_row, B_col, B_nnz,
    //                                                                                                                d_resulting_row, d_resulting_col, d_resulting_val);

    int enpansion_and_sorting_n_blocks = (A_row + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
    expansioAndSortingKernel<<<A_row, N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_order, d_counting, d_operations,
                                                                                                                    d_A_ptr, d_A_ind, d_A_val, A_row, A_col, A_nnz,
                                                                                                                    d_B_ptr, d_B_ind, d_B_val, B_row, B_col, B_nnz,
                                                                                                                    d_resulting_row, d_resulting_col, d_resulting_val);

    int * resulting_row = (int *)malloc(tot_operations * sizeof(int));
    int * resulting_col = (int *)malloc(tot_operations * sizeof(int));
    float * resulting_val = (float *)malloc(tot_operations * sizeof(float));
    cudaMemcpy(resulting_row, d_resulting_row, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_col, d_resulting_col, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_val, d_resulting_val, tot_operations * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "resulting_row: ";
    printVector(resulting_row, tot_operations);
    std::cout << "resulting_col: ";
    printVector(resulting_col, tot_operations);
    std::cout << "resulting_val: ";
    printVector(resulting_val, tot_operations);
    cudaDeviceSynchronize();
    return 0;
}
