
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include <cusp/convert.h>
//#include <cusp/detail/matrix_base.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>

#define HOST_DEBUG

// setup - counting reorder
__global__ void counting_kernel(const float* a, const float* b, int* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c++;
}

void sk_counting(const float* a, const float* b, int* c)
{
    counting_kernel << <1, 100 >> > (a, b, c);
    cudaDeviceSynchronize();
}
// expansion
__global__ void expansion_kernel()
{
    ;
}

void sk_expansion()
{
    cudaDeviceSynchronize();
}
// sort
void sk_sort(thrust::device_vector<int> &I, thrust::device_vector<int> &J, thrust::device_vector<float> &V)
{
    thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
    thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));
}

// contraction
void sk_contraction(thrust::device_vector<int>& I, thrust::device_vector<int>& J, thrust::device_vector<float>& V)
{
    thrust::device_vector<int> II(4);
    thrust::device_vector<int> JJ(4);
    thrust::device_vector<int> VV(4);

    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(I.end(), J.end())),
        V.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(II.begin(), JJ.begin())),
        VV.begin(),
        thrust::equal_to< thrust::tuple<int, int> >(),
        thrust::plus<float>());

#ifdef HOST_DEBUG
    thrust::host_vector<int>   H_I = II;
    thrust::host_vector<int>   H_J = JJ;
    thrust::host_vector<float> H_V = VV;
    int s = H_I.size();
    s = s;
#endif // HOST_DEBUG
}


int main()
{
    thrust::device_vector<int> I_(4);
    thrust::device_vector<int> J_(4);
    thrust::device_vector<float> V_(4);

    I_[0] = 0; J_[0] = 0; V_[0] = 5;
    I_[1] = 0; J_[1] = 0; V_[1] = 6;
    I_[2] = 2; J_[2] = 3; V_[2] = 7;
    I_[3] = 2; J_[3] = 1; V_[3] = 8;

    cusp::array2d<float, cusp::host_memory> A(4, 4);
    A(0, 0) = 1;    A(0, 1) = 0;    A(0, 2) = 1;    A(0, 3) = 0;
    A(1, 0) = 0;    A(1, 1) = 1;    A(1, 2) = 0;    A(1, 3) = 1;
    A(2, 0) = 0;    A(2, 1) = 1;    A(2, 2) = 1;    A(2, 3) = 0;
    A(3, 0) = 1;    A(3, 1) = 0;    A(3, 2) = 0;    A(3, 3) = 1;

    cusp::array2d<float, cusp::host_memory> B(4, 4);
    B(0, 0) = 1;    B(0, 1) = 1;    B(0, 2) = 0;    B(0, 3) = 0;
    B(1, 0) = 1;    B(1, 1) = 1;    B(1, 2) = 1;    B(1, 3) = 0;
    B(2, 0) = 0;    B(2, 1) = 1;    B(2, 2) = 1;    B(2, 3) = 1;
    B(3, 0) = 0;    B(3, 1) = 0;    B(3, 2) = 1;    B(3, 3) = 1;

    cusp::coo_matrix<int, float, cusp::device_memory> A_;
    cusp::convert(A, A_);

    cusp::coo_matrix<int, float, cusp::device_memory> B_;
    cusp::convert(B, B_);

    cusp::coo_matrix<int, float, cusp::device_memory> C_;
    cusp::multiply(A_, B_, C_);

    float* H_A = thrust::raw_pointer_cast(&A.values[0]);
    float* H_B = thrust::raw_pointer_cast(&B.values[0]);
    int num_entries = 100;
    sk_counting(H_A, H_B, &num_entries);

    sk_expansion();

    sk_sort(I_, J_, V_);

    sk_contraction(I_, J_, V_);

#ifdef HOST_DEBUG
    thrust::host_vector<int> H_I = I_;
    thrust::host_vector<int> H_J = J_;
    thrust::host_vector<float> H_V = V_;
#endif // HOST_DEBUG

    return 0;
}
