
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16//define tile size

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//__constant__ mask[6000] // 64KB shared memory can hold 16000 float number, I guess 6000 is enought for the kernel to run

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  // first try load all valuable from global memory
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  int b = blockIdx.x; // Batch number of images
  int m = blockIdx.y; // feature map
  int h = blockIdx.z / W_out;
  int w = blockIdx.z % W_out;

  extern __shared__ float sharedMextern[];
  float *sharedM = &sharedMextern[0];

  int index = threadIdx.y * TILE_WIDTH + threadIdx.x;

  for(int c = 0; c < C; c++){
    if(index < K * K){
      int p = index / K;
      int q = index % K;
      sharedM[index] = x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
    } else {
      sharedM[index] = 0.0;
    }
    __syncthreads();

    for(unsigned int stride = blockDim.x * blockDim.y / 2; stride >=1; stride >>= 1){
      if(index < stride){
        sharedM[index] += sharedM[index + stride];
      }
      __syncthreads();
    }
    if(index == 0){
      y4d(b, m, h, w) = y4d(b,m,h,w) + sharedM[0];
    }
  }




  #undef y4d
  #undef x4d
  #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = H - K + 1;
    //int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    //int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
    const int Z = H_out * W_out;
    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // Call the kernel
    size_t sharedExternSize = sizeof(float) * (256);
    forward_kernel<<<gridDim, blockDim, sharedExternSize>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
