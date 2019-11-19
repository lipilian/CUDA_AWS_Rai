#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_WIDTH 16
#define BLOCK_SIZE 256
#define TILE_WIDTH 32 //define tile size

#include <mxnet/base.h>

namespace mxnet
{
	namespace op
	{

		//__constant__ mask[6000] // 64KB shared memory can hold 16000 float number, I guess 6000 is enought for the kernel to run
		__global__ void unrollMatrix(float* y, const float* x, const int C, const int H, const int W, const int K) {
			int c,s,h_out,w_out,w_unroll,h_unroll,w_base,p,q;
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			const int H_out = H - K + 1;
			const int W_out = W - K + 1;
			#define y2d(i1, i0) y[(i1) * (H_out * W_out) + i0]
			#define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]
			int W_unroll = H_out * W_out; // size of each Channel layer
			if (tid < C * W_unroll) {
				c = tid / W_unroll;
				s = tid % W_unroll;
				h_out = s / W_out;
				w_out = s % W_out;
				w_unroll = s;
				w_base = c * K * K;
				for(p = 0; p < K; p++){
					for(q = 0; q < K; q++){
						h_unroll = w_base + p * K + q;
						y2d(h_unroll, w_unroll) = x3d(c, h_out + p, w_out + q);
					}
				}
			}
			__syncthreads();
			#undef y3d
			#undef x4d
		}
		__global__ void matrixMutiplyShared(const float *A, const float *B, float *C, int numAColumns, int numCRows, int numCColumns){
			__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  		__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
			int bx = blockIdx.x; int by = blockIdx.y;
			int tx = threadIdx.x; int ty = threadIdx.y;
			int RowIdx = by * TILE_WIDTH + ty;
			int ColIdx = bx * TILE_WIDTH + tx;
			float Pvalue = 0.0;
			for (int m = 0; m < (numAColumns - 1)/TILE_WIDTH + 1; m++) {
				int col = m * TILE_WIDTH + tx;
				if(col < numAColumns && RowIdx < numCRows){
					subTileM[ty][tx] = A[RowIdx * numAColumns + col];
				}else{
					subTileM[ty][tx] = 0;
				}
				int row = m * TILE_WIDTH + ty;
				if(row < numAColumns && ColIdx < numCColumns){
					subTileN[ty][tx] = B[row * numCColumns + ColIdx];
				}else{
					subTileN[ty][tx] = 0;
				}
				__syncthreads();
				for(int k = 0; k <TILE_WIDTH; k++){
					Pvalue += subTileM[ty][k] * subTileN[k][tx];
				}
			}
			__syncthreads();
			C[RowIdx * numCColumns + ColIdx] = Pvalue;
		}
		// CPU function that call GPU kernels
		void unroll_gpu(float* y, float* x, int C, int H, int W, int K){
			const int H_out = H - K + 1;
			const int W_out = W - K + 1;
			int W_unroll = H_out * W_out;
			int gridDim = ceil(float(C * W_unroll)/BLOCK_SIZE);
			unrollMatrix<<<gridDim, BLOCK_SIZE>>>(y, x, C, H, W, K);
		}

		void MatrixMultiply(float *x, float *kernel, float* y, int W_unroll, int M, int H_unroll){
			dim3 gridDim (ceil(float(W_unroll)/TILE_WIDTH), ceil(float(M)/TILE_WIDTH), 1);
			dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
			matrixMutiplyShared<<<gridDim, blockDim>>>(kernel, x, y, H_unroll, M, W_unroll);
		}

		template <>
		void forward<gpu, float>(mshadow::Tensor<gpu, 4, float>& y, const mshadow::Tensor<gpu, 4, float>& x, const mshadow::Tensor<gpu, 4, float>& w)
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
			int W_grid = ceil(W_out / (1.0 * BLOCK_WIDTH));
			int H_grid = ceil(H_out / (1.0 * BLOCK_WIDTH));
			// Set the kernel dimensions
			float *X_unroll;
			int W_unroll = H_out * W_out;
			int H_unroll = C * K * K;
			int Size_unroll = W_unroll * H_unroll;
			cudaMalloc(&X_unroll, sizeof(float) * Size_unroll);
			//run unroll kernels
			for(int b = 0; b < B; b++){
				unroll_gpu(X_unroll, x.dptr_ + b * C * H * W, C, H, W, K);
				cudaDeviceSynchronize();
				MatrixMultiply(X_unroll, w.dptr_, y.dptr_ + b * M * W_unroll, W_unroll, M, H_unroll);
				cudaDeviceSynchronize();
				//+ b * M * W_unroll, W_unroll
			}
			cudaFree(X_unroll);

			// Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.

		}

		/*
			This tells mxnet how to do an op when it's not a float.
			This is not used in the ECE408 project
		*/
		template <typename gpu, typename DType>
		void forward(mshadow::Tensor<gpu, 4, DType>& y, const mshadow::Tensor<gpu, 4, DType>& x, const mshadow::Tensor<gpu, 4, DType>& w)
		{
			CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
		}
	}
}

#endif
