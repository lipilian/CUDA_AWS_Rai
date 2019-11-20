#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_WIDTH 16


#include <mxnet/base.h>

namespace mxnet
{
	namespace op
	{

		//__constant__ mask[6000] // 64KB shared memory can hold 16000 float number, I guess 6000 is enought for the kernel to run
		__global__ void unrollMatrix(float* y, const float* x, const int C, const int H, const int W, const int K) {
			int c = blockIdx.x;
			int blockNum = blockIdx.y;
			const int H_out = H - K + 1;
			const int W_out = W - K + 1;
			int W_grid = ceil(W_out / (float)BLOCK_WIDTH);
			int H_grid = ceil(H_out / (float)BLOCK_WIDTH);
			int RowBlockStart = blockNum / W_grid;
			int ColBlockStart = blockNum % W_grid;
			int ty = threadIdx.y; int tx = threadIdx.x;
			int RowStart = RowBlockStart * BLOCK_WIDTH;
			int ColStart = ColBlockStart * BLOCK_WIDTH;
			int Row = RowStart + ty;
			int Col = ColStart + tx;

			#define y2d(i1, i0) y[i1 * (H_out * W_out) + i0]
			#define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]

			int Row_unroll, Col_unroll = Row * W_out + Col;
			if(Row < H_out && Col < W_out){
				for(int p = 0; p < K; p++){
					for(int q = 0; q < K; q++){
							Row_unroll = c * K * K + p * K + q;
							y2d(Row_unroll, Col_unroll) = x3d(c, Row + p, Col + q);
					}
				}
			}
			__syncthreads();
			#undef y2d
			#undef x3d
		}

		__global__ void matrixMutiplyShared(const float *A, const float *B, float *C, int numAColumns, int numCRows, int numCColumns){
			__shared__ float subTileM[BLOCK_WIDTH][BLOCK_WIDTH];
  		__shared__ float subTileN[BLOCK_WIDTH][BLOCK_WIDTH];
			int bx = blockIdx.x; int by = blockIdx.y;
			int tx = threadIdx.x; int ty = threadIdx.y;
			int RowIdx = by * BLOCK_WIDTH + ty;
			int ColIdx = bx * BLOCK_WIDTH + tx;
			float Pvalue = 0.0;
			for (int m = 0; m < (numAColumns - 1)/BLOCK_WIDTH + 1; m++) {
				int col = m * BLOCK_WIDTH + tx;
				if(col < numAColumns && RowIdx < numCRows){
					subTileM[ty][tx] = A[RowIdx * numAColumns + col];
				}else{
					subTileM[ty][tx] = 0.;
				}
				int row = m * BLOCK_WIDTH + ty;
				if(row < numAColumns && ColIdx < numCColumns){
					subTileN[ty][tx] = B[row * numCColumns + ColIdx];
				}else{
					subTileN[ty][tx] = 0.;
				}

				__syncthreads();

				for(int k = 0; k <BLOCK_WIDTH; k++){
					Pvalue += subTileM[ty][k] * subTileN[k][tx];
				}
				__syncthreads();
			}
			if(RowIdx < numCRows && ColIdx < numCColumns)
				C[RowIdx * numCColumns + ColIdx] = Pvalue;
		}
		// CPU function that call GPU kernels

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
			const int Z = H_grid * W_grid;
    	// Set the kernel dimensions
    	dim3 gridDim(C, Z);
    	dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
			float *X_unroll;
			int W_unroll = H_out * W_out;
			int H_unroll = C * K * K;
			int Size_unroll = W_unroll * H_unroll;
			cudaMalloc((void **)&X_unroll, sizeof(float) * Size_unroll);
			for(int b = B - 1; b >= 0; b--){
				gridDim = dim3(C,Z);
				unrollMatrix<<<gridDim, blockDim>>>(X_unroll, x.dptr_ + b * C * H * W, C, H, W, K);
				MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
				gridDim = dim3(ceil((float)W_unroll/BLOCK_WIDTH), ceil((float)M/BLOCK_WIDTH));
				matrixMutiplyShared<<<gridDim, blockDim>>>(w.dptr_, X_unroll, y.dptr_ + b * M * W_unroll, H_unroll, M, W_unroll);
				MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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
