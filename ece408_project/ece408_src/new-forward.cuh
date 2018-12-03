
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 32
#define NUM_THREADS 1024
#define NUM_IMAGES 1000

// Optimization 1: Unrolling and simple matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  float value = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < numARows && column < numBColumns) {
    for (int i = 0; i < numAColumns; i++) {
      value += A[row * numAColumns + i] * B[(numBRows * numBColumns) * blockIdx.z + i * numBColumns + column];
    }
    C[(numCRows * numCColumns) * blockIdx.z + row * numCColumns + column] = value;
  }
}

// Optimization 2: Unrolling and tiled matrix multiplication
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, 
                                     int numImages, int num) {
    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    for (int i = 0; i < (TILE_WIDTH+numAColumns-1)/TILE_WIDTH; i++) {
        if (i*TILE_WIDTH+threadIdx.x<numAColumns && row<numARows)
            subTileM[threadIdx.y][threadIdx.x] = A[row*numAColumns + i*TILE_WIDTH +threadIdx.x];
        else
            subTileM[threadIdx.y][threadIdx.x] = 0;

        if (i*TILE_WIDTH+threadIdx.y<numBRows && column<numBColumns)
            subTileN[threadIdx.y][threadIdx.x] = B[(numBRows * numBColumns) * blockIdx.z + numBColumns * (i*TILE_WIDTH+threadIdx.y) + column];
        else
            subTileN[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < numCRows && column < numCColumns) {
            for (int j = 0; j < TILE_WIDTH; j++)
                value += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (num * NUM_IMAGES + blockIdx.z < numImages && row < numCRows && column < numCColumns)
        C[(numCRows * numCColumns) * (num * NUM_IMAGES + blockIdx.z) + numCColumns * row + column] = value;
}

__global__ void forward_kernel_unroll(const float* x, float* unroll_x,
    const int H, const int W, const int B, const int C, const int K,
    const int W_out, const int matrixHeight, const int matrixWidth,
    const int numImage) {

    #define x4d(b,m,h,w) x[(b) * (C * H * W) + (m) * (H * W) + (h) * (W) + w]
    #define y4d(m,h,w) unroll_x[(m) * (matrixHeight * matrixWidth) + (h) * (matrixWidth) + w]

    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (numImage * NUM_IMAGES + blockIdx.y < B && threadIndex < C * matrixWidth) {
        const int row = (threadIndex % matrixWidth) / W_out;
        const int column = (threadIndex % matrixWidth) % W_out;

        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                y4d(blockIdx.y, (threadIndex / matrixWidth * K * K) + (i * K) + j, row * W_out + column) = x4d(numImage * NUM_IMAGES + blockIdx.y, threadIndex / matrixWidth, row + i, column + j);
            }
        }
    }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = k.shape_[0];
    const int C = k.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];

    const int W_out = W - K + 1;
    const int H_out = H - K + 1;
    const int matrixWidth = H_out * W_out;
    const int matrixHeight = C * K * K;

    mshadow::Tensor<gpu, 3, float> unroll_x;
    unroll_x.shape_ = mshadow::Shape3(matrixWidth, matrixHeight, NUM_IMAGES);
    mshadow::AllocSpace(&unroll_x);

    dim3 gridDim((NUM_THREADS+C*matrixWidth-1)/NUM_THREADS, NUM_IMAGES, 1);
    dim3 blockDim(NUM_THREADS, 1, 1);

    // Using simple matrix multiplication
    //dim3 dimBlock(16, 16, 1);
    //dim3 dimGrid((matrixWidth + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y, NUM_IMAGES);

    // Using tiled matrix multiplication
    dim3 gridMatrix((TILE_WIDTH+matrixWidth-1)/TILE_WIDTH, (TILE_WIDTH+M-1)/TILE_WIDTH, NUM_IMAGES);
    dim3 blockMatrix(TILE_WIDTH, TILE_WIDTH, 1);

    for (int i = 0; i < (B + NUM_IMAGES - 1)/NUM_IMAGES; i++) {
        forward_kernel_unroll<<<gridDim, blockDim>>>(x.dptr_, unroll_x.dptr_, H, W, B, C, K, W_out, matrixHeight, matrixWidth, i);
        matrixMultiplyShared<<<gridMatrix, blockMatrix>>>(k.dptr_, unroll_x.dptr_, y.dptr_, M, matrixHeight, matrixHeight, matrixWidth, M, matrixWidth, B, i);
        //matrixMultiply<<<dimGrid, dimBlock>>>(k.dptr_, unroll_x.dptr_, y.dptr_, M, matrixHeight, matrixHeight, matrixWidth, M, matrixWidth);
    }
    
    mshadow::FreeSpace(&unroll_x);

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
