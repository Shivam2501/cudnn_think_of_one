
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 32
#define NUM_THREADS 1024
#define UNROLL_BATCH_SIZE 1000

#define ceil(num,denom) ((denom + num - 1) / denom)

#define CONST_NUM_OUTPUT_CHANNELS 24
#define CONST_NUM_INPUT_CHANNELS 12
#define CONST_WEIGHT_DIM 7
__constant__ float weights[CONST_NUM_OUTPUT_CHANNELS * CONST_NUM_INPUT_CHANNELS * CONST_WEIGHT_DIM * CONST_WEIGHT_DIM];

// Optimization 1: Unrolling and simple matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
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
                                     int numImages, int batchSize) {
    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    for (int i = 0; i < ceil(numAColumns, TILE_WIDTH); i++) {
        if (i * TILE_WIDTH + threadIdx.x < numAColumns && row < numARows)
            subTileM[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
        else
            subTileM[threadIdx.y][threadIdx.x] = 0;

        if (i * TILE_WIDTH + threadIdx.y < numBRows && column < numBColumns)
            subTileN[threadIdx.y][threadIdx.x] = B[(numBRows * numBColumns) * blockIdx.z + numBColumns * (i * TILE_WIDTH + threadIdx.y) + column];
        else
            subTileN[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < numCRows && column < numCColumns) {
            for (int j = 0; j < TILE_WIDTH; j++)
                value += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (batchSize * UNROLL_BATCH_SIZE + blockIdx.z < numImages && row < numCRows && column < numCColumns)
        C[(numCRows * numCColumns) * (batchSize * UNROLL_BATCH_SIZE + blockIdx.z) + numCColumns * row + column] = value;
}

__global__ void matrixMultiplySharedUsingConstantMem(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, 
                                     int numImages, int batchSize, int numInputChannels) {

    //#define wd(m, c, h,w) weights[(m) * (numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (c) * (CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (h) * (CONST_WEIGHT_DIM) + w]

    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
    
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     for (int i = 0; i < 50; i++) {
    //         int index = i / (CONST_WEIGHT_DIM * CONST_WEIGHT_DIM);
    //         int index2 = i % (CONST_WEIGHT_DIM * CONST_WEIGHT_DIM);
    //         printf("%lf", weights[row][index][index2 / CONST_WEIGHT_DIM][index2 % CONST_WEIGHT_DIM]);
    //     }
    // }

    for (int i = 0; i < ceil(numAColumns, TILE_WIDTH); i++) {
        if (i * TILE_WIDTH + threadIdx.x < numAColumns && row < numARows)
            subTileM[threadIdx.y][threadIdx.x] = weights[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
        else
            subTileM[threadIdx.y][threadIdx.x] = 0;

        if (i * TILE_WIDTH + threadIdx.y < numBRows && column < numBColumns)
            subTileN[threadIdx.y][threadIdx.x] = B[(numBRows * numBColumns) * blockIdx.z + numBColumns * (i * TILE_WIDTH + threadIdx.y) + column];
        else
            subTileN[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < numCRows && column < numCColumns) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                //value += weights[(row) * (numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (i * TILE_WIDTH + j)] * subTileN[j][threadIdx.x];
                value += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (batchSize * UNROLL_BATCH_SIZE + blockIdx.z < numImages && row < numCRows && column < numCColumns)
        C[(numCRows * numCColumns) * (batchSize * UNROLL_BATCH_SIZE + blockIdx.z) + numCColumns * row + column] = value;
}

__global__ void forward_kernel_unroll(const float* x, float* unroll_x,
    const int inputImageHeight, const int inputImageWidth, const int numImages, const int numInputChannels, const int weightDim,
    const int outputImageWidth, const int matrixHeight, const int matrixWidth,
    const int batchSize) {

    #define x4d(b,m,h,w) x[(b) * (numInputChannels * inputImageHeight * inputImageWidth) + (m) * (inputImageHeight * inputImageWidth) + (h) * (inputImageWidth) + w]
    #define y4d(m,h,w) unroll_x[(m) * (matrixHeight * matrixWidth) + (h) * (matrixWidth) + w]

    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchSize * UNROLL_BATCH_SIZE + blockIdx.y < numImages && threadIndex < numInputChannels * matrixWidth) {
        const int row = (threadIndex % matrixWidth) / outputImageWidth;
        const int column = (threadIndex % matrixWidth) % outputImageWidth;

        for (int i = 0; i < weightDim; ++i) {
            for (int j = 0; j < weightDim; ++j) {
                int h_unroll = (threadIndex / matrixWidth * weightDim * weightDim) + (i * weightDim) + j;
                int w_unroll = row * outputImageWidth + column;
                y4d(blockIdx.y, h_unroll, w_unroll) = x4d(batchSize * UNROLL_BATCH_SIZE + blockIdx.y, threadIndex / matrixWidth, 
                                                          row + i, column + j);
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
    const int numImages = x.shape_[0];
    const int numOutputChannels = k.shape_[0];
    const int numInputChannels = k.shape_[1];
    const int inputImageHeight = x.shape_[2];
    const int inputImageWidth = x.shape_[3];
    const int weightDim = k.shape_[3];

    const int outputImageWidth = inputImageWidth - weightDim + 1;
    const int outputImageHeight = inputImageHeight - weightDim + 1;
    const int unrolledMatrixWidth = outputImageHeight * outputImageWidth;
    const int unrollMatrixHeight = numInputChannels * weightDim * weightDim;

    fprintf(stdout, "Weight Matrix: %d, %d, %d\n", numOutputChannels, numInputChannels, weightDim);
    cudaMemcpyToSymbol(weights, k.dptr_, numOutputChannels * numInputChannels * weightDim * weightDim * sizeof(float));

    mshadow::Tensor<gpu, 3, float> unroll_x;
    unroll_x.shape_ = mshadow::Shape3(unrolledMatrixWidth, unrollMatrixHeight, UNROLL_BATCH_SIZE);
    mshadow::AllocSpace(&unroll_x);

    dim3 unrollGridDim(ceil(numInputChannels*unrolledMatrixWidth, NUM_THREADS), UNROLL_BATCH_SIZE, 1);
    dim3 unrollBlockDim(NUM_THREADS, 1, 1);

    // Using simple matrix multiplication
    //dim3 dimBlock(16, 16, 1);
    //dim3 dimGrid(ceil(unrolledMatrixWidth, dimBlock.x), ceil(M, dimBlock.y), UNROLL_BATCH_SIZE);

    // Using tiled matrix multiplication
    dim3 matrixGridDim(ceil(unrolledMatrixWidth, TILE_WIDTH), ceil(numOutputChannels, TILE_WIDTH), UNROLL_BATCH_SIZE);
    dim3 matrixBlockDim(TILE_WIDTH, TILE_WIDTH, 1);

    for (int batch = 0; batch < ceil(numImages, UNROLL_BATCH_SIZE); batch++) {
        forward_kernel_unroll<<<unrollGridDim, unrollBlockDim>>>(x.dptr_, unroll_x.dptr_, inputImageHeight, inputImageWidth, 
                                                                 numImages, numInputChannels, weightDim, outputImageWidth, 
                                                                 unrollMatrixHeight, unrolledMatrixWidth, batch);
        matrixMultiplySharedUsingConstantMem<<<matrixGridDim, matrixBlockDim>>>(k.dptr_, unroll_x.dptr_, y.dptr_, numOutputChannels, 
                                                                                unrollMatrixHeight, unrollMatrixHeight, unrolledMatrixWidth, 
                                                                                numOutputChannels, unrolledMatrixWidth, numImages, batch, numInputChannels);
        //matrixMultiply<<<dimGrid, dimBlock>>>(k.dptr_, unroll_x.dptr_, y.dptr_, M, matrixHeight, matrixHeight, unrolledMatrixWidth, M, unrolledMatrixWidth);
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
