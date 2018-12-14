#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet {
namespace op {

#define ceil(num,denom) ((denom + num - 1) / denom)

// logical image unrolling for layer 2
__global__ void forward_kernel_logical_l2(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y) {

    float value_N = 0;
    float value_O = 0;
    float value_P = 0;
    float value_Q = 0;
    float load_val_N = 0;
    float load_val_O = 0;
    float load_val_P = 0;
    float load_val_Q = 0;
    const unsigned int column = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float subTileM[24][49];
    __shared__ float subTileN[49][27];
    __shared__ float subTileO[49][27];
    __shared__ float subTileP[49][27];
    __shared__ float subTileQ[49][27];

    // Loads data from input image
    const unsigned int threadIndex = (threadIdx.y * blockDim.x) + threadIdx.x;

    const unsigned int inputImageRow = threadIndex / 33;
    const unsigned int inputImageCol = threadIndex % 33;

    #pragma unroll
    for (unsigned int channelNum = 0; channelNum < 12; channelNum++) {
        if (threadIndex < 231) {
            load_val_N = x[((4 * blockIdx.z) * 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            load_val_O = x[((4 * blockIdx.z + 1 ) * 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            load_val_P = x[((4 * blockIdx.z + 2 ) * 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            load_val_Q = x[((4 * blockIdx.z + 3 ) * 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];

            int outputRow = inputImageRow * 7;
            int outputCol = inputImageCol;
            for (unsigned int i = 0; i < 7; i++) {
                if (outputCol >= 0 && outputCol < 27) {
                    subTileN[outputRow][outputCol] = load_val_N;
                    subTileO[outputRow][outputCol] = load_val_O;
                    subTileP[outputRow][outputCol] = load_val_P;
                    subTileQ[outputRow][outputCol] = load_val_Q;
                }
                outputCol -= 1;
                outputRow += 1;
            }
        }
        __syncthreads();

        // Loads data from weight matrix
        subTileM[threadIdx.y][threadIdx.x] = w[threadIdx.y * 588 + channelNum * 49 + threadIdx.x];
        if (threadIdx.x + blockDim.x < 49) {
            subTileM[threadIdx.y][threadIdx.x + blockDim.x] = w[threadIdx.y * 588 + channelNum * 49 + threadIdx.x + blockDim.x];
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < 49; i++) {
            value_N += subTileM[threadIdx.y][i] * subTileN[i][threadIdx.x];
            value_O += subTileM[threadIdx.y][i] * subTileO[i][threadIdx.x];
            value_P += subTileM[threadIdx.y][i] * subTileP[i][threadIdx.x];
            value_Q += subTileM[threadIdx.y][i] * subTileQ[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.y < 24 && column < 729) {
        y[(17496 * (4 * blockIdx.z)) + (729 * threadIdx.y) + column] = value_N;
        y[(17496 * (4 * blockIdx.z + 1)) + (729 * threadIdx.y) + column] = value_O;
        y[(17496 * (4 * blockIdx.z + 2)) + (729 * threadIdx.y) + column] = value_P;
        y[(17496 * (4 * blockIdx.z + 3)) + (729 * threadIdx.y) + column] = value_Q;
    }
}

__constant__ float weightsL1[12][7][7];

__global__ void forward_kernel(const float* __restrict__ x, float* __restrict__ y) {
    #define input_image(i3,i1,i0) x[(i3)*(5184) + (i1)*(72) + i0]
    #define output_image(i3,i2,i1,i0) y[(i3)*(52272) + (i2)*(4356) + (i1)*(66) + i0]
    #define shared_image(i1,i0) x_shared[(i1)*(32) + i0]

    __shared__ float x_shared[1024];

    const unsigned int row = 26 * blockIdx.y + threadIdx.y;
    const unsigned int column = 26 * blockIdx.x + threadIdx.x;

    if (row < 72 && column < 72)
        shared_image(threadIdx.y, threadIdx.x) = input_image(blockIdx.z, row, column);
    __syncthreads();

    unsigned int outputChan, k_y, k_x;
    if (threadIdx.x < 26 && threadIdx.y < 26 && row < 66 && column < 66) {
        for (outputChan = 0; outputChan < 12; outputChan++) {
            float val = 0.0;
            for (k_y = 0; k_y < 7; k_y++) {
                for (k_x = 0; k_x < 7; k_x++) {
                    val += weightsL1[outputChan][k_y][k_x] * shared_image(threadIdx.y + k_y, threadIdx.x + k_x);
                }
            }
            output_image(blockIdx.z, outputChan, row, column) = val;
        }
    }

    #undef shared_image
    #undef output_image
    #undef input_image
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k) {
    const unsigned int numImages = x.shape_[0];

    if (k.shape_[1] == 1) {
        cudaMemcpyToSymbol(weightsL1, k.dptr_, 588 * sizeof(float));
        dim3 gridDim(ceil(66, 26), ceil(66, 26), numImages);
        dim3 blockDim(32, 32, 1);

        forward_kernel<<<gridDim, blockDim>>>(x.dptr_, y.dptr_);
    } else {
        dim3 logicalUnrollGridDim(ceil(729, 27), 1, numImages / 4);
        dim3 logicalUnrollBlockDim(27, 24, 1);
        forward_kernel_logical_l2<<<logicalUnrollGridDim, logicalUnrollBlockDim>>>(x.dptr_, k.dptr_, y.dptr_);
    }

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif
