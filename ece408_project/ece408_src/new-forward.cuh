
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define ceil(num,denom) ((denom + num - 1) / denom)

#define CONST_NUM_OUTPUT_CHANNELS 24
#define CONST_NUM_INPUT_CHANNELS 12
#define CONST_WEIGHT_DIM 7

__constant__ float weights[CONST_NUM_OUTPUT_CHANNELS * CONST_NUM_INPUT_CHANNELS * CONST_WEIGHT_DIM * CONST_WEIGHT_DIM];
//texture<float, 2, cudaReadModeElementType> texDesc;

// logical image unrolling for layer 1
__global__ void forward_kernel_logical_l1(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, const int numImages, 
                                          const int numInputChannels, const int inputImageHeight, const int inputImageWidth, const int weightDim,
                                          const int numOutputChannels, const int outputMatrixWidth, const int outputImageWidth) {

    float value_N = 0;
    float value_O = 0;
    float value_P = 0;
    float load_val_N = 0;
    float load_val_O = 0;
    float load_val_P = 0;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int column = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float subTileM[12 * 49];
    __shared__ float subTileN[49][66];
    __shared__ float subTileO[49][66];
    __shared__ float subTileP[49][66];

    // Loads data from input image
    const unsigned int threadIndex = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadIndex < weightDim*inputImageWidth) {
        const unsigned int inputImageRow = threadIndex / inputImageWidth;
        const unsigned int inputImageCol = threadIndex % inputImageWidth;
        if (3 * blockIdx.z < numImages)
            load_val_N = x[((3 * blockIdx.z) * inputImageHeight * inputImageWidth) + ( (inputImageRow + blockIdx.x) * inputImageWidth) + inputImageCol];
        if (3 * blockIdx.z + 1 < numImages)
            load_val_O = x[((3 * blockIdx.z + 1) * inputImageHeight * inputImageWidth) + ( (inputImageRow + blockIdx.x) * inputImageWidth) + inputImageCol];
        if (3 * blockIdx.z + 2 < numImages)
            load_val_P = x[((3 * blockIdx.z + 2) * inputImageHeight * inputImageWidth) + ( (inputImageRow + blockIdx.x) * inputImageWidth) + inputImageCol];

        int outputRow = inputImageRow * weightDim;
        int outputCol = inputImageCol;
        for (unsigned int i = 0; i < weightDim; i++) {
            if (outputCol >= 0 && outputCol < outputImageWidth){
                if (3 * blockIdx.z < numImages)
                    subTileN[outputRow][outputCol] = load_val_N;
                if (3 * blockIdx.z + 1 < numImages)
                    subTileO[outputRow][outputCol] = load_val_O;
                if (3 * blockIdx.z + 2 < numImages)
                    subTileP[outputRow][outputCol] = load_val_P;
            }
            outputCol -= 1;
            outputRow += 1;
        }
    }

    // Loads data from weight matrix
    int weightIndexLimit = numOutputChannels*49;
    if (threadIndex < weightIndexLimit) {
        subTileM[threadIndex] = w[threadIndex];
    }
    __syncthreads();

    if (row < numOutputChannels && column < outputMatrixWidth) {
        #pragma unroll
        for (unsigned int i = 0; i < 49; i++) {
            //value += weights[(row*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileN[i][threadIdx.x];
            if (3 * blockIdx.z < numImages)
                value_N += subTileM[(row*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileN[i][threadIdx.x];
            if (3 * blockIdx.z + 1 < numImages)
                value_O += subTileM[(row*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileO[i][threadIdx.x];
            if (3 * blockIdx.z + 2 < numImages)
                value_P += subTileM[(row*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileP[i][threadIdx.x];
            //value += tex2D(texDesc, row, i) * subTileN[i][threadIdx.x];
        }
        if (3 * blockIdx.z < numImages)
            y[(numOutputChannels * outputMatrixWidth * (3 * blockIdx.z)) + (outputMatrixWidth * row) + column] = value_N;
        if (3 * blockIdx.z + 1 < numImages)
            y[(numOutputChannels * outputMatrixWidth * (3 * blockIdx.z + 1)) + (outputMatrixWidth * row) + column] = value_O;
        if (3 * blockIdx.z + 2 < numImages)
            y[(numOutputChannels * outputMatrixWidth * (3 * blockIdx.z + 2)) + (outputMatrixWidth * row) + column] = value_P;
    }
}

// logical image unrolling for layer 2
__global__ void forward_kernel_logical_l2(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, const int numImages, const int numInputChannels,
                                              const int inputImageHeight, const int inputImageWidth, const int weightDim,
                                              const int numOutputChannels, const int outputMatrixWidth, const int outputImageWidth) {

    float value_N = 0;
    float value_O = 0;
    float value_P = 0;
    float value_Q = 0;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
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
            float load_val_N = x[((4 * blockIdx.z) * 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            float load_val_O = x[((4 * blockIdx.z + 1 )* 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            float load_val_P = x[((4 * blockIdx.z + 2 )* 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];
            float load_val_Q = x[((4 * blockIdx.z + 3 )* 13068) + (channelNum * 1089) + ( (inputImageRow + blockIdx.x) * 33) + inputImageCol];

            int outputRow = inputImageRow * 7;
            int outputCol = inputImageCol;
            for (unsigned int i = 0; i < 7; i++) {
                if (outputCol >= 0 && outputCol < 27){
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
            //value += weights[(row*numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (channelNum*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileN[i][threadIdx.x];
            //value_O += weights[(row*numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (channelNum*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileO[i][threadIdx.x];
            //value_P += weights[(row*numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (channelNum*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileP[i][threadIdx.x];
            //value_Q += weights[(row*numInputChannels*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + (channelNum*CONST_WEIGHT_DIM*CONST_WEIGHT_DIM) + i] * subTileQ[i][threadIdx.x];
            value_N += subTileM[row][i] * subTileN[i][threadIdx.x];
            value_O += subTileM[row][i] * subTileO[i][threadIdx.x];
            value_P += subTileM[row][i] * subTileP[i][threadIdx.x];
            value_Q += subTileM[row][i] * subTileQ[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < 24 && column < 729) {
        y[(17496 * (4 * blockIdx.z)) + (729 * row) + column] = value_N;
        y[(17496 * (4 * blockIdx.z + 1)) + (729 * row) + column] = value_O;
        y[(17496 * (4 * blockIdx.z + 2)) + (729 * row) + column] = value_P;
        y[(17496 * (4 * blockIdx.z + 3)) + (729 * row) + column] = value_Q;
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
    const unsigned int numImages = x.shape_[0];
    const unsigned int numOutputChannels = k.shape_[0];
    const unsigned int numInputChannels = k.shape_[1];
    const unsigned int inputImageHeight = x.shape_[2];
    const unsigned int inputImageWidth = x.shape_[3];
    const unsigned int weightDim = k.shape_[3];

    const unsigned int outputImageWidth = inputImageWidth - weightDim + 1;
    const unsigned int outputImageHeight = inputImageHeight - weightDim + 1;
    const unsigned int unrolledMatrixWidth = outputImageHeight * outputImageWidth;

    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // cudaArray* cuArray;
    // cudaMallocArray(&cuArray, &channelDesc, numInputChannels * weightDim * weightDim, numOutputChannels, cudaArraySurfaceLoadStore);
    // cudaMemcpyToArray(cuArray, 0, 0, k.dptr_, numOutputChannels * numInputChannels * weightDim * weightDim * sizeof(float), cudaMemcpyHostToDevice);

    // texDesc.addressMode[0]   = cudaAddressModeWrap;
    // texDesc.addressMode[1]   = cudaAddressModeWrap;
    // texDesc.filterMode       = cudaFilterModeLinear;
    // texDesc.normalized       = true;
    // cudaBindTextureToArray(texDesc, cuArray, channelDesc);

    //cudaMemcpyToSymbol(weights, k.dptr_, numOutputChannels * numInputChannels * weightDim * weightDim * sizeof(float));

    if (numInputChannels == 1) {
        //fprintf(stdout, "Grid Dim Layer 1: %d, %d, %d, %d\n", unrolledMatrixWidth, numOutputChannels, ceil(unrolledMatrixWidth, 66), ceil(numOutputChannels, 12));
        dim3 logicalUnrollGridDim(ceil(unrolledMatrixWidth, 66), ceil(numOutputChannels, 12), ceil(numImages, 3));
        dim3 logicalUnrollBlockDim(66, 12, 1);
        forward_kernel_logical_l1<<<logicalUnrollGridDim, logicalUnrollBlockDim>>>(x.dptr_, k.dptr_, y.dptr_, numImages, numInputChannels,
                                                                                   inputImageHeight, inputImageWidth, weightDim, numOutputChannels,
                                                                                   unrolledMatrixWidth, outputImageWidth);
    } else {
        //fprintf(stdout, "Grid Dim Layer 2: %d, %d, %d, %d\n", unrolledMatrixWidth, numOutputChannels, ceil(unrolledMatrixWidth, 27), ceil(numOutputChannels, 24));
        dim3 logicalUnrollGridDim(ceil(unrolledMatrixWidth, 27), ceil(numOutputChannels, 24), numImages / 4);
        dim3 logicalUnrollBlockDim(27, 24, 1);
        forward_kernel_logical_l2<<<logicalUnrollGridDim, logicalUnrollBlockDim>>>(x.dptr_, k.dptr_, y.dptr_, numImages, numInputChannels,
                                                                                   inputImageHeight, inputImageWidth, weightDim, numOutputChannels,
                                                                                   unrolledMatrixWidth, outputImageWidth);
    }

    //cudaFreeArray(cuArray);
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
