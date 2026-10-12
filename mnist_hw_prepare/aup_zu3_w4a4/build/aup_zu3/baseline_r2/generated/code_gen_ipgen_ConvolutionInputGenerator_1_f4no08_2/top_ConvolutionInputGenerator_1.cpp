
#define AP_INT_MAX_W 64

#include "bnn-library.h"

// includes for network parameters
#include "slidingwindow.h"

// defines for network parameters
#define ConvKernelDim1 5
 #define IFMChannels1 16

                #define Input_precision1 4
 #define IFMDim1 12

                #define OFMDim1 8
 #define SIMD1 16

                #define Stride1 1
 #define numReps 1

void ConvolutionInputGenerator_1(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0_V,
                    hls::stream<ap_uint<SIMD1*Input_precision1>> &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
ConvolutionInputGenerator<ConvKernelDim1, IFMChannels1, Input_precision1, IFMDim1,
                    OFMDim1, SIMD1, Stride1> (in0_V, out_V, numReps, ap_resource_lutram());
}
