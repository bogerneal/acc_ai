
#define AP_INT_MAX_W 32

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "mvau.hpp"

// defines for network parameters
#define MW1 128
 #define MH1 10

            #define SIMD1 1
 #define PE1 1
 #define WMEM1 1280

            #define TMEM1 0
 #define numReps 1
#define WP1 4


void MVAU_3(
                    hls::stream<ap_uint<4>> &in0_V,
                    hls::stream<ap_uint<4>> &weights_V,
                    hls::stream<ap_uint<32>> &out_V
                    )
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=weights_V
Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, Slice<ap_uint<4>>, Slice<ap_int<32>>, Identity, ap_int<4> >
                (in0_V, out_V, weights_V, PassThroughActivation<ap_int<32>>(), numReps, ap_resource_dflt());
}
