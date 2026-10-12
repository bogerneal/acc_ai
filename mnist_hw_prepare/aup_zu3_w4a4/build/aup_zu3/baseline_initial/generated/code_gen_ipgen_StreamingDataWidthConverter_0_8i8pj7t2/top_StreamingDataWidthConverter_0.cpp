
#define AP_INT_MAX_W 64

#include "bnn-library.h"

// includes for network parameters
#include "streamtools.h"

// defines for network parameters
#define InWidth 4 
#define OutWidth 64 
#define NumInWords 9216 
#define numReps 1

void StreamingDataWidthConverter_0(hls::stream<ap_uint<4> > &in0_V, hls::stream<ap_uint<64> > &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingDataWidthConverter_Batch<InWidth, OutWidth, NumInWords>(in0_V, out_V, numReps);
}
