
#define AP_INT_MAX_W 128

#include "bnn-library.h"

// includes for network parameters
#include "streamtools.h"

// defines for network parameters
#define InWidth 4 
#define OutWidth 128 
#define NumInWords 2048 
#define numReps 1

void StreamingDataWidthConverter_2(hls::stream<ap_uint<4> > &in0_V, hls::stream<ap_uint<128> > &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingDataWidthConverter_Batch<InWidth, OutWidth, NumInWords>(in0_V, out_V, numReps);
}
