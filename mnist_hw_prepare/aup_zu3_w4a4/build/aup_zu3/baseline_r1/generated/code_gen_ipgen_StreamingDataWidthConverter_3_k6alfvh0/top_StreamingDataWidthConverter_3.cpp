
#define AP_INT_MAX_W 128

#include "bnn-library.h"

// includes for network parameters
#include "streamtools.h"

// defines for network parameters
#define InWidth 128 
#define OutWidth 4 
#define NumInWords 16 
#define numReps 1

void StreamingDataWidthConverter_3(hls::stream<ap_uint<128> > &in0_V, hls::stream<ap_uint<4> > &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingDataWidthConverter_Batch<InWidth, OutWidth, NumInWords>(in0_V, out_V, numReps);
}
