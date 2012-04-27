#ifndef NEMO_CUDA_PLUGIN_SYNAPSE_MODEL_HPP
#define NEMO_CUDA_PLUGIN_SYNAPSE_MODEL_HPP

/* Common API for CUDA synapse model plugins */

#include <cuda_runtime.h>

#include <nemo/cuda/fcm.cu_h>
#include <nemo/cuda/parameters.cu_h>
#include <nemo/cuda/globalQueue.cu_h>

#ifdef __cplusplus
extern "C" {
#endif

typedef cudaError_t cuda_gather_t(
		cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		param_t* d_globalParameters,
		float* d_current,
		const fcm_dt& d_fcm,
		gq_dt d_gq);

#ifdef __cplusplus
}
#endif

#endif
