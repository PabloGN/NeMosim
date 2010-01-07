#ifndef APPLY_STDP_CU
#define APPLY_STDP_CU

#include <cuda.h>
#include "cycleCounting.cu"
#include "connectivityMatrix.cu"
#include "util.h"
#include "dispatchTable.cu"


/*! Apply STDP 
 * 
 * The STDP statistics are stored in reverse CM order with potentiation and
 * depression already combined. This data needs to be re-ordered into the
 * forward order when updating the weight.
 *
 * The new weight is limited by a maximum weight, and is not allowed to fall
 * below 0.
 *
 * prefix r: reverse matrix
 * prefix f: forward matrix
 */
__global__
void
applySTDP_(
#ifdef KERNEL_TIMING
	unsigned long long* g_cc,
	size_t ccPitch,
#endif
	float reward,
	uint cmIdx,      // L0 or L1
	float maxWeight, // for excitatory synapses
	float minWeight) // for inhibitory synapses
	/*! \note reverse connectivity addresses are found in constant memory,
	 * while forward connectivity addresses are found in texture memory */
{
	SET_COUNTER(s_ccApplySTDP, 0);

	__shared__ uint s_chunkCount;
	__shared__ uint s_partitionSize;

	float* gr_stdp = cmIdx == 0
		? (float*) cr0_stdp[CURRENT_PARTITION]
		: (float*) cr1_stdp[CURRENT_PARTITION];

	uint r_pitch = cmIdx == 0
		? cr0_pitch[CURRENT_PARTITION]
		: cr1_pitch[CURRENT_PARTITION];

	uint32_t* gr_address = cmIdx == 0
		? (uint32_t*) cr0_address[CURRENT_PARTITION]
		: (uint32_t*) cr1_address[CURRENT_PARTITION];

	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		s_chunkCount = DIV_CEIL(r_pitch, THREADS_PER_BLOCK);
	}
	__syncthreads();

	for(uint target=0; target < s_partitionSize; ++target) {
		for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

			uint r_sidx = chunk * THREADS_PER_BLOCK + threadIdx.x;

			if(r_sidx < r_pitch) {

				size_t gr_offset = target * r_pitch + r_sidx;
				uint rsynapse = gr_address[gr_offset];

				if(rsynapse != INVALID_REVERSE_SYNAPSE) {

					/*! \todo try using atomicExch here instead. For m=20
					 * atomicExch is slightly faster, but this will probably
					 * work less well for e.g. m=1000 */
					float w_diff = gr_stdp[gr_offset] * reward;
					//float w_diff = reward * __int_as_float(atomicExch(gr_stdp + gr_offset, __float_as_int(0.0f)));

					if(w_diff != 0.0f) {

						gr_stdp[gr_offset] = 0.0f;

						//! \todo load this into smem exactly once
						fcm_ref_t fcm;
						if(cmIdx == 0) {
							fcm = getFCM2(sourcePartition(rsynapse), CURRENT_PARTITION, r_delay0(rsynapse));
						} else {
							fcm = getFCM(cmIdx, sourcePartition(rsynapse), r_delay0(rsynapse));
						}
						ASSERT(f0_base(fcm) != 0x0);
						ASSERT(forwardIdx(rsynapse) < f0_pitch(fcm));
						//! \todo share method with kernel.cu:synapesAddress2
						size_t gf_offset = f_synapseOffset(sourceNeuron(rsynapse), f0_pitch(fcm), forwardIdx(rsynapse));
						ASSERT(gf_offset < f0_size(fcm));
						float* gf_weight = f0_weights(fcm);

						float w_old = gf_weight[gf_offset];
						float w_new = 0.0f;
						if(w_old > 0.0f) {
							w_new = fmin(maxWeight, fmax(w_old + w_diff, 0.0f));
						} else if(w_old < 0.0f) {
							w_new = fmin(0.0f, fmax(w_old + w_diff, minWeight));
						}

						if(w_old != w_new) {
							gf_weight[gf_offset] = w_new;
							DEBUG_MSG("stdp (%u-%u -> %u-%u) %f %+f = %f\n",
									sourcePartition(rsynapse), sourceNeuron(rsynapse),
									CURRENT_PARTITION, target,
									w_old, w_diff, w_new);
						}
					}
				}
			}
		}
		//! \todo remove sync?
		__syncthreads();
	}

	SET_COUNTER(s_ccApplySTDP, 1);
	WRITE_COUNTERS(s_ccApplySTDP, g_cc, ccPitch, 2);
}

#endif
