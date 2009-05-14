
/*! \brief GPU/CUDA kernel for neural simulation using Izhikevich's model
 * 
 * The entry point for the kernel is 'step' which will do one or more
 * integrate-and-fire step.  
 *
 * \author Andreas Fidjeland
 */

#include <cutil.h>
#include <device_functions.h>
#include <stdio.h>
#include <assert.h>
extern "C" {
#include "kernel.h"
}

#include "time.hpp"
#include "error.cu"
#include "log.hpp"
#include "L1SpikeQueue.hpp"
#include "connectivityMatrix.cu"
#include "L1SpikeQueue.cu"
#include "FiringProbe.hpp"
#include "firingProbe.cu"
#include "partitionConfiguration.cu"
#include "RuntimeData.hpp"
#include "CycleCounters.hpp"
#include "ConnectivityMatrix.hpp"
#include "cycleCounting.cu"
#include "ThalamicInput.hpp"

#define SLOW_32B_INTS

#ifdef SLOW_32B_INTS
#define mul24(a,b) __mul24(a,b)
#else
#define mul24(a,b) a*b
#endif



unsigned int
maxPartitionSize(int useSTDP)
{
    return useSTDP == 0 ? MAX_PARTITION_SIZE : MAX_PARTITION_SIZE_STDP;
}

//-----------------------------------------------------------------------------


/*! \param chunk 
 *		Each thread processes n neurons. The chunk is the index (<n) of the
 *		neuron currently processed by the thread.
 * \return 
 *		true if the thread should be active when processing the specified
 * 		neuron, false otherwise.
 */
__device__
bool 
activeNeuron(int chunk, int s_partitionSize)
{
	return threadIdx.x + chunk*THREADS_PER_BLOCK < s_partitionSize;
}



//=============================================================================
// Firing 
//=============================================================================


/*! The external firing stimulus is densely packed with one bit per neuron.
 * Thus only the low-order threads need to read this data, and we need to
 * sync.  */  
__device__
void
loadExternalFiring(
        bool hasExternalInput,
		int s_partitionSize,
		size_t pitch,
		uint32_t* g_firing,
		uint32_t* s_firing)
{
	if(threadIdx.x < DIV_CEIL(s_partitionSize, 32)) {
		if(hasExternalInput) {
			s_firing[threadIdx.x] =
                g_firing[blockIdx.x * pitch + threadIdx.x];
		} else {
			s_firing[threadIdx.x] = 0;
		}
	}
	__syncthreads();
}



//=============================================================================
// Current buffer
//=============================================================================
// The current buffer stores (in shared memory) the accumulated current for
// each neuron in the block 
//=============================================================================


/* We need two versions of some device functions... */
#include "thalamicInput.cu"
#define STDP
#include "kernel.cu"
#undef STDP
#include "kernel.cu"

/* Force all asynchronously launced kernels to complete before returning */
__host__
void
syncSimulation(RTDATA rtdata)
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}



/* Copy network data and configuration to device, if this has not already been
 * done */
__host__
void
configureDevice(RTDATA rtdata)
{
	/* This would have been tidier if we did all the handling inside rtdata.
	 * However, problems with copying to constant memory in non-cuda code
	 * prevents this. */
	if(rtdata->deviceDirty()) {
        clearAssertions();
		rtdata->moveToDevice();
		configurePartition(c_maxL0SynapsesPerDelay, 
			rtdata->cm(CM_L0)->maxSynapsesPerDelay());
		configurePartition(c_maxL0RevSynapsesPerDelay, 
			rtdata->cm(CM_L0)->maxReverseSynapsesPerDelay());
		configurePartition(c_maxL1SynapsesPerDelay,
			rtdata->cm(CM_L1)->maxSynapsesPerDelay());
		configurePartition(c_maxL1RevSynapsesPerDelay,
			rtdata->cm(CM_L1)->maxReverseSynapsesPerDelay());
		if(rtdata->usingSTDP()) {
			configureStdp(
				rtdata->m_stdpTauP,
				rtdata->m_stdpTauD,
				rtdata->m_stdpAlphaP,
				rtdata->m_stdpAlphaD);
		}
        rtdata->setStart();
	}
}



/*! Wrapper for the __global__ call that performs a single simulation step */
__host__
status_t
step(	ushort cycle,
        //! \todo make all these unsigned
        //! \todo just get partitionCount from RTDATA
		int partitionCount,
		int maxPartitionSize,
		int substeps,
		int applySTDP,
		float stdpReward,
		// External firing (sparse)
		size_t extFiringCount,
		const int* extFiringCIdx, 
		const int* extFiringNIdx, 
		// Run-time data
		RTDATA rtdata)
{
	/* The L1 spike queues rotate among 'maxDelay' entries. We keep track of
	 * the entry corresponding to time 0, i.e. the spikes to be delivered right
	 * away. */
	static int currentL1SQBase = 0;

	rtdata->firingProbe->checkOverflow();
    rtdata->step();

	configureDevice(rtdata); // only done on first invocation

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	if(rtdata->usingSTDP() && applySTDP) {
        //! \todo deal especially with the case where reward=0, can do this faster

        /* Potentation and depression are handled separately due to different
         * traversal order. However, the order here is important; We want to
         * clamp the weight values at some maximum and minimum, and these can
         * only be set once both potentiation and depression have been
         * considered (done in LTP). We *could* do this clamping in a separate
         * stage for clarity, but would have to pay for this by doing another
         * whole iteration throught the connectivity matrix. */
		addL0LTD<<<dimGrid, dimBlock>>>(
				stdpReward,
				maxPartitionSize,
                rtdata->maxDelay(),
				rtdata->pitch32(),
				rtdata->cm(CM_L0)->deviceDelayBits(),
				rtdata->cm(CM_L0)->deviceSynapsesD(),
				rtdata->cm(CM_L0)->synapsePitchD(),
				rtdata->cm(CM_L0)->submatrixSize());
        addL0LTP<<<dimGrid, dimBlock>>>(
                stdpReward,
                maxPartitionSize,
                rtdata->maxDelay(),
                rtdata->pitch32(),
                rtdata->cm(CM_L0)->arrivalBits(),
                rtdata->cm(CM_L0)->deviceSynapsesD(),
                rtdata->cm(CM_L0)->synapsePitchD(),
                rtdata->cm(CM_L0)->submatrixSize(),
                rtdata->cm(CM_L0)->reverseConnectivity(),
                rtdata->cm(CM_L0)->reversePitch(),
                rtdata->cm(CM_L0)->reverseSubmatrixSize());
		if(stdpReward != 0.0f) {
			constrainL0Weights<<<dimGrid, dimBlock>>>(
				rtdata->stdpMaxWeight(),
				maxPartitionSize,
                rtdata->maxDelay(),
				rtdata->pitch32(),
				rtdata->cm(CM_L0)->deviceDelayBits(),
				rtdata->cm(CM_L0)->deviceSynapsesD(),
				rtdata->cm(CM_L0)->synapsePitchD(),
				rtdata->cm(CM_L0)->submatrixSize());
		}
	}

#ifdef VERBOSE
	static uint scycle = 0;
	fprintf(stdout, "cycle %u/%u\n", scycle++, rtdata->stdpCycle());
#endif

	uint32_t* d_extFiring = rtdata->setFiringStimulus(
            extFiringCount, 
			extFiringCIdx, 
			extFiringNIdx);

	if(rtdata->usingSTDP()) {
		step_STDP<<<dimGrid, dimBlock>>>(
				maxPartitionSize,
				substeps, 
                rtdata->cycle(),
                rtdata->maxDelay(),
				rtdata->recentFiring->deviceData(),
				// STDP
				rtdata->recentArrivals->deviceData(),
				rtdata->stdpCycle(),
				rtdata->cm(CM_L0)->reverseConnectivity(),
				rtdata->cm(CM_L0)->reversePitch(),
				rtdata->cm(CM_L0)->reverseSubmatrixSize(),
				rtdata->cm(CM_L0)->arrivalBits(),
				// neuron parameters
				rtdata->neuronParameters->deviceData(),
                rtdata->thalamicInput->deviceRngState(),
                rtdata->thalamicInput->deviceSigma(),
				rtdata->neuronParameters->size(),
				// L0 connectivity matrix
				rtdata->cm(CM_L0)->deviceSynapsesD(),
				rtdata->cm(CM_L0)->deviceDelayBits(),
				rtdata->cm(CM_L0)->synapsePitchD(), // word pitch
				rtdata->cm(CM_L0)->submatrixSize(),
				// L1 connectivity matrix
				rtdata->cm(CM_L1)->deviceSynapsesD(),
				rtdata->cm(CM_L0)->deviceDelayBits(),
				rtdata->cm(CM_L1)->synapsePitchD(),
				rtdata->cm(CM_L0)->submatrixSize(),
				// L1 spike queue
				rtdata->spikeQueue->data(),
				rtdata->spikeQueue->pitch(),
				rtdata->spikeQueue->heads(),
				rtdata->spikeQueue->headPitch(),
				currentL1SQBase,
				// firing stimulus
				d_extFiring,
				rtdata->firingStimulus->wordPitch(),
				rtdata->pitch32(),
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				// Firing output
				cycle,
				rtdata->firingProbe->deviceBuffer(),
				rtdata->firingProbe->deviceNextFree());
	} else {
		step_static<<<dimGrid, dimBlock>>>(
				maxPartitionSize,
				substeps, 
                rtdata->cycle(),
                rtdata->maxDelay(),
				rtdata->recentFiring->deviceData(),
				rtdata->neuronParameters->deviceData(),
                rtdata->thalamicInput->deviceRngState(),
                rtdata->thalamicInput->deviceSigma(),
                //! \todo get size directly from rtdata
				rtdata->neuronParameters->size(),
				// L0 connectivity matrix
				rtdata->cm(CM_L0)->deviceSynapsesD(),
				rtdata->cm(CM_L0)->deviceDelayBits(),
				rtdata->cm(CM_L0)->synapsePitchD(), // word pitch
				rtdata->cm(CM_L0)->submatrixSize(),
				// L1 connectivity matrix
				rtdata->cm(CM_L1)->deviceSynapsesD(),
				rtdata->cm(CM_L1)->deviceDelayBits(),
				rtdata->cm(CM_L1)->synapsePitchD(),
				rtdata->cm(CM_L1)->submatrixSize(),
				// L1 spike queue
				rtdata->spikeQueue->data(),
				rtdata->spikeQueue->pitch(),
				rtdata->spikeQueue->heads(),
				rtdata->spikeQueue->headPitch(),
				currentL1SQBase,
				// firing stimulus
				d_extFiring,
				rtdata->firingStimulus->wordPitch(),
				rtdata->pitch32(),
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				// Firing output
				cycle,
				rtdata->firingProbe->deviceBuffer(),
				rtdata->firingProbe->deviceNextFree());
	}

    if(assertionsFailed(partitionCount, cycle)) {
        fprintf(stderr, "checking assertions\n");
        clearAssertions();
        return KERNEL_ASSERTION_FAILURE;
    }

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess) {
		WARNING("%s", cudaGetErrorString(status));
		LOG("", "Kernel parameters: <<<%d, %d>>>\n",
			dimGrid.x, dimBlock.x);
		return KERNEL_INVOCATION_ERROR;
	}

	if(rtdata->maxDelay()) {
		currentL1SQBase = (currentL1SQBase + 1) % (rtdata->maxDelay() + 1);
	}

	return KERNEL_OK;
}
