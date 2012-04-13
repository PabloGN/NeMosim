#ifndef NEMO_CUDA_CURRENT_CU
#define NEMO_CUDA_CURRENT_CU

/*! \file current.cu Functions related to neuron input current */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


/*! \brief Load externally provided current stimulus from gmem
 *
 * The user can provide per-neuron current stimulus
 * (nemo::cuda::Simulation::addCurrentStimulus).
 *
 * \param[in] partition
 *		\i global index of partition
 * \param[in] psize
 *		number of neurons in current partition
 * \param[in] pitch
 *		pitch of g_current, i.e. distance in words between each partitions data
 * \param[in] g_current
 *		global memory vector containing current for all neurons in partition.
 *		If set to NULL, no input current will be delivered.
 * \param[out] s_current
 *		shared memory vector which will be set to contain input stimulus (or
 *		zero, if there's no stimulus).
 *
 * \pre neuron < size of current partition
 * \pre all shared memory buffers have at least as many entries as the size of
 * 		the current partition
 *
 * \see nemo::cuda::Simulation::addCurrentStimulus
 */
__device__
void
loadCurrentStimulus(
		unsigned partition,
		unsigned psize,
		size_t pitch,
		const float* g_current,
		float* s_current)
{
	if(g_current != NULL) {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			unsigned pstart = partition * pitch;
			float stimulus = g_current[pstart + neuron];
			s_current[neuron] = stimulus;
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
			DEBUG_MSG_SYNAPSE("c%u %u-%u: +%f (external)\n",
					s_cycle, partition, neuron, g_current[pstart + neuron]);
#endif
		}
	} else {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			s_current[neuron] = 0;
		}
	}
	__syncthreads();
}



/* \return address for a given current accumulator for a given partition
 *
 * \param pcount
 *		total number of partitions in the network
 * \param partition
 *		index of partition of interest
 * \param accIndex
 *		accumulator index. Accumulators are indexed from 0, and there's
 *		typically one accumulator per synapse type.
 */
__device__
float*
accumulator(float* g_base, unsigned pcount, unsigned partition, unsigned accIndex, size_t pitch32)
{
	return g_base + (accIndex * pcount + partition) * pitch32;
}



#endif
