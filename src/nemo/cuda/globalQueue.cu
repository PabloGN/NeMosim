#ifndef NEMO_CUDA_GLOBAL_QUEUE_CU
#define NEMO_CUDA_GLOBAL_QUEUE_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "globalQueue.cu_h"




/*! Return offset into full buffer data structure to beginning of buffer for a
 * particular targetPartition and a particular delay. */
__device__
unsigned
gq_bufferStart(unsigned targetPartition, unsigned slot, size_t pitch)
{
	return (targetPartition * 2 + slot) * pitch;
}



/*! Return a single entry from the global queue
 *
 * \param slot double buffer entry slot (0 or 1)
 * \param offset word offset into queue
 * \param g_queue full global queue structure in gmem
 */
__device__
gq_entry_t
gq_read(unsigned slot, unsigned offset, const gq_dt& gq)
{
	return gq.data[gq_bufferStart(CURRENT_PARTITION, slot, gq.pitch) + offset];
}


/*! \return incoming spike group from a particular source */
__device__ unsigned gq_warpOffset(gq_entry_t in) { return in; }


/*! \return address into matrix with number of incoming synapse groups
 * \param targetPartition
 * \param slot read or write slot
 *
 * \see readBuffer writeBuffer
 */
__device__
size_t
gq_fillOffset(unsigned targetPartition, unsigned slot)
{
	return targetPartition * 2 + slot;
}


#endif
