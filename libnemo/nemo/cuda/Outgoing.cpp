/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Outgoing.hpp"

#include <vector>
#include <cuda_runtime.h>

#include <nemo/util.h>
#include <nemo/bitops.h>

#include "WarpAddressTable.hpp"
#include "device_memory.hpp"
#include "exception.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {

Outgoing::Outgoing() : m_pitch(0), m_allocated(0), m_maxIncomingWarps(0) {}


Outgoing::Outgoing(size_t partitionCount, const WarpAddressTable& wtable) :
		m_pitch(0),
		m_allocated(0),
		m_maxIncomingWarps(0)
{
	init(partitionCount, wtable);
}



bool
compare_warp_counts(
		const std::pair<pidx_t, size_t>& lhs,
		const std::pair<pidx_t, size_t>& rhs)
{
	return lhs.second < rhs.second;
}


void
Outgoing::init(size_t partitionCount, const WarpAddressTable& wtable)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE * MAX_DELAY;

	/* allocate device and host memory for row lengths */
	outgoing_addr_t* d_addr = NULL;
	d_malloc((void**)&d_addr, height * sizeof(outgoing_addr_t), "outgoing spikes (row lengths)");
	md_rowLength = boost::shared_ptr<outgoing_addr_t>(d_addr, d_free);
	m_allocated = height * sizeof(outgoing_addr_t);
	std::vector<outgoing_addr_t> h_addr(height, make_outgoing(0,0));

	/* allocate temporary host memory for table */
	std::vector<outgoing_t> h_data;

	/* accumulate the number of incoming warps for each partition, such that we
	 * can size the global queue correctly */
	std::map<pidx_t, size_t> incoming;

	size_t allocated = 0; // words, so far
	unsigned wpitch = 0;  // maximum, so far

	/* populate host memory */
	for(WarpAddressTable::iterator ti = wtable.begin(); ti != wtable.end(); ++ti) {

		const WarpAddressTable::key& k = ti->first;

		pidx_t sourcePartition = get<0>(k);
		nidx_t sourceNeuron = get<1>(k);
		delay_t delay1 = get<2>(k);

		/* Allocate memory for this row. Add padding to ensure each row starts
		 * at warp boundaries */
		unsigned nWarps = wtable.warpsPerNeuronDelay(sourcePartition, sourceNeuron, delay1);
		unsigned nWords = ALIGN(nWarps, WARP_SIZE);
		wpitch = std::max(wpitch, nWords);
		assert(nWords >= nWarps);
		h_data.resize(allocated + nWords, INVALID_OUTGOING);

		unsigned rowBegin = allocated;
		unsigned col = 0;

		/* iterate over target partitions in a row */
		const WarpAddressTable::row_t& r = ti->second;
		for(WarpAddressTable::row_iterator ri = r.begin(); ri != r.end(); ++ri) {

			pidx_t targetPartition = ri->first;
			const std::vector<size_t>& warps = ri->second;
			size_t len = warps.size();
			incoming[targetPartition] += len;
			outgoing_t* p = &h_data[rowBegin + col];
			col += len;
			assert(col <= nWarps);

			/* iterate over warps specific to a target partition */
			for(std::vector<size_t>::const_iterator wi = warps.begin();
					wi != warps.end(); ++wi, ++p) {
				*p = make_outgoing(targetPartition, *wi);
			}
		}

		/* Set address info here, since both start and length are now known.
		 * Col points to next free entry, which is also the length. */
		size_t r_addr = outgoingAddrOffset(sourcePartition, sourceNeuron, delay1-1);
		h_addr[r_addr] = make_outgoing_addr(rowBegin, col);
		allocated += nWords;
	}

	memcpyToDevice(d_addr, h_addr);

	/* allocate device memory for table */
	if(allocated != 0) {
		outgoing_t* d_data = NULL;
		d_malloc((void**)&d_data, allocated*sizeof(outgoing_t), "outgoing spikes");
		md_arr = boost::shared_ptr<outgoing_t>(d_data, d_free);
		memcpyToDevice(d_data, h_data, allocated);
		m_allocated += allocated * sizeof(outgoing_t);
	}

	setConstants(wpitch);

	//! \todo compute this on forward pass (in WarpAddressTable)
	m_maxIncomingWarps = incoming.size() ? std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second : 0;
}



/*! Set outgoing parameters in constant memory on device
 *
 * In the inner loop in scatterGlobal the kernel processes potentially multiple
 * rows of outgoing data. We set the relevant loop parameters in constant
 * memory, namely the pitch (max width of row) and step (the number of rows a
 * thread block can process in parallel).
 *
 * It would be possible, and perhaps desirable, to store the pitch/row length
 * on a per-partition basis rather than use a global maximum.
 *
 * \todo store pitch/step on per-partition basis
 * \todo support handling of pitch greater than THREADS_PER_BLOCK
 */
void
Outgoing::setConstants(unsigned maxWarpsPerNeuronDelay)
{
	/* We need the step to exactly divide the pitch, in order for the inner
	 * loop in scatterGlobal to work out. */
	unsigned wpitch = std::max(1U, unsigned(ceilPowerOfTwo(maxWarpsPerNeuronDelay)));

	/* Additionally scatterGlobal assumes that wpitch <= THREADS_PER_BLOCK. It
	 * would possible to modify scatterGLobal to handle the other case as well,
	 * with different looping logic. Separate kernels might be more sensible. */
	assert_or_throw(wpitch <= THREADS_PER_BLOCK, "Outgoing pitch too wide");

	CUDA_SAFE_CALL(setOutgoingPitch(wpitch));

	unsigned step = THREADS_PER_BLOCK / wpitch;
	CUDA_SAFE_CALL(setOutgoingStep(step));

	assert_or_throw(step * wpitch == THREADS_PER_BLOCK, "Invalid outgoing pitch/step");
}



	} // end namespace cuda
} // end namespace nemo
