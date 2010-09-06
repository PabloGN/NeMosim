/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>

#include "FiringOutput.hpp"
#include "bitvector.cu_h"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {

FiringOutput::FiringOutput(const Mapper& mapper):
	m_pitch(0),
	m_bufferedCycles(0),
	md_allocated(0),
	m_mapper(mapper)
{
	size_t width = BV_BYTE_PITCH;
	size_t height = m_mapper.partitionCount();

	size_t bytePitch;
	uint32_t* d_buffer;
	d_mallocPitch((void**)(&d_buffer), &bytePitch, width, height, "firing output");
	md_buffer = boost::shared_ptr<uint32_t>(d_buffer, d_free);
	d_memset2D(d_buffer, bytePitch, 0, height);
	m_pitch = bytePitch / sizeof(uint32_t);

	size_t md_allocated = bytePitch * height;
	uint32_t* h_buffer;

	mallocPinned((void**) &h_buffer, md_allocated);
	mh_buffer = boost::shared_ptr<uint32_t>(h_buffer, freePinned);
	memset(h_buffer, 0, md_allocated);
}



void
FiringOutput::sync()
{
	memcpyFromDevice(mh_buffer.get(), md_buffer.get(),
				m_mapper.partitionCount() * m_pitch * sizeof(uint32_t));
	populateSparse(m_bufferedCycles, mh_buffer.get(), m_cycles, m_neuronIdx);
	m_bufferedCycles += 1;
}



unsigned
FiringOutput::readFiring(
		const std::vector<unsigned>** cycles,
		const std::vector<unsigned>** neuronIdx)
{
	m_cyclesOut = m_cycles;
	m_neuronIdxOut = m_neuronIdx;
	*cycles = &m_cyclesOut;
	*neuronIdx = &m_neuronIdxOut;
	unsigned readCycles = m_bufferedCycles;
	m_bufferedCycles = 0;
	m_cycles.clear();
	m_neuronIdx.clear();
	return readCycles;
}



void
FiringOutput::populateSparse(
		unsigned cycle,
		const uint32_t* hostBuffer,
		std::vector<unsigned>& firingCycle,
		std::vector<unsigned>& neuronIdx)
{
	unsigned pcount = m_mapper.partitionCount();

	for(size_t partition=0; partition < pcount; ++partition) {

		size_t partitionOffset = partition * m_pitch;

		for(size_t nword=0; nword < m_pitch; ++nword) {

			/* Within a partition we might go into the padding part of the
			 * firing buffer. We rely on the device not leaving any garbage
			 * in the unused entries */
			uint32_t word = hostBuffer[partitionOffset + nword];
			if(word == 0)
				continue;

			for(size_t nbit=0; nbit < 32; ++nbit) {
				bool fired = (word & (1 << nbit)) != 0;
				if(fired) {
					firingCycle.push_back(cycle);
					neuronIdx.push_back(m_mapper.hostIdx(partition, nword*32 + nbit));
				}
			}
		}
	}
}



void
FiringOutput::flushBuffer()
{
	m_cycles.clear();
	m_neuronIdx.clear();
	m_bufferedCycles = 0;
}

	} // end namespace cuda
} // end namespace nemo
