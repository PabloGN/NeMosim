#ifndef NEMO_CPU_MAPPER_HPP
#define NEMO_CPU_MAPPER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <set>
#include <nemo/types.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/Mapper.hpp>

namespace nemo {
	namespace cpu {

class Mapper : public nemo::Mapper<nidx_t, nidx_t>
{
	public :

		Mapper(const network::Generator& net) :
			m_offset(0),
			m_ncount(0)
		{
			if(net.neuronCount() > 0) {
				m_offset = net.minNeuronIndex();
				m_ncount = net.maxNeuronIndex() - net.minNeuronIndex() + 1;
			}
		}

		nidx_t addGlobal(const nidx_t& global) {
			m_validGlobal.insert(global);
			return localIdx(global);
		}

		/* Convert global neuron index to local */
		nidx_t localIdx(const nidx_t& global) const {
			assert(global >= m_offset);
			assert(global - m_offset < m_ncount);
			return global - m_offset;
		}
		
		/* Convert local neuron index to global */
		nidx_t globalIdx(const nidx_t& local) const {
			assert(local < m_ncount);
			return local + m_offset;
		}

		/*! \copydoc nemo::Mapper::neuronsInValidRange */
		unsigned neuronsInValidRange() const {
			return m_ncount;
		}

		nidx_t minLocalIdx() const {
			return m_offset;
		}

		nidx_t maxLocalIdx() const {
			return m_offset + m_ncount - 1;
		}

		bool validGlobal(const nidx_t& global) const {
			return m_validGlobal.count(global) == 1;
		}

		bool validLocal(const nidx_t& local) const {
			return validGlobal(globalIdx(local));
		}

	private :

		nidx_t m_offset;

		unsigned m_ncount;

		std::set<nidx_t> m_validGlobal;
};

	}
}

#endif
