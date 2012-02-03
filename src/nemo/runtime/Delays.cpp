/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/construction/Delays.hpp>
#include <nemo/exception.hpp>
#include <nemo/types.hpp>

#include "Delays.hpp"

namespace nemo {
	namespace runtime {


Delays::Delays(const construction::Delays& acc) :
	m_maxDelay(0)
{
	m_maxDelay = acc.maxDelay();

	typedef std::map<unsigned, std::set<unsigned> >::const_iterator it;
	for(it i = acc.m_delays.begin(), i_end = acc.m_delays.end(); i != i_end; ++i) {
		const std::set<unsigned>& delays = i->second;
		unsigned neuron = i->first;
		m_data[neuron] = std::vector<delay_t>(delays.begin(), delays.end());
	}
}



Delays::const_iterator
Delays::begin(nidx_t source) const
{
	boost::unordered_map<nidx_t, std::vector<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.begin();
}



Delays::const_iterator
Delays::end(nidx_t source) const
{
	boost::unordered_map<nidx_t, std::vector<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.end();
}


bool
Delays::hasSynapses(nidx_t source) const
{
	return m_data.find(source) != m_data.end();
}


uint64_t
Delays::delayBits(nidx_t source) const
{
	uint64_t bits = 0;
	if(hasSynapses(source)) {
		for(const_iterator d = begin(source), d_end = end(source); d != d_end; ++d) {
			bits = bits | (uint64_t(0x1) << uint64_t(*d - 1));
		}
	}
	return bits;
}

}	}
