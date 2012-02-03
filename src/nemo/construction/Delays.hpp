#ifndef NEMO_CONSTRUCTION_DELAYS_HPP
#define NEMO_CONSTRUCTION_DELAYS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <set>

#include <nemo/internal_types.h>

namespace nemo {

	class OutgoingDelays;

	namespace construction {


/*! Per-neuron collection of outoing delays */
class Delays
{
	public :

		Delays() : m_maxDelay(0) { }

		/*! \param neuron global index
		 *  \param delay
		 */
		void addDelay(nidx_t neuron, delay_t delay) {
			m_delays[neuron].insert(delay);
			m_maxDelay = std::max(m_maxDelay, delay);
		}

		delay_t maxDelay() const { return m_maxDelay; }

	private :

		friend class nemo::OutgoingDelays;

		std::map<nidx_t, std::set<delay_t> > m_delays;

		delay_t m_maxDelay;
};

	}
}

#endif
