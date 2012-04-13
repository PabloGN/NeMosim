#ifndef NEMO_RUNTIME_DELAYS_HPP
#define NEMO_RUNTIME_DELAYS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/unordered_map.hpp>

#include <nemo/internal_types.h>

namespace nemo {

	namespace construction {
		class Delays;
	}

	namespace runtime {


/*! Per-neuron collection of outgoing delays (run-time) */
class Delays
{
	public :

		/*! Create a run-time represenation of the outgoing delays
		 *
		 * \param neuronCount
		 * 		number of neurons in the network
		 */
		Delays(size_t neuronCount, const construction::Delays&);

		delay_t maxDelay() const { return m_maxDelay; }

		typedef std::vector<delay_t>::const_iterator const_iterator;

		/*! \param neuron
		 * 		global neuron index
		 *  \return
		 *  	iterator pointing to first delay for the \a neuron
		 */
		const_iterator begin(nidx_t neuron) const;

		/*! \param neuron
		 * 		global neuron index
		 *  \return
		 *  	iterator pointing beyond the last delay for the \a neuron
		 */
		const_iterator end(nidx_t neuron) const;

		const std::vector<uint64_t>& delayBits() const { return m_bits; }

	private :

		/*! \note Did some experimentation with replacing std::vector with raw
		 * arrays but this did not improve performance appreciably. The main
		 * cost of using this data structure is in calling find on the hash
		 * table */
		boost::unordered_map<nidx_t, std::vector<delay_t> > m_data;

		/* Delays stored in compact form, supporting up to 64 delays. One bit
		 * per delay with LSb the lowest delay. */
		std::vector<uint64_t> m_bits;

		delay_t m_maxDelay;

		Delays(const Delays& );
		Delays& operator=(const Delays&);

		bool hasSynapses(nidx_t source) const;

		/*! \return a bitwise representation of the delays for a single source.
		 * The least significant bit corresponds to a delay of 1 */
		uint64_t delayBits(nidx_t source) const;

};

}	}

#endif
