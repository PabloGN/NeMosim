#ifndef NEMO_NEURONS_HPP
#define NEMO_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <deque>

#include <nemo/config.h>
#include "NeuronType.hpp"

namespace nemo {

	namespace network {
		class NetworkImpl;
	}

/*! \brief collection of neurons of the same type
 *
 * A network consisting of neurons of multiple types can be created from
 * several instances of this class.
 */
class Neurons 
{
	public :
		
		Neurons(const NeuronType&);

		/*! Add a new neuron
		 *
		 * \param fParam array of floating point parameters
		 * \param fState array of floating point state variables
		 *
		 * \return local index (wihtin this class) of the newly added neuron
		 *
		 * \pre the input arrays have the lengths specified by the neuron type
		 * 		used when this object was created.
		 */
		size_t add(const float fParam[], const float fState[]);

		/*! Modify an existing neuron
		 *
		 * \param local neuron index, as returned by \a add
		 * \param fParam array of floating point parameters
		 * \param fState array of floating point state variables
		 *
		 * \pre nidx refers to a valid neuron in this collection
		 * \pre the input arrays have the lengths specified by the neuron type
		 * 		used when this object was created.
		 */
		void set(size_t nidx, const float fParam[], const float fState[]);

		/*! \copydoc NetworkImpl::getNeuronParameter */
		float getParameter(size_t nidx, unsigned pidx) const;

		/*! \copydoc NetworkImpl::getNeuronState */
		float getState(size_t nidx, unsigned sidx) const;

		/*! \copydoc NetworkImpl::setNeuronParameter */
		void setParameter(size_t nidx, unsigned pidx, float val);

		/*! \copydoc NetworkImpl::setNeuronState */
		void setState(size_t nidx, unsigned sidx, float val);

		/*! \return number of neurons in this collection */
		size_t size() const { return m_size; }

		/*! \return neuron type common to all neurons in this collection */
		const NeuronType& type() const { return m_type; }

	private :

		/* Neurons are stored in several Structure-of-arrays, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 * invariant that the shapes are the same. */
		std::vector< std::deque<float> > mf_param;
		std::vector< std::deque<float> > mf_state;

		size_t m_size;

		NeuronType m_type;

		/*! \return parameter index after checking its validity */
		unsigned parameterIndex(unsigned i) const;

		/*! \return state variable index after checking its validity */
		unsigned stateIndex(unsigned i) const;

		friend class nemo::network::NetworkImpl;
};

}

#endif