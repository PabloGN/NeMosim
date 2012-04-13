#ifndef NEMO_IZHIKEVICH_HPP
#define NEMO_IZHIKEVICH_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/Network.hpp>

namespace nemo {
	namespace izhikevich {


/*! Network of Izhikevich neurons */
class Network : public nemo::Network
{
	public :

		Network() {
			m_stype = addSynapseType();
			m_ntype = addNeuronType("Izhikevich", 1, &m_stype);
		}

		/*! \brief Add a single Izhikevich neuron to the network
		 *
		 * The neuron uses the Izhikevich neuron model. See E. M. Izhikevich
		 * "Simple model of spiking neurons", \e IEEE \e Trans. \e Neural \e
		 * Networks, vol 14, pp 1569-1572, 2003 for a full description of the
		 * model and the parameters.
		 *
		 * \param idx
		 *		Neuron index. This should be unique
		 * \param a
		 *		Time scale of the recovery variable \a u
		 * \param b
		 *		Sensitivity to sub-threshold fluctutations in the membrane
		 *		potential \a v
		 * \param c
		 *		After-spike reset value of the membrane potential \a v
		 * \param d
		 *		After-spike reset of the recovery variable \a u
		 * \param u
		 *		Initial value for the membrane recovery variable
		 * \param v
		 * 		Initial value for the membrane potential
		 * \param sigma
		 *		Parameter for a random gaussian per-neuron process which
		 *		generates random input current drawn from an N(0,\a sigma)
		 *		distribution. If set to zero no random input current will be
		 *		generated.
		 */
		void addNeuron(unsigned nidx,
				float a, float b, float c, float d,
				float u, float v, float sigma = 0.0f) {
			static float args[7] = {a, b, c, d, sigma, u, v};
			nemo::Network::addNeuron(m_ntype, nidx, 7, args);
		}

		void addNeuron(unsigned idx, unsigned nargs, const float args[]) {
			nemo::Network::addNeuron(m_ntype, idx, nargs, args);
		}

		using nemo::Network::addNeuron;

		/*! Change parameters/state variables of a single existing
		 *  Izhikevich-type neuron
		 *
		 * The parameters are the same as for \a nemo::Network::addNeuron
		 */
		void setNeuron(unsigned nidx,
				float a, float b, float c, float d,
				float u, float v, float sigma = 0.0f) {
			static float args[7] = {a, b, c, d, sigma, u, v};
			nemo::Network::setNeuron(nidx, 7, args);
		}

		unsigned synapseType() const { return m_stype; }

	private :

		unsigned m_ntype;
		unsigned m_stype;
};


}	}

#endif
