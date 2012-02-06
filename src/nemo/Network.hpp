#ifndef NEMO_NETWORK_HPP
#define NEMO_NETWORK_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <string>

#include <nemo/config.h>
#include <nemo/types.h>
#include <nemo/ReadableNetwork.hpp>

namespace nemo {

namespace mpi {
	class Master;
}

namespace network {
	class NetworkImpl;

}

class Simulation;
class SimulationBackend;
class Configuration;

/*! Networks are constructed by adding individual neurons, and single or groups
 * of synapses to the network. Neurons are given indices (from 0) which should
 * be unique for each neuron. When adding synapses the source or target neurons
 * need not necessarily exist yet, but should be defined before the network is
 * finalised. */
class NEMO_BASE_DLL_PUBLIC Network : public ReadableNetwork
{
	public :

		Network();

		~Network();

		/*! \brief Register a new neuron type with the network.
		 *
		 * \param name
		 * 		canonical name of the neuron type. The neuron type data is
		 * 		loaded from a plugin configuration file of the same name.
		 * \param nInputs
		 * 		number of different synapse input types this neuron model
		 * 		expects.
		 * \param inputs
		 * 		indices of the synapse types from which this neuron type
		 * 		receives input.
		 * \return
		 * 		index of the the neuron type, to be used when adding neurons.
		 *
		 * This function must be called before neurons of the specified type
		 * can be added to the network.
		 *
		 * \throws if any of the synapse types do not exist
		 * \throws if on a subsequent call the synapse types are not the same
		 */
		unsigned addNeuronType(const std::string& name,
				unsigned nInputs,
				const unsigned inputs[]);

		/*! \brief Register a new synapse type with the network.
		 *
		 * \return index of the synape type, to be used when adding synapses
		 *
		 * All synapses have the same fundamental type (simple additive).
		 *
		 * This function must be called before synapses of the specified type
		 * can be added to the network.
		 */
		unsigned addSynapseType(synapse_type t=NEMO_SYNAPSE_ADDITIVE);

		/*! \brief Add a neuron to the network
		 *
		 * \param type
		 * 		index of the neuron type, as returned by \a addNeuronType
		 * \param idx
		 * 		index of the neuron
		 * \param nargs
		 * 		length of \a args
		 * \param args
		 * 		parameters and state variables of the neuron (in that order)
		 *
		 * \pre The parameter and state arrays must have the dimensions
		 * 		matching the neuron type represented by \a type.
		 */
		void addNeuron(unsigned type, unsigned idx,
				unsigned nargs, const float args[]);

		/*! Set an existing neuron
		 *
		 * \param idx
		 * 		index of the neuron, as used when calling \a addNeuron
		 * \param nargs
		 * 		length of \a args
		 * \param args
		 * 		parameters and state variables of the neuron (in that order)
		 *
		 * \pre The parameter and state arrays must have the dimensions
		 * 		matching the neuron type specified when the neuron was first
		 * 		added.
		 */
		void setNeuron(unsigned idx, unsigned nargs, const float args[]);

		/* Add a single synapse and return its unique id */
		synapse_id addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		/* Add a single synapse and return its unique id
		 *
		 * \param typeIdx synapse type index, as returned by addSynapseType
		 */
		synapse_id addSynapse(
				unsigned typeIdx,
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight);

		/*! Get a single parameter for a single neuron
		 *
		 * \param neuron neuron index
		 * \param parameter parameter index
		 * \return parameter with index \a parameter.
		 *
		 * For the Izhikevich model the parameter indices are 0=a, 1=b, 2=c, 3=d, 4=sigma.
		 */
		float getNeuronParameter(unsigned neuron, unsigned parameter) const;

		/*! Get a single state variable for a single neuron
		 *
		 * \param neuron neuron index
		 * \param var variable index
		 * \return state variable with index \a var.
		 *
		 * For the Izhikevich model the variable indices are 0=u, 1=v.
		 */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*! Change a single parameter for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param parameter parameter index
		 * \param value new value of the state variable
		 *
		 * For the Izhikevich model the parameter indices are 0=a, 1=b, 2=c, 3=d, 4=sigma.
		 */
		void setNeuronParameter(unsigned neuron, unsigned parameter, float value);

		/*! Change a single state variable for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param var state variable index
		 * \param value new value of the state variable
		 *
		 * For the Izhikevich model variable indices 0=u, 1=v.
		 */
		void setNeuronState(unsigned neuron, unsigned var, float value);

		/*! \return target neuron id for a synapse */
		unsigned getSynapseTarget(const synapse_id&) const;

		/*! \return conductance delay for a synapse */
		unsigned getSynapseDelay(const synapse_id&) const;

		/*! \return weight for a synapse */
		float getSynapseWeight(const synapse_id&) const;

		/*! \return plasticity status for a synapse */
		unsigned char getSynapsePlastic(const synapse_id&) const;

		/*! \copydoc nemo::ReadableNetwork::getSynapsesFrom */
		const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

		/*! \copydoc nemo::network::Generator::maxDelay */
		unsigned maxDelay() const;

		float maxWeight() const;
		float minWeight() const;

		/*! \copydoc nemo::network::Generator::maxDelay */
		unsigned neuronCount() const;

	private :

		friend SimulationBackend* simulationBackend(const Network&, const Configuration&);
		friend class nemo::mpi::Master;

		class network::NetworkImpl* m_impl;

		// undefined
		Network(const Network&);
		Network& operator=(const Network&);

		/* hack for backwards-compatability with original construction API */
		unsigned iz_type;
};


} // end namespace nemo

#endif
