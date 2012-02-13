/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NetworkImpl.hpp"

#include <limits>
#include <boost/format.hpp>

#include <nemo/bitops.h>
#include <nemo/network/programmatic/neuron_iterator.hpp>
#include <nemo/network/programmatic/synapse_iterator.hpp>
#include "exception.hpp"
#include "synapse_indices.hpp"

namespace nemo {
	namespace network {


NetworkImpl::NetworkImpl() :
	m_minIdx(std::numeric_limits<int>::max()),
	m_maxIdx(std::numeric_limits<int>::min()),
	m_maxDelay(0),
	m_minWeight(0),
	m_maxWeight(0)
{
	;
}



unsigned
NetworkImpl::addNeuronType(
		const std::string& name,
		unsigned nInputs,
		const unsigned inputs[])
{
	using boost::format;

	for(unsigned s=0; s < nInputs; ++s) {
		if(inputs[s] >= m_synapses.size()) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse id (%u) used as input to neuron type %s")
						% inputs[s] % name));
		}
	}

	unsigned type_id = m_neurons.size();

	if(type_id > 1000) {
		/* There's no reason why in principle it shouldn't work to have an
		 * arbitrary number of neurons. However, a very large number of neurons
		 * is indicative of a user error where the neuron type is added once
		 * for each neuron */
		throw nemo::exception(NEMO_INVALID_INPUT, "Only 1000 neuron types supported");
	}

	m_neurons.push_back(Neurons(
				NeuronType(name, nInputs),
				std::vector<unsigned>(inputs, inputs+nInputs)));
	return type_id;
}



unsigned
NetworkImpl::addSynapseType(const synapse_type& type)
{
	if(type != NEMO_SYNAPSE_ADDITIVE) {
		throw nemo::exception(NEMO_API_UNSUPPORTED, "This version of NeMo only supports simple additive synapses");
	}
	m_synapses.push_back(type);
	m_fcm.push_back(fcm_t());
	return m_synapses.size() - 1;
}



const NeuronType&
NetworkImpl::neuronType(unsigned id) const
{
	return neuronCollection(id).type();
}



const std::vector<unsigned>&
NetworkImpl::neuronInputs(unsigned id) const
{
	return neuronCollection(id).inputs();
}



const Neurons&
NetworkImpl::neuronCollection(unsigned type_id) const
{
	using boost::format;
	if(type_id >= m_neurons.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron type id %u") % type_id));
	}
	return m_neurons[type_id];
}



Neurons&
NetworkImpl::neuronCollection(unsigned type_id)
{
	return const_cast<Neurons&>(static_cast<const NetworkImpl&>(*this).neuronCollection(type_id));
}



void
NetworkImpl::addNeuron(unsigned type_id, unsigned g_idx,
		unsigned nargs, const float args[])
{
	m_maxIdx = std::max(m_maxIdx, int(g_idx));
	m_minIdx = std::min(m_minIdx, int(g_idx));
	unsigned l_nidx = neuronCollection(type_id).add(g_idx, nargs, args);
	m_mapper.insert(g_idx, NeuronAddress(type_id, l_nidx));
}



void
NetworkImpl::setNeuron(unsigned nidx, unsigned nargs, const float args[])
{
	const NeuronAddress& addr = m_mapper.localIdx(nidx);
	neuronCollection(addr.first).set(addr.second, nargs, args);
}



synapse_id
NetworkImpl::addSynapse(
		unsigned typeIdx,
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight)
{
	using boost::format;

	if(delay < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid delay (%u) for synapse between neurons %u and %u")
					% delay
					% source
					% target));
	}

	/* only non-plastic synapses supported via this API function */
	try {
		id32_t id = m_fcm.at(typeIdx)[source].addSynapse(target, delay, weight, false);

		//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
		m_maxIdx = std::max(m_maxIdx, int(std::max(source, target)));
		m_minIdx = std::min(m_minIdx, int(std::min(source, target)));
		m_maxDelay = std::max(m_maxDelay, delay);
		m_maxWeight = std::max(m_maxWeight, weight);
		m_minWeight = std::min(m_minWeight, weight);

		return make_synapse_id(source, typeIdx, id);
	} catch(std::out_of_range& e) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid synapse type index (%u) for synapse between neurons %u and %u")
					% typeIdx
					% source
					% target));
	}

}



float
NetworkImpl::getNeuronState(unsigned nidx, unsigned var) const
{
	const NeuronAddress& addr = m_mapper.localIdx(nidx);
	return neuronCollection(addr.first).getState(addr.second, var);
}



float
NetworkImpl::getNeuronParameter(unsigned nidx, unsigned parameter) const
{
	const NeuronAddress& addr = m_mapper.localIdx(nidx);
	return neuronCollection(addr.first).getParameter(addr.second, parameter);
}



void
NetworkImpl::setNeuronState(unsigned nidx, unsigned var, float val)
{
	const NeuronAddress& addr = m_mapper.localIdx(nidx);
	return neuronCollection(addr.first).setState(addr.second, var, val);
}



void
NetworkImpl::setNeuronParameter(unsigned nidx, unsigned parameter, float val)
{
	const NeuronAddress& addr = m_mapper.localIdx(nidx);
	return neuronCollection(addr.first).setParameter(addr.second, parameter, val);
}



const Axon&
NetworkImpl::axon(nidx_t source, unsigned typeIdx) const
{
	using boost::format;
	const fcm_t& fcm = m_fcm.at(typeIdx);
	fcm_t::const_iterator i_src = fcm.find(source);
	if(i_src == fcm.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("synapses of non-existing neuron (%u) requested") % source));
	}
	return i_src->second;
}



unsigned
NetworkImpl::getSynapseTarget(const synapse_id& id) const
{
	return axon(neuronIndex(id), typeIndex(id)).getTarget(synapseIndex(id));
}



unsigned
NetworkImpl::getSynapseDelay(const synapse_id& id) const
{
	return axon(neuronIndex(id), typeIndex(id)).getDelay(synapseIndex(id));
}



float
NetworkImpl::getSynapseWeight(const synapse_id& id) const
{
	return axon(neuronIndex(id), typeIndex(id)).getWeight(synapseIndex(id));
}



unsigned char
NetworkImpl::getSynapsePlastic(const synapse_id& id) const
{
	return axon(neuronIndex(id), typeIndex(id)).getPlastic(synapseIndex(id));
}



const std::vector<synapse_id>&
NetworkImpl::getSynapsesFrom(unsigned source)
{
	m_queriedSynapseIds.clear();
	unsigned typeIdx = 0U;
	for(std::vector<fcm_t>::const_iterator fcm = m_fcm.begin();
			fcm != m_fcm.end(); ++fcm, ++typeIdx) {
		fcm_t::const_iterator i_src = fcm->find(source);
		if(i_src != fcm->end()) {
			i_src->second.appendSynapseIds(source, typeIdx, m_queriedSynapseIds);
		}
	}
	return m_queriedSynapseIds;
}



unsigned
NetworkImpl::neuronCount() const
{
	unsigned total = 0;
	for(std::vector<Neurons>::const_iterator i = m_neurons.begin();
			i != m_neurons.end(); ++i) {
		total += i->size();
	}
	return total;
}



unsigned
NetworkImpl::neuronCount(unsigned type_id) const
{
	return m_neurons.at(type_id).size();
}



nidx_t
NetworkImpl::minNeuronIndex() const
{
	if(neuronCount() == 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"minimum neuron index requested for empty network");
	}
	return m_minIdx;
}


nidx_t
NetworkImpl::maxNeuronIndex() const
{
	if(neuronCount() == 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"maximum neuron index requested for empty network");
	}
	return m_maxIdx;
}


/* Neuron iterators */


unsigned
NetworkImpl::neuronTypeCount() const
{
	return m_neurons.size();
}


neuron_iterator
NetworkImpl::neuron_begin(unsigned id) const
{
	if(id >= m_neurons.size()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Invalid neuron type id");
	}
	const Neurons& neurons = m_neurons.at(id);
	//! \todo fold this into Neurons
	return neuron_iterator(new programmatic::neuron_iterator(
				neurons.m_gidx.begin(),
				neurons.m_param, neurons.m_state, neurons.type()));
}


neuron_iterator
NetworkImpl::neuron_end(unsigned id) const
{
	if(id >= m_neurons.size()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Invalid neuron type id");
	}
	const Neurons& neurons = m_neurons.at(id);
	//! \todo fold this into Neurons
	return neuron_iterator(new programmatic::neuron_iterator(neurons.m_gidx.end(),
				neurons.m_param, neurons.m_state, neurons.type()));
}



unsigned
NetworkImpl::synapseTypeCount() const
{
	return m_synapses.size();
}



synapse_iterator
NetworkImpl::synapse_begin(unsigned type) const
{
	fcm_t::const_iterator ni = m_fcm.at(type).begin();
	fcm_t::const_iterator ni_end = m_fcm.at(type).end();

	size_t gi = 0;
	size_t gi_end = 0;

	if(ni != ni_end) {
		gi_end = ni->second.size();
	}
	return synapse_iterator(
		new programmatic::synapse_iterator(ni, ni_end, gi, gi_end));
}


synapse_iterator
NetworkImpl::synapse_end(unsigned type) const
{
	fcm_t::const_iterator ni = m_fcm.at(type).end();
	size_t gi = 0;

	if(m_fcm.at(type).begin() != ni) {
		gi = m_fcm.at(type).rbegin()->second.size();
	}

	return synapse_iterator(new programmatic::synapse_iterator(ni, ni, gi, gi));
}


}	}
