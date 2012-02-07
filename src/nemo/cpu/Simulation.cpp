#include "Simulation.hpp"

#include <boost/format.hpp>

#ifdef NEMO_CPU_OPENMP_ENABLED
#include <omp.h>
#endif

#include <nemo/internals.hpp>
#include <nemo/exception.hpp>
#include <nemo/ConnectivityMatrix.hpp>



namespace nemo {
	namespace cpu {


Simulation::Simulation(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf) :
	m_neuronCount(net.neuronCount()),
	m_fractionalBits(conf.fractionalBits()),
	m_fired(m_neuronCount, 0),
	m_recentFiring(m_neuronCount, 0),
	m_currentExt(m_neuronCount, 0.0f),
	m_fstim(m_neuronCount, 0)
{
	using boost::format;

	if(net.maxDelay() > 64) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("The network has synapses with delay %ums. The CPU backend supports a maximum of 64 ms")
						% net.maxDelay()));
	}

	/* Contigous local neuron indices */
	nidx_t l_idx = 0;

	for(unsigned type_id=0, id_end=net.neuronTypeCount(); type_id < id_end; ++type_id) {

		/* Wrap in smart pointer to ensure the class is not copied */
		m_mapper.insertTypeBase(type_id, l_idx);

		if(net.neuronCount(type_id) == 0) {
			continue;
		}

		boost::shared_ptr<Neurons> ns(new Neurons(net, type_id, m_mapper));
		l_idx += ns->size();
		m_neurons.push_back(ns);
	}

	for(unsigned typeIdx=0; typeIdx < net.synapseTypeCount(); ++typeIdx) {
		m_cm.push_back(cm_t(new nemo::ConnectivityMatrix(net, conf, m_mapper, typeIdx)));
		m_accumulator.push_back(std::vector<float>(m_neuronCount, 0.0f));
	}

	for(unsigned n=0, n_end=net.neuronTypeCount(); n < n_end; ++n) {
		std::vector<float*> ptrs;
		const std::vector<unsigned>& inputs = net.neuronInputs(n);
		for(std::vector<unsigned>::const_iterator i = inputs.begin();
				i != inputs.end(); ++i) {
			ptrs.push_back(&m_accumulator[*i][0]);
		}
		m_accumulatorPointers.push_back(ptrs);
	}

	resetTimer();
}



unsigned
Simulation::getFractionalBits() const
{
	return m_fractionalBits;
}



void
Simulation::fire()
{
	deliverSpikes();

	unsigned ngIdx = 0;
	for(neuron_groups::const_iterator ng = m_neurons.begin();
			ng != m_neurons.end(); ++ng, ++ngIdx) {

		//! \todo deal with use of the RCM here.
		(*ng)->update(
			m_timer.elapsedSimulation(), getFractionalBits(),
			&m_accumulatorPointers[ngIdx][0],
			&m_currentExt[0],
			&m_fstim[0], &m_recentFiring[0], &m_fired[0],
			NULL
#warning "Kuramoto plugin will not work"
			//const_cast<void*>(static_cast<const void*>(m_cm->rcm()))
			);
	}

#ifdef NEMO_STDP_ENABLED
	//! \todo do this in the postfire step
	m_cm->accumulateStdp(m_recentFiring);
#endif
	setFiring();
	m_timer.step();
}



#ifdef NEMO_BRIAN_ENABLED
float*
Simulation::propagate(unsigned synapseTypeIdx, uint32_t* fired, int nfired)
{
	//! \todo assert that STDP is not enabled

	/* convert the input firing to the format required by deliverSpikes */
#pragma omp parallel for default(shared)
	for(unsigned n=0; n <= m_mapper.maxGlobalIdx(); ++n) {
		m_recentFiring[n] <<= 1;
	}

#pragma omp parallel for default(shared)
	for(int i=0; i < nfired; ++i) {
		uint32_t n = fired[i];
		m_recentFiring[n] |= uint64_t(1);
	}

	//! \todo error handling
	m_cm.at(synapseTypeIdx)->deliverSpikes(elapsedSimulation(), m_recentFiring, m_accumulator.at(synapseTypeIdx));
	m_timer.step();

	return &m_accumulator[synapseTypeIdx][0];
}
#endif


void
Simulation::setFiringStimulus(const std::vector<unsigned>& fstim)
{
	for(std::vector<unsigned>::const_iterator i = fstim.begin();
			i != fstim.end(); ++i) {
		m_fstim.at(m_mapper.localIdx(*i)) = 1;
	}
}



void
Simulation::initCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
}



void
Simulation::addCurrentStimulus(nidx_t neuron, float current)
{
	m_currentExt[m_mapper.localIdx(neuron)] = current;
}



void
Simulation::finalizeCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
}



void
Simulation::setCurrentStimulus(const std::vector<float>& current)
{
	if(m_currentExt.empty()) {
		//! do we need to clear current?
		return;
	}
	/*! \todo We need to deal with the mapping from global to local neuron
	 * indices. Before doing this, we should probably change the interface
	 * here. Note that this function is only used internally (see mpi::Worker),
	 * so we might be able to use the existing interface, and make sure that we
	 * only use local indices. */
	throw nemo::exception(NEMO_API_UNSUPPORTED, "setting current stimulus vector not supported for CPU backend");
#if 0
	if(current.size() != m_current.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "current stimulus vector not of expected size");
	}
	m_current = current;
#endif
}



//! \todo use per-thread buffers and just copy these in bulk
void
Simulation::setFiring()
{
	m_firingBuffer.enqueueCycle();
	for(unsigned n=0; n < m_neuronCount; ++n) {
		if(m_fired[n]) {
			m_firingBuffer.addFiredNeuron(m_mapper.globalIdx(n));
		}
	}
}



FiredList
Simulation::readFiring()
{
	return m_firingBuffer.dequeueCycle();
}



void
Simulation::setNeuron(unsigned g_idx, unsigned nargs, const float args[])
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->set(l_idx, nargs, args);
}



void
Simulation::setNeuronState(unsigned g_idx, unsigned var, float val)
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->setState(l_idx, var, val);
}



void
Simulation::setNeuronParameter(unsigned g_idx, unsigned parameter, float val)
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->setParameter(l_idx, parameter, val);
}



void
Simulation::applyStdp(float reward)
{
#ifdef NEMO_STDP_ENABLED
	m_cm->applyStdp(reward);
#else
	throw nemo::exception(NEMO_API_UNSUPPORTED, "This version of NeMo does not support STDP");
#endif
}



void
Simulation::deliverSpikes()
{
	size_t i = 0;
	for(std::vector<cm_t>::iterator cm = m_cm.begin();
			cm != m_cm.end(); ++cm, ++i) {
		(*cm)->deliverSpikes(elapsedSimulation(),
				m_recentFiring,
				m_accumulator.at(i));
	}
}





float
Simulation::getNeuronState(unsigned g_idx, unsigned var) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getState(l_idx, var);
}



float
Simulation::getNeuronParameter(unsigned g_idx, unsigned param) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getParameter(l_idx, param);
}


float
Simulation::getMembranePotential(unsigned g_idx) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getMembranePotential(l_idx);
}



const std::vector<synapse_id>&
Simulation::getSynapsesFrom(unsigned neuron)
{
#warning "Unsupported function"
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Unsupported function");
	//! \todo combine the synapses from different neurons
	// return m_cm->getSynapsesFrom(neuron);
}



unsigned
Simulation::getSynapseTarget(const synapse_id& synapse) const
{
#warning "Unsupported function"
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Unsupported function");
	// return m_cm->getTarget(synapse);
}



unsigned
Simulation::getSynapseDelay(const synapse_id& synapse) const
{
#warning "Unsupported function"
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Unsupported function");
	//return m_cm->getDelay(synapse);
}



float
Simulation::getSynapseWeight(const synapse_id& synapse) const
{
#warning "Unsupported function"
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Unsupported function");
	//return m_cm->getWeight(synapse);
}



unsigned char
Simulation::getSynapsePlastic(const synapse_id& synapse) const
{
#warning "Unsupported function"
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Unsupported function");
	//return m_cm->getPlastic(synapse);
}



unsigned long
Simulation::elapsedWallclock() const
{
	return m_timer.elapsedWallclock();
}



unsigned long
Simulation::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}



void
Simulation::resetTimer()
{
	m_timer.reset();
}



const char*
deviceDescription()
{
	/* Store a static string here so we can safely pass a char* rather than a
	 * string object across DLL interface */
#ifdef NEMO_CPU_OPENMP_ENABLED
	using boost::format;
	static std::string descr = str(format("CPU backend (OpenMP, %u cores)") % omp_get_num_procs());
#else
	static std::string descr("CPU backend (single-threaded)");
#endif
	return descr.c_str();
}


	} // namespace cpu
} // namespace nemo
