#ifndef NEMO_CPU_SIMULATION_HPP
#define NEMO_CPU_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <vector>

#include <nemo/config.h>
#include <nemo/internal_types.h>
#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/Timer.hpp>
#include <nemo/FiringBuffer.hpp>
#include <nemo/RNG.hpp>

#include "Worker.hpp"
#include "Mapper.hpp"


namespace nemo {
	namespace cpu {

class NEMO_CPU_DLL_PUBLIC Simulation : public nemo::SimulationBackend
{
	public:

		Simulation(const network::Generator&, const nemo::ConfigurationImpl&);

		unsigned getFractionalBits() const;

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::initCurrentStimulus */
		void initCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::addCurrentStimulus */
		void addCurrentStimulus(nidx_t neuron, float current);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void finalizeCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::update */
		void update();

		/*! \copydoc nemo::SimulationBackend::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::SimulationBackend::readFiring */
		FiredList readFiring();

		/*! \copydoc nemo::Simulation::getMembranePotential */
		float getMembranePotential(unsigned neuron) const;

		/*! \copydoc nemo::Simulation::getTargets */
		const std::vector<unsigned>& getTargets(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getDelays */
		const std::vector<unsigned>& getDelays(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getWeights */
		const std::vector<float>& getWeights(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getPlastic */
		const std::vector<unsigned char>& getPlastic(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

		/*! \copydoc nemo::SimulationBackend::mapper */
		virtual Mapper& mapper() { return m_mapper; }

	private:

		Mapper m_mapper;

		typedef std::vector<fix_t> current_vector_t;
		typedef std::vector<unsigned> stimulus_vector_t;

		size_t m_neuronCount;

		/* At run-time data is put into regular vectors for vectorizable
		 * operations */
		//! \todo enforce 16-byte allignment to support vectorisation
		std::vector<float> m_a;
		std::vector<float> m_b;
		std::vector<float> m_c;
		std::vector<float> m_d;

		std::vector<float> m_u;
		std::vector<float> m_v;
		std::vector<float> m_sigma;

		/* Not all neuron indices may correspond to actual neurons. At run-time
		 * this is read-only. */
		//! \todo consider *not* using vector<bool> due to the odd optimisations which are done to it
		std::vector<bool> m_valid;

		/* last cycles firing, one entry per neuron */
		std::vector<unsigned> m_fired;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* bit-mask containing delays at which neuron has *any* outoing
		 * synapses */
		std::vector<uint64_t> m_delays;

		/* Set all neuron parameters from input network in
		 * local data structures. Also add valid neuron
		 * indices to the mapper as a side effect.  */
		void setNeuronParameters(const network::Generator& net, Mapper& mapper);

		/*! Update state of all neurons */
		void update(const stimulus_vector_t&, const current_vector_t&);

		nemo::ConnectivityMatrix m_cm;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fix_t> m_current;

		/*! Deliver spikes due for delivery */
		current_vector_t& deliverSpikes();

		/* firing stimulus (for a single cycle) */
		stimulus_vector_t m_fstim;

		//! \todo may want to have one rng per neuron or at least per thread
		std::vector<nemo::RNG> m_rng;

		void setFiring();

		FiringBuffer m_firingBuffer;

#ifdef NEMO_CPU_MULTITHREADED

		std::vector<Worker> m_workers;

		void initWorkers(size_t neurons, unsigned threads);

		friend class Worker;
#endif

		void updateRange(int begin, int end);

		void deliverSpikesOne(nidx_t source, delay_t delay);

		Timer m_timer;
};



/* If threadCount is -1, use default values */
NEMO_CPU_DLL_PUBLIC
void chooseHardwareConfiguration(nemo::ConfigurationImpl&, int threadCount = -1);

	} // namespace cpu
} // namespace nemo


#endif
