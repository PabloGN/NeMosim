#ifndef NEMO_SIMULATION_HPP
#define NEMO_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <utility>
#include <nemo/config.h>
#include <nemo/types.h>

namespace nemo {

class Network;
class Configuration;


/*! \class Simulation
 *
 * \brief Simulation of a single network
 *
 * Concrete instances are created using the \a nemo::simulation factory
 * function.
 *
 * Internal errors are signaled by exceptions. Thrown exceptions are all
 * of the type \a nemo::exception which in turn subclass std::exception.
 *
 * \ingroup cpp-api
 */
class NEMO_BASE_DLL_PUBLIC Simulation
{
	public :

		virtual ~Simulation();

		typedef std::vector<unsigned> firing_output;
		typedef std::vector<unsigned> firing_stimulus;
		typedef std::vector< std::pair<unsigned, float> > current_stimulus;

		/*! Run simulation for a single cycle (1ms) without external stimulus */
		virtual const firing_output& step() = 0;

		/*! Run simulation for a single cycle (1ms) with firing stimulus
		 *
		 * \param fstim
		 * 		An list of neurons, which will be forced to fire this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(const firing_stimulus& fstim) = 0;

		/*! Run simulation for a single cycle (1ms) with current stimulus
		 *
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(const current_stimulus& istim) = 0;

		/*! Run simulation for a single cycle (1ms) with both firing stimulus
		 * and current stimulus
		 *
		 * \param fstim
		 * 		An list of neurons, which will be forced to fire this cycle.
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(
					const firing_stimulus& fstim,
					const current_stimulus& istim) = 0;

		/*! \name Modifying the network
		 *
		 * Neuron parameters and state variables can be modified during
		 * simulation. However, synapses can not be modified during simulation
		 * in the current version of NeMo
		 */

		/*! Change the parameters of an existing neuron.
		 *
		 * \see nemo::Network::addNeuron for parameters
		 */
		virtual void setNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma) = 0;

		/*! \copydoc nemo::Network::setNeuronState */
		virtual void setNeuronState(unsigned neuron, unsigned var, float val) = 0;

		/*! \copydoc nemo::Network::setNeuronParameter */
		virtual void setNeuronParameter(unsigned neuron, unsigned param, float val) = 0;

		/*! Update synapse weights using the accumulated STDP statistics
		 *
		 * \param reward
		 * 		Multiplier for the accumulated weight change
		 */
		virtual void applyStdp(float reward) = 0;

		/*! \name Queries
		 *
		 * Neuron and synapse state is availble at run-time.
		 *
		 * The synapse state can be read back at run-time by specifiying a list
		 * of synpase ids (see \a addSynapse). The weights may change at
		 * run-time, while the other synapse data is static.
		 *
		 * \{ */

		/*! \copydoc nemo::Network::getNeuronState */
		virtual float getNeuronState(unsigned neuron, unsigned var) const = 0;

		/*! \copydoc nemo::Network::getNeuronParameter */
		virtual float getNeuronParameter(unsigned neuron, unsigned param) const = 0;

		/*! \return
		 * 		membrane potential of the specified neuron
		 */
		virtual float getMembranePotential(unsigned neuron) const = 0;

		/*! \return
		 * 		vector of synapse ids for all synapses with the given source
		 * 		neuron
		 */
		virtual const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron) = 0;

		/*! \return
		 * 		target neurons for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		virtual const std::vector<unsigned>& getTargets(const std::vector<synapse_id>& synapses) = 0;

		/*! \return
		 * 		conductance delays for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		virtual const std::vector<unsigned>& getDelays(const std::vector<synapse_id>& synapses) = 0;

		/*! \return
		 * 		synaptic weights for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		virtual const std::vector<float>& getWeights(const std::vector<synapse_id>& synapses) = 0;

		/*! \return
		 * 		plasticity status for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		virtual const std::vector<unsigned char>& getPlastic(const std::vector<synapse_id>& synapses) = 0;

		/* \} */ // end simulation (queries) section

		/*! \name Simulation (timing)
		 *
		 * The simulation has two internal timers which keep track of the
		 * elapsed \e simulated time and \e wallclock time. Both timers measure
		 * from the first simulation step, or from the last timer reset,
		 * whichever comes last.
		 *
		 * \{ */

		/*! \return number of milliseconds of wall-clock time elapsed since
		 * first simulation step (or last timer reset). */
		virtual unsigned long elapsedWallclock() const = 0;

		/*! \return number of milliseconds of simulated time elapsed since first
		 * simulation step (or last timer reset) */
		virtual unsigned long elapsedSimulation() const = 0;

		/*! Reset both wall-clock and simulation timer */
		virtual void resetTimer() = 0;

		/* \} */ // end simulation (timing) section

	protected :

		Simulation() { };

	private :

		/* Disallow copying of Simulation object */
		Simulation(const Simulation&);
		Simulation& operator=(const Simulation&);

};

};

#endif