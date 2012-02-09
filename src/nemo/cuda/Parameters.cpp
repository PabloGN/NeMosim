/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>

#include "Parameters.hpp"
#include "Simulation.hpp"

namespace nemo {
	namespace cuda {


Parameters::Parameters(
		const Simulation& sim,
		unsigned fbits,
		size_t pitch1,
		size_t pitch32,
		unsigned maxDelay)
{
	/* Need a max of at least 1 in order for local queue to be non-empty */
	mh_params.maxDelay = std::max(1U, maxDelay);
	mh_params.pitch1 = pitch1;
	mh_params.pitch32 = pitch32;
	mh_params.pitch64 = sim.m_recentFiring.wordPitch();
	checkPitch(sim.m_currentStimulus.wordPitch(), mh_params.pitch32);
	checkPitch(sim.m_firingBuffer.wordPitch(), mh_params.pitch1);
	checkPitch(sim.m_firingStimulus.wordPitch(), mh_params.pitch1);;

	mh_params.fixedPointScale = 1 << fbits;
	mh_params.fixedPointFractionalBits = fbits;
}



Parameters::Parameters(const Parameters& rhs)
{
	mh_params = rhs.mh_params;
}



void
Parameters::setInputs(const std::vector<unsigned>& inputs)
{
	if(inputs.size() > MAX_NEURON_INPUTS) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"The CUDA backend supports no more than MAX_NEURON_INPUTS input types for each neuron type");
	}
	std::copy(inputs.begin(), inputs.end(), mh_params.inputs);
}



/*! Copy parameters from host to device */
void
Parameters::copyToDevice()
{
	void* d_ptr;
	d_malloc(&d_ptr, sizeof(param_t), "Global parameters");
	md_params = boost::shared_ptr<param_t>(static_cast<param_t*>(d_ptr), d_free);
	memcpyBytesToDevice(d_ptr, &mh_params, sizeof(param_t));
}



void
checkPitch(size_t found, size_t expected)
{
	using boost::format;
	if(found != 0 && expected != found) {
		throw nemo::exception(NEMO_CUDA_MEMORY_ERROR,
				str(format("Pitch mismatch in device memory allocation. Found %u, expected %u")
					% found % expected));
	}
}


	}
}
