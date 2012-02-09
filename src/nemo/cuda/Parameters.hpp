#ifndef NEMO_CUDA_PARAMETERS_HPP
#define NEMO_CUDA_PARAMETERS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_ptr.hpp>

#include "parameters.cu_h"

namespace nemo {
	namespace cuda {

		class Simulation;


/* Kernel-specific parameters */
class Parameters 
{
	public :

		/*! Initialise the simulation-wide parameters
		 *
		 * All kernels use a single pitch for all 64-, 32-, and 1-bit
		 * per-neuron data This function sets these common pitches and also
		 * checks that all relevant arrays have the same pitch.
		 *
		 * \param fbits number of fractional bits
		 * \param pitch1 pitch of 1-bit per-neuron data
		 * \param pitch32 pitch of 32-bit per-neuron data
		 * \param maxDelay maximum delay found in the network
		 */
		Parameters(
				const Simulation& sim,
				unsigned fbits, size_t pitch1,
				size_t pitch32, unsigned maxDelay);

		/* Create a copy, with no associated device memory */
		Parameters(const Parameters&);

		void copyToDevice();

		void setInputs(const std::vector<unsigned>& inputs);

		param_t* d_data() const { return md_params.get(); }

	private :

		const Parameters& operator=(const Parameters&);

		Parameters();

		param_t mh_params;

		boost::shared_ptr<param_t> md_params;
};



/*! Verify device memory pitch
 *
 * On the device a number of arrays have exactly the same shape. These share a
 * common pitch parameter. This function verifies that the memory allocation
 * does what we expect.
 *
 * \throws nemo::exception if there is a mismatch
 */
void
checkPitch(size_t found, size_t expected);

	}

}

#endif
