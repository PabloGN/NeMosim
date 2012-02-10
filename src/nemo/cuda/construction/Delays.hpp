#ifndef NEMO_CUDA_CONSTRUCTION_DELAYS_HPP
#define NEMO_CUDA_CONSTRUCTION_DELAYS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <nemo/cuda/types.h>

namespace nemo {
	namespace cuda {
		
		namespace runtime {
			class Delays;
		}
		namespace construction {

		class FcmIndex;


/*! Mapping from neurons to delays
 *
 * This is the temporary construction-time data structure. It is constructed
 * incrementally as it forms a set union of the mappings found in each of
 * potentially several connectivity matrices (for different synapse types).
 *
 * \see nemo::cuda::runtime::Delays
 *
 * \author Andreas K. Fidjeland
 */

class Delays
{
	public :

		Delays(unsigned partitionCount);
		
		void insert(const construction::FcmIndex& index);

	private :

		friend class nemo::cuda::runtime::Delays;

		/* Number of words in each row of data */
		size_t m_width;
		size_t m_height;

		std::vector<delay_dt> m_data;
		std::vector<unsigned> m_fill;
};

}	}	} // end namespaces

#endif
