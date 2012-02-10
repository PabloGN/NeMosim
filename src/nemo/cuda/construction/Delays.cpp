/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Delays.hpp"
#include "FcmIndex.hpp"

#include <nemo/util.h>
#include <nemo/cuda/kernel.cu_h>

namespace nemo {
	namespace cuda {
		namespace construction {


Delays::Delays(unsigned partitionCount) :
	m_width(ALIGN(MAX_DELAY, 32)),
	m_height(partitionCount * MAX_PARTITION_SIZE),
	m_data(m_height * m_width),
	m_fill(m_height, 0U)
{
	;
}


void
Delays::insert(const construction::FcmIndex& index)
{
	using namespace boost::tuples;

	/* Populate */
	for(construction::FcmIndex::iterator i = index.begin(); i != index.end(); ++i) {
		const construction::FcmIndex::index_key& k = i->first;
		pidx_t p  = get<0>(k);
		nidx_t nl = get<1>(k);
		nidx_t n = p * MAX_PARTITION_SIZE + nl;
		delay_t delay1 = get<2>(k);
		m_data[n * m_width + m_fill[n]] = delay1-1;
		m_fill[n] += 1;
	}
}


}	}	} // end namespaces
