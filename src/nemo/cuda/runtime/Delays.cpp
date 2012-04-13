/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Delays.hpp"
#include <nemo/cuda/device_memory.hpp>
#include <nemo/cuda/construction/Delays.hpp>

namespace nemo {
	namespace cuda {
		namespace runtime {


Delays::Delays(const construction::Delays& h_delays) :
	mb_allocated(0)
{
	size_t height = h_delays.m_height;
	size_t width = h_delays.m_width;

	/* Allocate device data for fill */
	md_fill = d_array<unsigned>(height, true, "delays fill");
	mb_allocated += height * sizeof(unsigned);
	memcpyToDevice(md_fill.get(), h_delays.m_fill);
	
	/* Allocate device data for data proper */
	md_data = d_array<delay_dt>(height*width, true, "delays");
	mb_allocated += height * width * sizeof(delay_dt);
	memcpyToDevice(md_data.get(), h_delays.m_data);
}


}	}	} // end namespace
