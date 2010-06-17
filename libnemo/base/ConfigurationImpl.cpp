/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConfigurationImpl.hpp"
#include <CudaSimulation.hpp>

namespace nemo {

ConfigurationImpl::ConfigurationImpl() :
	m_logging(false),
	m_cudaPartitionSize(cuda::Simulation::defaultPartitionSize()),
	m_cudaFiringBufferLength(cuda::Simulation::defaultFiringBufferLength())
{
	;
}

void
ConfigurationImpl::setStdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight)
{
	m_stdpFn = STDP<float>(prefire, postfire, minWeight, maxWeight);
}



int
ConfigurationImpl::setCudaDevice(int dev)
{
	return cuda::Simulation::setDevice(dev);
}



} // namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf)
{
	return o
		<< "STDP=" << conf.stdpFunction().enabled() << " "
		<< "cuda_ps=" << conf.cudaPartitionSize();
	//! \todo print more info about STDP
}