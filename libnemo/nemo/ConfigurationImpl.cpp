/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConfigurationImpl.hpp"

#include <boost/format.hpp>

#include "exception.hpp"


namespace nemo {

ConfigurationImpl::ConfigurationImpl() :
	m_logging(false),
	m_fractionalBits(s_defaultFractionalBits),
	// m_cpuThreadCount(0), // set properly by interface class ctor
	m_cudaPartitionSize(0),
	m_cudaFiringBufferLength(0),
	m_cudaDevice(-1),
	m_backend(-1), // the wrapper class will set this
	m_backendDescription("No backend specified")
{
	;
}



void
ConfigurationImpl::setCpuThreadCount(unsigned threadCount)
{
	if(threadCount < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Attempt to set number of threads < 1");
	}
	m_cpuThreadCount = threadCount;
}



void
ConfigurationImpl::setStdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight)
{
	m_stdpFn = StdpFunction(prefire, postfire, minWeight, maxWeight);
}



void
ConfigurationImpl::setFractionalBits(unsigned bits)
{
	using boost::format;

	const unsigned max_bits = 31;
	if(bits > max_bits) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid number of fractional bits (%u) specified. Max is %u")
					% bits % max_bits));
	}

	m_fractionalBits = static_cast<int>(bits);
}



unsigned
ConfigurationImpl::fractionalBits() const
{
	if(!fractionalBitsSet()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Fractional bits requested but never set");
	}
	return static_cast<unsigned>(m_fractionalBits);
}



bool
ConfigurationImpl::fractionalBitsSet() const
{
	return m_fractionalBits != s_defaultFractionalBits;
}



void
ConfigurationImpl::setBackend(backend_t backend)
{
	switch(backend) {
		case NEMO_BACKEND_CUDA :
		case NEMO_BACKEND_CPU :
			m_backend = backend;
			break;
		default :
			throw std::runtime_error("Invalid backend selected");
	}
}


} // namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf)
{
	return o
		<< "STDP: " << conf.stdpFunction() << ", "
		<< "cuda_ps: " << conf.cudaPartitionSize() << ", "
		<< "device: " << conf.backendDescription();
	//! \todo print more info about STDP
}
