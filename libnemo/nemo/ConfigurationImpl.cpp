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
	m_cudaPartitionSize(0),     // set by interface class
	m_cudaFiringBufferLength(0) // ditto
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



void
ConfigurationImpl::setFractionalBits(unsigned bits)
{
	using boost::format;

	const unsigned max_bits = 32;
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



} // namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf)
{
	return o
		<< "STDP=" << conf.stdpFunction().enabled() << " "
		<< "cuda_ps=" << conf.cudaPartitionSize();
	//! \todo print more info about STDP
}