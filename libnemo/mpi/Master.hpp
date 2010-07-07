#ifndef NEMO_MPI_MASTER_HPP
#define NEMO_MPI_MASTER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <deque>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <nemo/types.hpp>

#include "Mapper.hpp"

namespace nemo {

	class Network;
	class NetworkImpl;
	class Configuration;

	namespace mpi {


class Master
{
	public :

		Master( boost::mpi::environment& env,
				boost::mpi::communicator& world,
				const Network&, 
				const Configuration&);

		~Master();

		void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());

		/* Return reference to first buffered cycle's worth of firing. The
		 * reference is invalidated by any further calls to readFiring, or to
		 * step. */
		const std::vector<unsigned>& readFiring();

	private :

		boost::mpi::communicator m_world;

		Mapper m_mapper;

		void distributeNetwork(const Mapper&, const nemo::NetworkImpl* net);

		unsigned workers() const;

		void terminate();

		std::deque< std::vector<unsigned> > m_firing;
};

	} // end namespace mpi
} // end namespace nemo

#endif
