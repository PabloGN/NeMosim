#ifndef NEMO_MPI_MAPPER_HPP
#define NEMO_MPI_MAPPER_HPP

#include <set>

#include <nemo/internal_types.h>
#include <nemo/Mapper.hpp>

namespace nemo {
	namespace mpi {

/* Translate between global and rank/local indices
 *
 * Each neuron is processed on a single node. The index of a neuron can thus be
 * specified either in a global index or with a rank/local index pair. This
 * class performs that mapping.
 */
class Mapper : public nemo::Mapper<nidx_t, nidx_t>
{
	public:

		Mapper(unsigned neurons, unsigned workers, int rank);

		/*! \return the rank of the process which should process a particular neuron */ 
		int rankOf(nidx_t) const;

		nidx_t localIdx(const nidx_t& global) const;

		nidx_t globalIdx(const nidx_t& local) const;

		nidx_t addGlobal(const nidx_t& global);

		bool validGlobal(const nidx_t& global) const;

		bool validLocal(const nidx_t& local) const;

		unsigned neuronCount() const { return localCount(); }

		nidx_t maxLocalIdx() const;

		//! \todo tidy by folding into neuronCount
		/*! \return number of neurons handled locally */
		unsigned localCount() const;

	private:

		/* Number of neurons processed on each node */
		int m_nodeSize;

		/* Total number of nodes */
		int m_workers;

		int m_rank;

		/* First index dealt with in *this* node */
		nidx_t m_startIdx;

		std::set<nidx_t> m_validGlobal;

};


	} // end namespace mpi
} // end namespace nemo

#endif
